import importlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


class TorchQINNLayer(Layer):
    """TensorFlow->PyTorch bridge for external qINN modules.

    This layer lets FastCaloQ call a PyTorch/PennyLane model while keeping
    the surrounding generator in Keras. The torch module is loaded dynamically
    from a python module path and class name.

    Notes
    -----
    * The output is produced through ``tf.py_function``.
    * The backward pass uses a straight-through estimator (identity gradient)
      with respect to the input tensor to keep Keras training runnable.
    * Internal torch parameters are *not* optimized by Keras.
    """

    def __init__(
        self,
        module_path: str,
        module_class: str,
        module_kwargs_json: str = "{}",
        output_dim: Optional[int] = None,
        torch_device: str = "cpu",
        state_path: Optional[str] = None,
        save_state_if_missing: bool = True,
        deterministic_init: bool = False,
        init_seed: int = 11,
        require_state: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.module_path = module_path
        self.module_class = module_class
        self.module_kwargs_json = module_kwargs_json
        self.output_dim = output_dim
        self.torch_device = torch_device
        self.state_path = state_path
        self.save_state_if_missing = save_state_if_missing
        self.deterministic_init = deterministic_init
        self.init_seed = init_seed
        self.require_state = require_state

        self._torch = None
        self._torch_model = None

    def _load_torch_model(self):
        if self._torch_model is not None:
            return

        import torch  # local import to keep TF-only workflows alive

        module_kwargs: Dict[str, Any] = json.loads(self.module_kwargs_json)
        imported_module = importlib.import_module(self.module_path)
        qinn_class = getattr(imported_module, self.module_class)

        if self.deterministic_init:
            torch.manual_seed(int(self.init_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(self.init_seed))

        self._torch = torch
        self._torch_model = qinn_class(**module_kwargs).to(self.torch_device)

        if self.state_path:
            state_file = Path(self.state_path)
            if state_file.exists():
                state_dict = torch.load(str(state_file), map_location=self.torch_device)
                self._torch_model.load_state_dict(state_dict, strict=True)
            elif self.require_state:
                raise FileNotFoundError(f"qINN state file not found: {state_file}")
            elif self.save_state_if_missing:
                state_file.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self._torch_model.state_dict(), str(state_file))

        self._torch_model.eval()

    def _forward_numpy(self, x_np: np.ndarray) -> np.ndarray:
        self._load_torch_model()

        # tf.py_function may provide a read-only NumPy view. Torch warns (and may
        # behave unsafely) when building tensors from non-writable arrays, so we
        # force an owned writable copy before from_numpy.
        x_np = np.array(x_np, dtype=np.float32, copy=True)
        x_tensor = self._torch.from_numpy(x_np).to(self.torch_device)

        with self._torch.no_grad():
            y_tensor = self._torch_model(x_tensor)

        y_np = y_tensor.detach().cpu().numpy().astype(np.float32)
        return y_np

    def save_checkpoint_artifacts(self, checkpoint_dir: str, iteration: int):
        """Persist qINN artifacts alongside TF checkpoints."""
        self._load_torch_model()

        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        if hasattr(self._torch_model, "save_checkpoint_artifacts"):
            self._torch_model.save_checkpoint_artifacts(str(checkpoint_path), int(iteration))
            return

        # Fallback for generic torch modules without helper APIs.
        self._torch.save(
            self._torch_model.state_dict(),
            checkpoint_path / f"qinn_module_state-{int(iteration)}.pt",
        )

    def call(self, inputs):
        @tf.custom_gradient
        def _wrapped(x):
            y = tf.py_function(func=self._forward_numpy, inp=[x], Tout=tf.float32)
            if self.output_dim is not None:
                y.set_shape((None, self.output_dim))

            def grad(dy):
                return dy

            return y, grad

        return _wrapped(inputs)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "module_path": self.module_path,
                "module_class": self.module_class,
                "module_kwargs_json": self.module_kwargs_json,
                "output_dim": self.output_dim,
                "torch_device": self.torch_device,
                "state_path": self.state_path,
                "save_state_if_missing": self.save_state_if_missing,
                "deterministic_init": self.deterministic_init,
                "init_seed": self.init_seed,
                "require_state": self.require_state,
            }
        )
        return cfg
