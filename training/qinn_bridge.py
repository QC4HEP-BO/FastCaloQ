import importlib
import json
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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.module_path = module_path
        self.module_class = module_class
        self.module_kwargs_json = module_kwargs_json
        self.output_dim = output_dim
        self.torch_device = torch_device

        self._torch = None
        self._torch_model = None

    def _load_torch_model(self):
        if self._torch_model is not None:
            return

        import torch  # local import to keep TF-only workflows alive

        module_kwargs: Dict[str, Any] = json.loads(self.module_kwargs_json)
        imported_module = importlib.import_module(self.module_path)
        qinn_class = getattr(imported_module, self.module_class)

        self._torch = torch
        self._torch_model = qinn_class(**module_kwargs).to(self.torch_device)
        self._torch_model.eval()

    def _forward_numpy(self, x_np: np.ndarray) -> np.ndarray:
        self._load_torch_model()

        # x_np = np.asarray(x_np, dtype=np.float32)
        # tf.py_function may provide a read-only NumPy view. Torch warns (and may
        # behave unsafely) when building tensors from non-writable arrays, so we
        # force an owned writable copy before from_numpy.
        x_np = np.array(x_np, dtype=np.float32, copy=True)
        x_tensor = self._torch.from_numpy(x_np).to(self.torch_device)

        with self._torch.no_grad():
            y_tensor = self._torch_model(x_tensor)

        y_np = y_tensor.detach().cpu().numpy().astype(np.float32)
        return y_np

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
            }
        )
        return cfg
