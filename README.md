```
cd training
```

Get input data
```
mkdir ../input/dataset1/
wget https://zenodo.org/record/6368338/files/dataset_1_photons_1.hdf5 ../input/dataset1/
```

Training
```
source /afs/cern.ch/work/z/zhangr/HH4b/hh4bStat/scripts/setup.sh

python train.py    -i ../input/dataset1/dataset_1_photons_1.hdf5 -o ../output/dataset1/v1/GANv1_GANv1 -c ../config/config_GANv1.json
python evaluate.py -i ../input/dataset1/dataset_1_photons_1.hdf5 -t ../output/dataset1/v1/GANv1_GANv1
```

Best config
```
photon:
python evaluate.py -i ../input/dataset1/dataset_1_photons_1.hdf5 -t ../output/dataset1/v2/BNswish_hpo4-M1 --checkpoint --save_h5

python evaluate.py -i ../input/dataset1/dataset_1_photons_1.hdf5 -t ../output/dataset1/v1/BNswish_hpo101-M-P-L-Sge12       --checkpoint --save_h5 --split_energy_position ge12
python evaluate.py -i ../input/dataset1/dataset_1_photons_1.hdf5 -t ../output/dataset1/v1/BNswish_hpo101-M-P-L-Sge12le18.2 --checkpoint --save_h5 --split_energy_position ge12le18
python evaluate.py -i ../input/dataset1/dataset_1_photons_1.hdf5 -t ../output/dataset1/v1/BNLeakyReLU_hpo31-M-P-L-Sle12.3  --checkpoint --save_h5 --split_energy_position le12
python evaluate.py -i ../input/dataset1/dataset_1_photons_1.hdf5 -t ../output/dataset1/v1/BNswish_hpo101-M-P-L-Sge18       --checkpoint --save_h5 --split_energy_position ge18
pions:
python evaluate.py -i ../input/dataset1/dataset_1_pions_1.hdf5 -t ../output/dataset1/v2/BNReLU_hpo27-M1 --checkpoint --save_h5
```

## Integrazione qINN (PyTorch/PennyLane) nella pipeline Keras

Da questo branch e' disponibile il modello `BNReLUqINN` nel generatore FastCaloQ.

### 1) Mappatura firma input/output (plug-and-play)

Per essere compatibile con `train.py` / `evaluate.py`, il modulo qINN deve rispettare:

- input: `[batch, latent_dim + conditional_dim]`
- output: `[batch, qinn_output_dim]`

Nel file `config/config_qinn_example.json`:

- `qinn_module_kwargs.in_features` deve coincidere con `latent_dim + conditional_dim`
- `qinn_module_kwargs.out_features` deve coincidere con `qinn_output_dim`
- opzionale: `qinn_state_path` per salvare/caricare lo stato Torch del modulo qINN (default relativo alla run: `checkpoints/qinn_module_state-init.pt`)
- opzionale: `qinn_deterministic_init=true` e `qinn_init_seed=<int>` per inizializzare il qINN in modo riproducibile quando il file stato manca
- opzionale: `qinn_require_state=true` per fallire esplicitamente se `qinn_state_path` non esiste (controllo robusto in evaluate)

**Importante:** con il bridge TF->Torch i pesi Torch non sono aggiornati da Keras.
Per evitare mismatch tra train/evaluate dovuti a inizializzazioni casuali diverse, usa lo stesso `qinn_state_path`.
Per debug riproducibile senza file stato, abilita `qinn_deterministic_init`.

### 2) Configurazione base

```json
{
  "model": "BNReLUqINN",
  "latent_dim": 10,
  "conditional_dim": 0,
  "generatorLayers": [100, 64, 0],
  "qinn_module_path": "qinn_module",
  "qinn_module_class": "QINNModule",
  "qinn_module_kwargs": {
    "in_features": 10,
    "out_features": 100,
    "hidden_features": 64,
    "use_pennylane": true,
    "n_qubits": 6,
    "n_q_layers": 2,
    "q_device": "default.qubit",
    "q_diff_method": "best",
    "q_shots": null,
    "q_entanglement": "linear"
  },
  "qinn_output_dim": 100,
  "qinn_torch_device": "cpu"
}
```

### 3) Modulo qINN reale

`training/qinn_module.py` ora include un modulo ibrido:

- `use_pennylane=false`: percorso MLP classico (debug veloce)
- `use_pennylane=true`: percorso quantum con blocchi invertibili PennyLane

Cosi' puoi fare debug in modo incrementale e passare al blocco quantistico senza cambiare il bridge.

Per il debug di biettivita' (toy studies) puoi anche salvare lo stato quantistico completo per ogni blocco:

- `qinn_module_kwargs.q_capture_quantum_state=true`
- `qinn_module_kwargs.q_capture_state_kind="statevector"` oppure `"density_matrix"`
- `qinn_module_kwargs.q_capture_every_n_calls` per controllare quanto spesso aggiornare lo snapshot in memoria usato al checkpoint

Quando scatta un checkpoint TensorFlow, il bridge salva anche artifact qINN nella stessa cartella:
`.../checkpoints/qinn_module_state-<iter>.pt` e `.../checkpoints/qinn_quantum_state-<iter>.pt`.

Nota: questa opzione e' pensata per simulatori e pochi qubit; lo stato cresce esponenzialmente con `n_qubits`.

### 4) Nota importante su training

L'integrazione avviene via bridge TensorFlow->PyTorch (`tf.py_function`):

- il training Keras resta eseguibile
- il gradiente verso input del blocco qINN usa ST estimator
- i pesi interni del modulo torch non sono aggiornati direttamente da Keras

Questa modalita' e' adatta a integrazione/esecuzione e validazione iniziale.

