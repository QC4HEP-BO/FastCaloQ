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

### 1) Scegliere il modello nel file di configurazione

Imposta in `hp_config`:

```json
{
  "model": "BNReLUqINN",
  "latent_dim": 10,
  "conditional_dim": 0,
  "generatorLayers": [100, 0, 0],
  "qinn_module_path": "qinn_module",
  "qinn_module_class": "QINNModule",
  "qinn_module_kwargs": {
    "in_features": 10,
    "out_features": 100,
    "hidden_features": 64,
    "use_pennylane": false
  },
  "qinn_output_dim": 100,
  "qinn_torch_device": "cpu"
}
```

### 2) Modulo qINN

Il bridge carica dinamicamente una classe PyTorch (`qinn_module.QINNModule`) e la invoca dentro al grafo Keras.
Il file `training/qinn_module.py` e' un esempio minimo, da sostituire con il modulo qINN finale.

### 3) Limitazioni attuali

- Il forward qINN viene eseguito tramite `tf.py_function`.
- Il gradiente usato verso l'input e' uno *straight-through estimator* (identita').
- I pesi interni del modulo torch non vengono aggiornati da Keras.

Questa modalita' e' utile come primo step di integrazione per esecuzione/validazione.
