## Train CaloChallenge dataset
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
## setup a conda environment
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

## Train ATLAS samples
ATLAS users could use ATLAS samples.
Here is an example of a single eta slice 0.2 < |eta| < 0.25 photon sample.:
```
/eos/atlas/atlascerngroupdisk/proj-simul/AF4/Development/FrozenShowerInputSamples/eta_020_binnings/eta_020_very_coarse_binning_remove_phi_mod/dataset_eta_020_positive.hdf5
```
To match the expected file name in the code, rename it by making a softlink at
```
mkdir -p input/dataset1/
ln -s /eos/atlas/atlascerngroupdisk/proj-simul/AF4/Development/FrozenShowerInputSamples/eta_020_binnings/eta_020_very_coarse_binning_remove_phi_mod/dataset_eta_020_positive.hdf5 input/dataset1/dataset_020_photons_positive.hdf5
```
ML models are defined in `training/model.py`.
The best model found so far is `BNswishCustMichele2Add2DenseToAllLayers`.
The following commands run this model (or other models with different names) will the desired relevant layers for photons (0, 1, 2, 3, 12) and 1 million iterations:
```
python train.py -i ../input/dataset1/dataset_020_photons_positive.hdf5 -m BNswishCustMichele2Add2DenseToAllLayers -o <output_dir> -c ../config/config_hpo8.json -p BNswishCustMichele2Add2DenseToAllLayers --relevant_layer 0 1 2 3 12 --max_iter 10000000
python evaluate.py -i ../input/dataset1/dataset_020_photons_positive.hdf5 -t <output_dir> --relevant_layer 0 1 2 3 12 --checkpoint -p BNswishCustMichele2Add2DenseToAllLayers
```
`config/config_hpo8.json` stores the hyperparameter values that were found to give good performance.
