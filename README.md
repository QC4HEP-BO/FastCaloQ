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
python evaluate.py -i ../input/dataset1/dataset_1_photons_1.hdf5 -t ../output/dataset1/v2/BNswish_hpo4-M1
pions:
python evaluate.py -i ../input/dataset1/dataset_1_pions_1.hdf5 -t ../output/dataset1/v3/BNReLU_hpo27-M1
```
