## pion
#python evaluate.py -i ../input/dataset1/dataset_1_pions_1.hdf5 -t ../output/dataset1/v1/BNReLUCustActiv_hpo27-M-Pnormlayer2.1 --checkpoint --save_h5 -p normlayer2
#
## photon
#
#python evaluate.py -i ../input/dataset1/dataset_1_photons_1.hdf5 -t ../output/dataset1/v1/BNswishCustActiv_hpo113-M-Pnormlayer2-L --checkpoint --save_h5 -p normlayer2
python evaluate.py -i ../input/dataset1/dataset_1_photons_1.hdf5 -t ../output/dataset1/v1/BNswishCustActiv_hpo113-M-Pnormlayer2-L-Sle12.1 --checkpoint --save_h5 -p normlayer2  --split_energy_position le12
python evaluate.py -i ../input/dataset1/dataset_1_photons_1.hdf5 -t ../output/dataset1/v1/BNswishCustActiv_hpo113-M-Pnormlayer2-L-ge12le18.1 --checkpoint --save_h5 -p normlayer2  --split_energy_position ge12le18
python evaluate.py -i ../input/dataset1/dataset_1_photons_1.hdf5 -t ../output/dataset1/v1/BNswishCustActiv_hpo113-M-Pnormlayer2-L-ge18.1 --checkpoint --save_h5 -p normlayer2  --split_energy_position ge18

python evaluate.py -i ../input/dataset1/dataset_1_photons_1.hdf5 -t ../output/dataset1/v1/BNswishCustActiv_hpo113-M-Pnormlayer2-L-Sge12 --checkpoint --save_h5 -p normlayer2  --split_energy_position ge12

python evaluate.py -i ../input/dataset1/dataset_1_photons_1.hdf5 -t ../output/dataset1/v1/BNswishCustMichele_hpo117-M-PnormlayerMichele-L-ge18.Michele --checkpoint --save_h5 -p normlayerMichele --split_energy_position ge18


#../output/dataset1/v1/BNReLUCustActiv_hpo27-M-Pnormlayer2.1/pions_eta_20_25/selected/h5/gan.h5
#
#../output/dataset1/v1/BNswishCustActiv_hpo113-M-Pnormlayer2-L/photons_eta_20_25/selected/h5/gan.h5
#../output/dataset1/v1/BNswishCustActiv_hpo113-M-Pnormlayer2-L-Sle12.1/photons_eta_20_25/selected/h5/gan.h5
#../output/dataset1/v1/BNswishCustActiv_hpo113-M-Pnormlayer2-L-ge12le18.1/photons_eta_20_25/selected/h5/gan.h5
#../output/dataset1/v1/BNswishCustActiv_hpo113-M-Pnormlayer2-L-ge18.1/photons_eta_20_25/selected/h5/gan.h5
#../output/dataset1/v1/BNswishCustActiv_hpo113-M-Pnormlayer2-L-Sge12/photons_eta_20_25/selected/h5/gan.h5
#../output/dataset1/v1/BNswishCustMichele_hpo117-M-PnormlayerMichele-L-ge18.Michele/photons_eta_20_25/selected/h5/gan.h5
