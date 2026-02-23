# use code from https://github.com/CaloChallenge/homepage/blob/main/code/HighLevelFeatures.ipynb
from argparse import ArgumentParser
from HighLevelFeatures import HighLevelFeatures
import numpy as np
import h5py, os, json
import matplotlib.pyplot as plt
from model import WGANGP
from pdb import set_trace
from common import *
from data import *
import re
np.set_printoptions(suppress=True)

def apply_mask(mask, X_train, input_file, add_noise=False):
    np.seterr(divide = 'ignore', invalid='ignore')
    event_energy_before = X_train.sum(axis=1)[:]
    if add_noise:
        # X_train is in MeV, add uniform noise of [0, 0.1keV]
        X_train += np.random.uniform(low=0, high=0.0001, size=X_train.shape)
        print('\033[92m[INFO] Add noise\033[0m', 0, 0.1, '[keV] for voxel energy')
    event_energy_before2 = X_train.sum(axis=1)[:]

    # mask too low energy to zeros
    if isinstance(mask, (int, float)):
        X_train[X_train < (mask / 1000)] = 0
    elif isinstance(mask, dict):
        # X_train is un-sorted!
        energies = get_energies(input_file)
        for k,m in mask.items():
            X_train[np.logical_and(energies == k, X_train < (m / 1000))] = 0
    else:
        raise NotImplementedError

    # plot energy change before and after masking
    event_energy_after  = X_train.sum(axis=1)[:]
    event_energy = np.concatenate([event_energy_before.reshape(-1,1), event_energy_before2.reshape(-1,1), event_energy_after.reshape(-1,1)], axis=1)

    kin, particle = get_kin(args.input_file)
    input_data = h5py.File(f'{input_file}', 'r')
    kin = filter_energy(particle, input_data['incident_energies'][:], args.split_energy_position, kin)
    categories, vector_list  = split_energy(kin, event_energy)
    fig, axes = plot_frame(categories, xlabel="Rel. change in E total", ylabel="Events")
    for index, energy in enumerate(categories):
        ax = axes[index]
        before, after = vector_list[index][:,0], vector_list[index][:,-1]
        x = 1 - np.divide(after, before, out=np.zeros_like(before), where=before!=0)
        if x.max() < 1E-4:
            high = 1E-4
        elif x.max() < 1E-3:
            high = 1E-3
        elif x.max() < 1E-2:
            high = 1E-2
        elif x.max() < 0.1:
            high = 0.1
        else:
            high = 1
        n, _, _ = ax.hist(x, bins=100, range=(0,high))
        ax.set_yscale('symlog')
        ax.set_ylim(bottom=0)
        if isinstance(mask, (int, float)):
            mask_legend = f'Mask {mask} keV\nMax {high}'
        elif isinstance(mask, dict):
            if mask[energy] < 1E3:
                mask_legend = f'Mask {mask[energy]} keV\nMax {high}'
            elif mask[energy] < 1E6:
                mask_legend = f'Mask {mask[energy]/1E3} MeV\nMax {high}'
            else:
                mask_legend = f'Mask {mask[energy]/1E6} GeV\nMax {high}'
        ax.text(0.98, 0.88, mask_legend, transform=ax.transAxes, va="top", ha="right", fontsize=15)
    ax = axes[-1]
    ax.axis("on")
    x = 1 - event_energy_after / event_energy_before
    if x.max() < 1E-4:
        high = 1E-4
    elif x.max() < 1E-3:
        high = 1E-3
    elif x.max() < 1E-2:
        high = 1E-2
    elif x.max() < 0.1:
        high = 0.1
    else:
        high = 1
    ax.hist(x, bins=100, range=(0,high))
    ax.set_yscale('symlog')
    ax.set_ylim(bottom=0)
    os.makedirs(args.output_path, exist_ok=True)
    particle = input_file.split('/')[-1].split('_')[-2][:-1]
    plt.savefig(os.path.join(args.output_path, f'mask_{particle}_{args.mask}keV.pdf'))
    print('\033[92m[INFO] Mask\033[0m', args.mask, mask, '[keV] for voxel energy')
        
    # return masked input
    return X_train

def main(args):
    # creating instance of HighLevelFeatures class to handle geometry based on binning file
    input_file = args.input_file
    particle = input_file.split('/')[-1].split('_')[-2][:-1]
    #hlf = HighLevelFeatures(particle, filename=f'{os.path.dirname(input_file)}/binning_dataset_1_{particle}s.xml')
    print('\033[92m[INFO] Run\033[0m', particle, input_file)
    
    # loading the .hdf5 datasets
    input_data = h5py.File(f'{input_file}', 'r')
    
    energies = get_energies(input_file)
    kin, particle = get_kin(input_file)
    kin = filter_energy(particle, input_data['incident_energies'][:], args.split_energy_position, kin)
    if args.label_scheme:
        label_scheme = args.label_scheme
    else:
        label_scheme = {
        'photon': 'split_at_12_18',
        'pion': 'log_ratio',
        'electron': 'log_ratio',
        }[particle]
    label_kin = kin_to_label(kin, scheme=label_scheme)
    
    X_train = filter_energy(particle, input_data['incident_energies'][:], args.split_energy_position, input_data['showers'][:])
    if args.mask is not None:
        if args.mask < 0:
            mask = list(np.unique(energies)/256 * abs(args.mask)) # E/256 * (-mask)
            mask = dict(zip(list(np.unique(energies)), mask))
        else:
            mask = args.mask
        X_train = apply_mask(mask, X_train, input_file, add_noise=args.add_noise)

    if args.preprocess is not None:
        if (re.compile("^log10.([0-9.]+)+$").match(args.preprocess) \
                or re.compile("^scale.([0-9.]+)+$").match(args.preprocess) \
                or re.compile("^slope.([0-9.]+)+$").match(args.preprocess)
            ): # log10.x, scale.x, slope
            X_train, scale = preprocessing(X_train, kin, name=args.preprocess, input_file=input_file)
        elif args.preprocess in ['concatlayer', 'normlayer1', 'normlayer2', 'normlayer3', 'normlayerMichele', 'normlayerMichele2']:
            from XMLHandler import XMLHandler
            xml = XMLHandler(particle, filename=f'{os.path.dirname(input_file)}/binning_dataset_1_{particle}s.xml')
            shower_shape = X_train.shape
            X_train = preprocessing(X_train, kin, name=args.preprocess, input_file=input_file, xml=xml)
            scale = None
            print('\033[92m[INFO] Training size enlarged with layer info\033[0m', shower_shape, '->', X_train.shape)
    else:
        X_train = preprocessing(X_train, kin, name=args.preprocess, input_file=input_file)
        scale = None

    if 'photon' in particle:
        hp_config = {
            'model': args.model if args.model else 'BNswish',
            'dmodel': 'dense',
            'G_size': 1,
            'D_size': 1,
            'optimizer': 'adam',
            'G_lr': 1E-4,
            'D_lr': 1E-4,
            'G_beta1': 0.5,
            'D_beta1': 0.5,
            'batchsize': 1024,
            'datasize': X_train.shape[0],
            'dgratio': 8,
            'latent_dim': 50,
            'lam': 3,
            'conditional_dim': label_kin.shape[1],
            'generatorLayers': [50, 100, 200],
            'nvoxels': X_train.shape[1],
            'use_bias': True,
            'label_scheme': label_scheme,
        }
    elif 'pion' in particle: # pion
        hp_config = {
            'model': args.model if args.model else 'noBN',
            'G_size': 1,
            'D_size': 1,
            'optimizer': 'adam',
            'G_lr': 1E-4,
            'D_lr': 1E-4,
            'G_beta1': 0.5,
            'D_beta1': 0.5,
            'batchsize': 1024,
            'dgratio': 5,
            'latent_dim': 50,
            'lam': 10,
            'conditional_dim': label_kin.shape[1],
            'generatorLayers': [50, 100, 200],
            'discriminatorLayers': [800, 400, 200],
            'nvoxels': X_train.shape[1],
            'use_bias': True,
            'preprocess': args.preprocess,
            'label_scheme': label_scheme,
        }
    elif 'electron': # dataset2 electron
        hp_config = {
            'model': args.model if args.model else 'BNswish',
            'G_size': 1,
            'D_size': 1,
            'optimizer': 'adam',
            'G_lr': 1E-4,
            'D_lr': 1E-4,
            'G_beta1': 0.5,
            'D_beta1': 0.5,
            'batchsize': 1024,
            'datasize': X_train.shape[0],
            'dgratio': 8,
            'latent_dim': 50,
            'lam': 3,
            'conditional_dim': label_kin.shape[1],
            'generatorLayers': [200, 400, 800],
            'discriminatorLayers': [800, 400, 200],
            'nvoxels': X_train.shape[1],
            'use_bias': True,
            'label_scheme': label_scheme,
        }
    if args.config:
        from quickstats.utils.common_utils import combine_dict
        hp_config = combine_dict(hp_config, json.load(open(args.config, 'r')))

    job_config = {
        'particle': particle+'s',
        'eta_slice': '20_25',
        'checkpoint_interval': 1000 if not args.debug else 10,
        'output': args.output_path,
        'max_iter': 4E5 if args.loading else args.max_iter,
        'cache': False,
        'loading': args.loading,
    }

    if args.preprocess in ['normlayer1', 'normlayer2', 'normlayer3', 'normlayerMichele', 'normlayerMichele2']:
        config_string = f'normlayer__{len(xml.GetRelevantLayers())}__{":".join([ str(x) for x in xml.bin_number if x > 0 ])}'
        if args.preprocess in ['normlayer3']:
            config_string += '__mergelayer'
    else:
        config_string = None
    wgan = WGANGP(job_config=job_config, hp_config=hp_config, logger=__file__, config_string=config_string)

    if scale:
        with open(f'{wgan.train_folder}/scale_{args.preprocess}.json', 'w') as fp:
            json.dump(scale, fp, indent=2)
    plot_input(args, X_train, output=wgan.train_folder)
    print('\033[92m[INFO] Training size\033[0m', X_train.shape, 'kinematic and counts:', np.unique(kin,return_counts=True))
    wgan.train(X_train, label_kin)

def plot_input(args, X_train, output):
    kin, particle = get_kin(args.input_file)
    input_data = h5py.File(f'{args.input_file}', 'r')
    kin = filter_energy(particle, input_data['incident_energies'][:], args.split_energy_position, kin)
    categories, xtrain_list = split_energy(kin, X_train)
    out_file = os.path.join(output, f'input_{particle}_{args.preprocess}.pdf')
    plot_energy_vox(categories, [xtrain_list], label_list=['Input'], nvox='all', logx=False, \
            particle=particle, output=out_file, draw_ref=False, xlabel='Energy of voxel as training input [MeV]')
    print('\033[92m[INFO] Save to\033[0m', out_file)

if __name__ == '__main__':

    """Get arguments from command line."""
    parser = ArgumentParser(description="\033[92mConfig for training.\033[0m")
    parser.add_argument('-i', '--input_file', type=str, required=False, default='', help='Training h5 file name (default: %(default)s)')
    parser.add_argument('-o', '--output_path', type=str, required=True, default='../output/dataset1/v1', help='Training h5 file path (default: %(default)s)')
    parser.add_argument('-c', '--config', type=str, required=False, default=None, help='External config file (default: %(default)s)')
    parser.add_argument('-m', '--model', type=str, required=False, default=None, help='Model name (default: %(default)s)')
    parser.add_argument('--mask', type=float, required=False, default=None, help='Mask low noisy voxels in keV (default: %(default)s)')
    parser.add_argument('--debug', required=False, action='store_true', help='Debug mode (default: %(default)s)')
    parser.add_argument('-p', '--preprocess', type=str, required=False, default=None, help='Preprocessing name (default: %(default)s)')
    parser.add_argument('-l', '--loading', type=str, required=False, default=None, help='Load model (default: %(default)s)')
    parser.add_argument('--add_noise', required=False, action='store_true', help='Add noise (default: %(default)s)')
    parser.add_argument('--label_scheme', type=str, required=False, default='log_ratio', help='Label scheme defined in common.py (default: %(default)s)')
    parser.add_argument('--split_energy_position', type=str, required=False, default='', choices=['', 'le12', 'ge12', 'ge12le18', 'ge18'], help='Energy split training (default: %(default)s)')
    parser.add_argument('--max_iter', type=int, required=False, default=1E6, help='Number of iterations (default: %(default)s)')

    args = parser.parse_args()
    main(args)
