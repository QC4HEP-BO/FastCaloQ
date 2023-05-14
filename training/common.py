import numpy as np
np.set_printoptions(suppress=True)
import h5py, os, glob
import pandas as pd
import matplotlib.pyplot as plt
from pdb import set_trace

def particle_latex_name(particle):
    return {'photon': r"$\gamma$",
            'photons': r"$\gamma$",
            'electrons': r"$e$",
            'pion': r"$\pi$",
            'pions': r"$\pi$",
            'electron': r"$e$",
            }[particle]

def get_best_mode_i(train_path, particle, eta_slice='20_25'):
    evaluate_path = os.path.join(train_path, f'{particle}*_eta_{eta_slice}', 'selected', 'model-*.index')
    models = glob.glob(evaluate_path)
    if len(models) > 1:
        print('\033[91m[WARN] Multiple selected models\033[0m', models)
    elif len(models) < 1:
        print('\033[91m[ERROR] No selected models\033[0m', evaluate_path)
        return None
    return os.path.basename(models[-1]).split('.')[0].split('-')[-1]

def particle_mass(particle=None):
    if 'photon' in particle or particle == 22:
        mass = 0
        #mass = 100
    elif 'electron' in particle or particle == 11:
        mass = 0.512
    elif 'pion' in particle or particle == 211:
        mass = 139.6
    elif 'proton' in particle or particle == 2212:
        mass = 938.27
    return mass

def kin_to_label(kin, scheme='log_ratio'):
    kin_min = np.min(kin)
    kin_max = np.max(kin)
    if scheme == 'log_ratio':
        label = np.log10(kin / kin_min) / np.log10(kin_max / kin_min)
    elif scheme == 'log_ratio_full_photon':
        kin_min, kin_max = 256, 4194304
        label = np.log10(kin / kin_min) / np.log10(kin_max / kin_min)
    elif scheme == 'log_ratio_full_pion':
        kin_min, kin_max = 151.98902586, 4194164.40232317
        label = np.log10(kin / kin_min) / np.log10(kin_max / kin_min)
    elif scheme == 'split_at_12_18':
        def sigmoid_factor(x): # to smooth the labels around the cut values
            return 1 / (1 + np.exp(-x))
        label = np.log2(kin)
        label -= ( np.min(label) -1 ) # convert labels to integer-like starting from 1.0
        raise_step = 10
        raise_speed = 5

        # real labels will be [0.1, 0.2, ..., 0.5, 6, 7, ..., 11, 120, 130, ...]
        for i, cut in enumerate([5, 11]):
            label = np.where(label > cut * np.power(raise_step, (i)), label*raise_step*sigmoid_factor(raise_speed*(label-cut)), label)
        label /= 10
    elif scheme == 'split_at_18':
        def sigmoid_factor(x): # to smooth the labels around the cut values
            return 1 / (1 + np.exp(-x))
        label = np.log2(kin)
        label -= ( np.min(label) -1 ) # convert labels to integer-like starting from 1.0
        raise_step = 10
        raise_speed = 5

        # real labels will be [0.1, 0.2, ..., 1.1, 120, 130, ...]
        for i, cut in enumerate([11]):
            label = np.where(label > cut * np.power(raise_step, (i)), label*raise_step*sigmoid_factor(raise_speed*(label-cut)), label)
        label /= 10
    elif scheme == 'split_at_12':
        def sigmoid_factor(x): # to smooth the labels around the cut values
            return 1 / (1 + np.exp(-x))
        label = np.log2(kin)
        label -= ( np.min(label) -1 ) # convert labels to integer-like starting from 1.0
        raise_step = 10
        raise_speed = 5

        # real labels will be [0.1, 0.2, ..., 0.5, 6, 7, ...]
        for i, cut in enumerate([5]):
            label = np.where(label > cut * np.power(raise_step, (i)), label*raise_step*sigmoid_factor(raise_speed*(label-cut)), label)
        label /= 10
    elif scheme == 'index':
        label = np.log2(kin/kin_min) + 1
    else:
        raise NotImplementedError(f'{scheme} is not implemented in common.py')
    return label

def get_energies(input_file, label=False):
    input_file = h5py.File(f'{input_file}', 'r')
    energies = input_file['incident_energies'][:]
    if np.all(np.mod(energies, 1) == 0):
        energies = energies.astype(int)
    else:
        if label == True: # when energies are not integral (dataset 2&3 in calochallenge, return digitised labels
            return np.log2(energies).astype(int)
    return energies

def get_kin(input_file, label = False):
    particle = input_file.split('/')[-1].split('_')[-2][:-1]
    input_file = h5py.File(f'{input_file}', 'r')
    mass = particle_mass(particle)
    energies = input_file['incident_energies'][:]
    kin = np.sqrt( np.square(energies) + np.square(mass) ) - mass
    if label == True: # when energies are not integral (dataset 2&3 in calochallenge, return digitised labels
        return np.log2(energies).astype(int)
    return kin, particle

def plot_frame(categories, xlabel, ylabel, label_pos='left', add_summary_panel=True):
    if len(categories) == 1:
        width = 1
        height = 1
        fig, ax = plt.subplots(nrows=width, ncols=height, figsize=(4*width, 4*height))
        ax.tick_params(axis="both", which="major", width=1, length=6, labelsize=10, direction="in")
        ax.tick_params(axis="both", which="minor", width=0.5, length=3, labelsize=10, direction="in")
        ax.minorticks_on()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return fig, [ax]
    else:
        if add_summary_panel:
            categories = np.append(categories, 0)
        length = len(categories)
        width = int(np.ceil(np.sqrt(length)))
        height = int(np.ceil(length / width))
        fig, axes = plt.subplots(nrows=height, ncols=width, figsize=(4*width, 4*height))
        for ax in axes.flatten():
            ax.axis("off")
        for index, energy in enumerate(categories):
            ax = axes[(index) // width, (index) % width]
            ax.tick_params(axis="both", which="major", width=1, length=6, labelsize=10, direction="in")
            ax.tick_params(axis="both", which="minor", width=0.5, length=3, labelsize=10, direction="in")
            ax.minorticks_on()
            if index == length-1 and add_summary_panel:
                pass
            else:
                ax.axis("on")
                if isinstance(energy, str):
                    energy_legend = energy
                else:
                    energy_legend = (str(round(energy / 1000, 1)) + " GeV") if energy > 1024 else (str(energy) + " MeV")
                if label_pos == 'left':
                    ax.text(0.02, 0.98, energy_legend, transform=ax.transAxes, va="top", ha="left", fontsize=20)
                elif label_pos == 'right':
                    ax.text(0.98, 0.98, energy_legend, transform=ax.transAxes, va="top", ha="right", fontsize=20)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
        return fig, axes.flatten()

def get_counts(input_file):
    energies = get_kin(input_file, label = True)
    categories = np.unique(energies)

    counts = [np.count_nonzero(energies == c) for c in categories]
    return categories, counts

def split_energy(input_file, vector):
    if isinstance(vector, dict):
        new_dict = {}
        for k in vector:
            categories, new_dict[k] = _split_energy(input_file, vector[k].reshape(-1,1))
        return categories, new_dict
    else:
        categories, vector_list = _split_energy(input_file, vector)
        return categories, vector_list

def _split_energy(input_file, vector):
    '''
        Input: h5file and vector with length of nevents
        Output: a list of vectors splitted by energies, and the energies
    '''
    if isinstance(input_file, str):
        #energies = get_energies(input_file)
        kin_label = get_kin(input_file, label=True)
        categories, counts = get_counts(input_file)
    else: # for GAN evaluation: predict more statistics than the Geant size
        #energies = input_file
        #kin_label = np.log2(input_file).astype(int)
        kin_label = input_file
        categories = np.unique(kin_label)

    joint_array = np.concatenate([kin_label, vector], axis=1)
    joint_array = joint_array[joint_array[:, 0].argsort()]
    vector_list = np.split(joint_array[:,1:], np.unique(joint_array[:, 0], return_index=True)[1][1:])
    return categories, vector_list

def plot_energy_vox(categories, E_vox_list, label_list=None, kin_list=None, nvox='all', output=None, logx=True, particle=None, draw_ref=True, xlabel='E'):
    np.seterr(divide = 'ignore', invalid='ignore')
    GeV = 1 # no energy correction
    if nvox == 'all': loop = ['all']
    else: loop = range(nvox)
    for vox_i in loop:
        fig, axes = plot_frame(categories, xlabel=xlabel, ylabel="Events", label_pos='right')
        colors = ['k', 'r']
        for index, energy in enumerate(categories):
            ax = axes[index]
            for icurve, E_list in enumerate(E_vox_list):
                if nvox == 'all':
                    if kin_list is not None:
                        x = E_list[index][:,:] / kin_list[index]
                    else:
                        x = E_list[index][:,:]
                else:
                    if kin_list is not None:
                        x = E_list[index][:,vox_i] / kin_list[index]
                    else:
                        x = E_list[index][:,vox_i]
                if logx:
                    x = - np.log10(x.flatten() / GeV)
                else:
                    x = x.flatten() / GeV
                if icurve == 0:
                    low, high = np.nanmin(x[x != -np.inf]), np.nanmax(x[x != np.inf])
                ax.hist(x, range=(low,high), bins=40, histtype='step', color=colors[icurve], label=None if label_list is None else label_list[icurve]) # [GeV]
                ax.set_ylim(bottom=0.5)
            if draw_ref is None:
                if logx:
                    ax.axvline(x=-3, ymax=0.5, color='orange', ls='--', label='MeV')
                    ax.axvline(x=-6, ymax=0.5, color='b', ls='--', label='keV')
                else:
                    ax.axvline(x=1, ymax=0.5, color='orange', ls='--', label='MeV')
                    ax.axvline(x=1E-3, ymax=0.5, color='b', ls='--', label='keV')
            ax.ticklabel_format(style='plain')
            ax.ticklabel_format(useOffset=False, style='plain')
            ax.set_yscale('log')
            if logx:
                ax.legend(loc='center left')
            else:
                ax.legend(loc='center right')

        ax = axes[-1]
        ax.text(0.5, 0.5, particle_latex_name(particle), transform=ax.transAxes, fontsize=20)
        plt.tight_layout()
        if output is not None:
            plot_name = output.format(vox_i=vox_i)
            os.makedirs(os.path.dirname(plot_name), exist_ok=True)
            plt.savefig(plot_name)
            print('\033[92m[INFO] Save to\033[0m', plot_name)


def get_bins_given_edges(low_edge:float, high_edge:float, nbins:int, decimals:int=8, logscale=False):
    if logscale:
        bins = np.around(np.geomspace(low_edge, high_edge, num=nbins), decimals)
    else:
        bin_width = (high_edge - low_edge) / nbins
        low_bin_center  = low_edge + bin_width / 2
        high_bin_center = high_edge - bin_width /2
        bins = np.around(np.linspace(low_bin_center, high_bin_center, nbins), decimals)
    return bins

def get_xrange_from_caloflow(particle, energy, normalise=False):

    bin_map = {
        'photons': {
            256     : [0, 2.09],
            512     : [0.0037, 1.74],
            1024    : [0.5, 1.26],
            2048    : [0.7, 1.2],
            4096    : [0.82, 1.15],
            8192    : [0.78, 1.08],
            16384   : [0.85, 1.03],
            32768   : [0.92, 1.02],
            65536   : [0.9, 1.01],
            131072  : [0.88, 1.0],
            262144  : [0.88, 1.0],
            524288  : [0.9, 0.998],
            1048576 : [0.89, 0.995],
            2097152 : [0.9, 0.99],
            4194304 : [0.89, 0.988],
        },
        'pions': {
            256     : [0, 2.0] ,
            512     : [0, 2.0] ,
            1024    : [0, 2.0] ,
            2048    : [0, 1.7] ,
            4096    : [0.1, 1.7] ,
            8192    : [0.2, 1.4] ,
            16384   : [0.25, 1.3] ,
            32768   : [0.3, 1.2] ,
            65536   : [0.34, 1.2] ,
            131072  : [0.3,1.2] ,
            262144  : [0.32, 1.15] ,
            524288  : [0.5, 1.15] ,
            1048576 : [0.5, 1.15] ,
            2097152 : [0.62, 1.1] ,
            4194304 : [0.8, 1.0] ,
        },
        'electrons': {
            256     : [0, 2.09],
            512     : [0.0037, 1.74],
            1024    : [0.5, 1.26],
            2048    : [0.7, 1.2],
            4096    : [0.82, 1.15],
            8192    : [0.78, 1.08],
            16384   : [0.85, 1.03],
            32768   : [0.92, 1.02],
            65536   : [0.9, 1.01],
            131072  : [0.88, 1.0],
            262144  : [0.88, 1.0],
            524288  : [0.9, 0.998],
            1048576 : [0.89, 0.995],
            2097152 : [0.9, 0.99],
            4194304 : [0.89, 0.988],
        },
    }
    if normalise:
        return tuple([i for i in bin_map[particle][energy]])
    return tuple([i*energy/1000 for i in bin_map[particle][energy]])

