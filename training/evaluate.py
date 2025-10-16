import json, time
from argparse import ArgumentParser, Namespace
from HighLevelFeatures import HighLevelFeatures
import numpy as np
import h5py, os
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
from model import WGANGP
from quickstats.utils.common_utils import execute_multi_tasks
#from quickstats.maths.numerics import get_bins_given_edges
from itertools import repeat
from glob import glob
from common import *
from data import *
from evaluate_classifier import *
import re
from pdb import set_trace

def get_E_truth(input_file_name, mode='total', return_E_vox=False, normalise=False):
    # creating instance of HighLevelFeatures class to handle geometry based on binning file
    particle = input_file_name.split('/')[-1].split('_')[-2][:-1]
    input_data = h5py.File(f'{input_file_name}', 'r')

    if 'showers' in input_data:
        showers = input_data['showers'][:]
    else:
        layers_to_concatenate = []
        # Loop through the relevant layer numbers and append the corresponding data
        for layer in args.relevant_layers:
            layer_data = input_data[f'energy_layer_{layer}'][:] * input_data[f'incident_energy'][:][:, np.newaxis]
            layers_to_concatenate.append(layer_data)
        showers = np.concatenate(layers_to_concatenate, axis=1)

    X_train = filter_energy(particle, input_data['incident_energy'][:], args.split_energy_position, showers)
    Y_train = filter_energy(particle, input_data['incident_energy'][:], args.split_energy_position, input_data['incident_energy'][:])
    if mode == 'total':
        hlf = HighLevelFeatures(particle, filename=args.input_file, relevant_layers=args.relevant_layers)
        hlf.CalculateFeatures(X_train)
        E_tot = hlf.GetEtot()
    elif mode == 'voxel':
        hlf = HighLevelFeatures(particle, filename=args.input_file, relevant_layers=args.relevant_layers)
        hlf.CalculateFeatures(X_train)
        E_vox = X_train
    elif mode == 'layer':
        hlf = HighLevelFeatures(particle, filename=args.input_file, relevant_layers=args.relevant_layers)
        hlf.CalculateFeatures(X_train)
        E_lay = hlf.GetElayers()

    if mode == 'total':
        vector = E_tot.reshape(-1,1)
    elif mode == 'voxel':
        vector = E_vox
    elif mode == 'layer':
        vector = E_lay

    if normalise:
        vector /= Y_train
    kin = get_kin(input_file_name, label=True) # added in DS2
    kin = filter_energy(particle, input_data['incident_energy'][:], args.split_energy_position, kin)
    if 'dataset2' in input_file_name:
        categories, vector_list = split_energy(kin, vector)
    elif 'dataset1' in input_file_name and 'pion' in particle:
        categories, vector_list = split_energy(input_data['incident_energy'], vector)
    elif 'dataset1' in input_file_name and 'photon' in particle:
        categories, vector_list = split_energy(kin, vector)
    else:
        raise NotImplementedError("This feature is not implemented yet.")
    if return_E_vox:
        return categories, vector_list, vector, Y_train
    if normalise:
        return categories, vector_list, Y_train
    if mode == 'voxel':
        return categories, vector_list, hlf, Y_train

    return categories, vector_list

def get_E_gan(model_i, input_file_name, train_path, eta_slice, mode='total', preprocess=None, suffix='', return_E_vox=False, normalise_by=None, istiming=False):
    kin, particle = get_kin(input_file_name)
    input_data = h5py.File(f'{input_file_name}', 'r')
    kin = filter_energy(particle, input_data['incident_energy'][:], args.split_energy_position, kin)
    if args.center_eta_conditioning:
        center_eta = input_data['center_eta'][:]
    if args.phi_mod_conditioning:
        phi_mod = input_data['phi_mod'][:]
    config = json.load(open(os.path.join(train_path, f'{particle}s_eta_{eta_slice}{suffix}', 'train', 'config.json')))

    gan_statistics = -1 # -1 means the same statistics as input training, alternatively can use 10000 for every energy point, but this is found to be unstable in terms of chi2 values
    if gan_statistics > 0:
        unique_vals, counts = np.unique(kin,return_counts=True)
        kin = np.repeat(unique_vals, np.ones(counts.size, dtype=int) * gan_statistics)
        kin = kin.reshape(-1,1)

    label_kin = kin_to_label(kin, scheme=config['hp_config']['label_scheme'])
    if args.center_eta_conditioning:
        label_kin = np.concatenate((label_kin, center_eta.reshape(-1,1)), axis=1)
    if args.phi_mod_conditioning:
        label_kin = np.concatenate((label_kin, phi_mod.reshape(-1,1)), axis=1)
    if args.preprocess in ['normlayer1', 'normlayer2', 'normlayer3', 'normlayerMichele', 'normlayerMichele2']:
        layer_boundaries = get_layer_boundaries(input_data, args.relevant_layers)
        bin_number = layer_boundaries[1:] - layer_boundaries[:-1]
        config_string = f'normlayer__{len(args.relevant_layers)}__{":".join([ str(x) for x in bin_number if x > 0 ])}'
        if args.preprocess in ['normlayer3']:
            config_string += '__mergelayer'
    else:
        config_string = None
    wgan = WGANGP(job_config=config['job_config'], hp_config=config['hp_config'], logger=__file__, config_string=config_string, toggleConditionEtaPhi=args.eta_phi_conditioning)
    if args.eta_phi_conditioning:
        coordinates = CalculateCoordinates(input_data=input_data, relevant_layers=args.relevant_layers)
        coordinates = np.tile(coordinates, (label_kin.shape[0], 1))
        print("Also adding coordinates with shape", coordinates.shape)
        E_vox = wgan.predict(model_i=model_i, labels=label_kin, istiming=istiming, coordinates=coordinates)
    else:
        E_vox = wgan.predict(model_i=model_i, labels=label_kin, istiming=istiming)
    if istiming:
        return
    if preprocess is not None:
        if (re.compile("^log10.([0-9.]+)+$").match(preprocess) \
                or re.compile("^scale.([0-9.]+)+$").match(preprocess) \
                or re.compile("^slope.([0-9.]+)+$").match(preprocess)
            ): # log10.x, scale.x, slope
            scale = os.path.join(train_path, f'{particle}s_eta_{eta_slice}{suffix}', 'train', f'scale_{preprocess}.json')
            E_vox = preprocessing(E_vox, kin, name=preprocess, reverse=True, input_file=scale)
        elif preprocess in ['concatlayer', 'normlayer1', 'normlayer2', 'normlayer3', 'normlayerMichele', 'normlayerMichele2']:
            E_vox = preprocessing(E_vox, kin, name=preprocess, reverse=True, input_file=None, relevant_layers=args.relevant_layers, all_data=input_data)
    else:
        E_vox = preprocessing(E_vox, kin, name=preprocess, reverse=True)

    if mode == 'total':
        hlf = HighLevelFeatures(particle, filename=args.input_file, relevant_layers=args.relevant_layers)
        hlf.CalculateFeatures(np.array(E_vox))
        E_tot = hlf.GetEtot()
    elif mode == 'voxel':
        hlf = HighLevelFeatures(particle, filename=args.input_file, relevant_layers=args.relevant_layers)
        hlf.CalculateFeatures(np.array(E_vox))
        #input_data= h5py.File(f'{input_file_name}', 'r')
        #E_vox = input_data['showers'][:]
    elif mode == 'layer':
        hlf = HighLevelFeatures(particle, filename=args.input_file, relevant_layers=args.relevant_layers)
        hlf.CalculateFeatures(np.array(E_vox))
        E_lay = hlf.GetElayers()

    if mode == 'total':
        vector = E_tot.reshape(-1,1)
    elif mode == 'voxel':
        vector = E_vox
    elif mode == 'layer':
        vector = E_lay

    if normalise_by is not None:
        vector /= normalise_by

    if 'dataset2' in input_file_name:
        kin = get_kin(input_file_name, label=True) # added in DS2
        kin = filter_energy(particle, input_data['incident_energy'][:], args.split_energy_position, kin)
    if 'dataset2' in input_file_name:
        categories, vector_list = split_energy(kin, vector)
    elif 'dataset1' in input_file_name and 'pion' in particle:
        vector = filter_energy(particle, input_data['incident_energy'][:], args.split_energy_position, vector)
        categories, vector_list = split_energy(input_data['incident_energy'], vector)
    elif 'dataset1' in input_file_name and 'photon' in particle:
        vector = filter_energy(particle, input_data['incident_energy'][:], args.split_energy_position, vector)
        categories, vector_list = split_energy(kin, vector)
    else:
        raise NotImplementedError("This feature is not implemented yet.")
    if return_E_vox:
        return categories, vector_list, E_vox
    if mode == 'voxel':
        return categories, vector_list, hlf
    return categories, vector_list

def plot_energy_layer(particle, model_i, input_file_name, train_path, eta_slice):

    def merge_energies(E_list):
        concate = np.concatenate(E_list, axis=0)
        return [concate.flatten()]

    suffix = '_load' if args.loading else ''
    categories1, E_gan_list = get_E_gan(model_i, input_file_name=input_file_name, train_path=train_path, eta_slice=eta_slice, preprocess=args.preprocess, mode='layer', suffix=suffix)
    categories2, E_vox_list = get_E_truth(input_file_name, mode='layer')

    E_vox_list_merge_energy, E_gan_list_merge_energy = {}, {}
    for ilayer in E_vox_list:
        E_vox_list_merge_energy[ilayer] = merge_energies(E_vox_list[ilayer])
        E_gan_list_merge_energy[ilayer] = merge_energies(E_gan_list[ilayer])

    for ilayer in E_vox_list.keys():
        ax_text = particle_latex_name(particle) + ' Layer {}'.format(ilayer)
        plot_name = os.path.join(args.train_path, f'{particle}s_eta_{args.eta_slice}{suffix}', 'selected_{}'.format(args.showers if args.showers is not None else 'energy'), 'layer', f'plot_{particle}_{args.eta_slice}_{model_i}_layer{ilayer}.pdf')
        config = {
            'plot_chi2': False,
            'ax_text': ax_text,
            'ax_pos': (0.38, 0.19),
            'leg_loc': "lower center",
            'logx': True,
            'logy': True,
            'range_factor_factor': (30000, 20000),
            'leg_size': 10,
            'nbins': 80,
            'output_name': plot_name,
            'lw': 1,
            'xrange_from_caloflow': not args.normalise,
        }
        plot_Etot([''], E_vox_list_merge_energy[ilayer], E_gan_list_merge_energy[ilayer], config=config)


def chi2testWW(y1, y2):
    y1_err = np.sqrt(y1)
    y2_err = np.sqrt(y2)
    zeros = (y1 == 0) * (y2 == 0)
    ndf = y1.size - 1 - zeros.sum()
    if zeros.sum():
        y1 = np.delete(y1, np.where(zeros))
        y2 = np.delete(y2, np.where(zeros))
        y1_err = np.delete(y1_err, np.where(zeros))
        y2_err = np.delete(y2_err, np.where(zeros))

    W1, W2 = y1.sum(), y2.sum()
    delta = W1 * y2 - W2 * y1
    sigma = W1 * W1 * y2_err * y2_err + W2 * W2 * y1_err * y1_err
    chi2 = (delta * delta / sigma).sum()
    return chi2, ndf

def chi2caloflow(hist1, hist2):
    total_counts_ref, total_counts_data = sum(hist1), sum(hist2)
    hist1_norm = hist1 / total_counts_ref
    hist2_norm = hist2 / total_counts_data
    ret = (hist1_norm - hist2_norm)**2
    sigma_sq = hist1/((total_counts_ref)**2)+hist2/((total_counts_data)**2)
    ret = np.divide(ret, sigma_sq, out=np.zeros_like(ret), where=sigma_sq!=0)
    return ret.sum()

def plot_Etot(categories, Etot_list, Egan_list, config=None, x_axis_label="Energy [GeV]"):
    plot_chi2 = config.get('plot_chi2', False)
    ax_text = config.get('ax_text', '')
    ax_pos = config.get('ax_pos', (0.0, 0.1))
    output_name = config.get('output_name', None)
    leg_loc = config.get('leg_loc', "upper left")
    logx = config.get('logx', False)
    logy = config.get('logy', False)
    plot_range_factor = config.get('range_factor_factor', (2, 10))
    leg_size = config.get('leg_size', 20)
    nbins = config.get('nbins', 30)
    lw = config.get('lw', 2)
    xrange_from_caloflow = config.get('xrange_from_caloflow', False)

    fig, axes = plot_frame(categories, xlabel=x_axis_label, ylabel="Entries")
    results = []
    dict([(f'{energy} MeV', 0) for energy in categories])

    ndf_tot = chi2_tot = 0
    for index, energy in enumerate(categories):
        # Convert energy to GeV
        GeV = 1 if (args.normalise or args.showers) else 1000 
        etot = Etot_list[index] / GeV
        egan = Egan_list[index] / GeV

        ax = axes[index]

        if xrange_from_caloflow and energy != '':
            if '$\\gamma$' in config['ax_text']:
                particle = 'photons'
            elif '$\\pi$' in config['ax_text']:
                particle = 'pions'
            elif '$e $' in config['ax_text']:
                particle = 'electrons'
            low, high = get_xrange_from_caloflow(particle, energy, args.normalise)
        else:
            median = np.median(etot)
            high = median + min([np.absolute(np.max(etot) - median), np.absolute(np.quantile(etot, q=1-0.05) - median) * plot_range_factor[0], np.absolute(np.quantile(etot, q=1-0.16) - median) * plot_range_factor[1]])
            low  = median - min([np.absolute(np.min(etot) - median), np.absolute(np.quantile(etot, q=0.05) - median) * plot_range_factor[0], np.absolute(np.quantile(etot, q=1-0.16) - median) * plot_range_factor[1]])

        if logx:
            bins = get_bins_given_edges(low if low > 0 else 0.00001, high if high > 0 else 0.00002, nbins, 9, logscale=logx)
        else:
            bins = get_bins_given_edges(low, high, nbins, 3, logscale=logx)
        y_tot, x_tot, _ = ax.hist(np.clip(etot, bins[0], bins[-1]), bins=bins, label='G4', histtype='step', density=False, color='k', linestyle='-', alpha=0.8, linewidth=lw)
        y_gan, x_gan, _ = ax.hist(np.clip(egan, bins[0], bins[-1]), bins=bins, label='GAN', histtype='step', density=False, color='r', linestyle='--', alpha=0.8, linewidth=lw)
        chi2, ndf = chi2testWW(y_tot, y_gan)
        chi2_tot += chi2
        ndf_tot += ndf
        energy = round(energy) if len(str(energy)) > 0 and str(energy)[-1] == '0' else energy
        if args.showers is not None:
            results.append((f'{energy} MeV', 0 if chi2==0 and ndf==0 else chi2/ndf))
        else:
            results.append((f'{energy} MeV', chi2/ndf))
        if logx:
            ax.set_xscale('log')
        if logy:
            ax.set_yscale('symlog')
        if plot_chi2:
            print("chi^2 =", chi2, " ndf =", ndf)
            ax.text(0.02, 0.88, "$\chi^2$:{:.1f}".format(chi2 / ndf), transform=ax.transAxes, va="top", ha="left", fontsize=20)

    handles, labels = ax.get_legend_handles_labels()

    chi2_o_ndf = chi2_tot/ndf_tot
    results.insert(0, (f'All', chi2_o_ndf))
    ax = axes[-1]
    ax.legend(handles=handles[:2], labels=["Geant4", "GAN"], loc=leg_loc, frameon=False, fontsize=leg_size)
    if plot_chi2:
        ax.text(ax_pos[0], ax_pos[1], ax_text + "\n$\chi^2$/NDF = {:.0f}/{:.0f}\n= {:.1f}".format(chi2_tot, ndf_tot, chi2_o_ndf), transform=ax.transAxes, fontsize=leg_size)
    else:
        ax.text(ax_pos[0], ax_pos[1], ax_text, transform=ax.transAxes, fontsize=leg_size)
    if logx:
        ax.set_xscale('symlog')
    if logy:
        ax.set_yscale('symlog')

    plt.tight_layout()
    if output_name:
        os.makedirs(os.path.dirname(output_name), exist_ok=True)
        plt.savefig(output_name)
        print('\033[92m[INFO] Save to\033[0m', output_name)
        fig.clear()
        plt.close(fig)
    return dict(results)

def normalise_energy(Etot_list, Egan_list):
    Egan_list_new = []
    for Etot, Egan in zip(Etot_list, Egan_list):
        Egan_list_new.append(Egan / Etot)
    return Egan_list_new

def plot_model_i(args, model_i):
    start_time = time.time()
    particle = args.input_file.split('/')[-1].split('_')[-2][:-1]
    suffix = '_load' if args.loading else ''
    df_name = os.path.join(args.train_path, f'{particle}s_eta_{args.eta_slice}{suffix}', '{}_{}'.format(os.path.splitext(os.path.basename(__file__))[0], args.showers if args.showers is not None else 'energy'), f'chi2.csv')
    plot_name = os.path.join(args.train_path, f'{particle}s_eta_{args.eta_slice}{suffix}', '{}_{}'.format(os.path.splitext(os.path.basename(__file__))[0], args.showers if args.showers is not None else 'energy'), f'plot_{particle}_{args.eta_slice}_{model_i}.pdf')
    if os.path.exists(df_name) and os.path.exists(plot_name):
        df = pd.read_csv(df_name)
        if not args.debug and model_i in df['ckpt'].values:
            chi2_results = df[df['ckpt'] == model_i].to_dict(orient='records')[0]
            print('\033[92m[INFO] Cache\033[0m', 'model', model_i, 'chi2', chi2_results['All'])
            return chi2_results

    if args.normalise:
        categories, Etot_list, Y_train = get_E_truth(args.input_file, normalise=args.normalise)
    else:
        if args.showers is not None:
            categories, Etot_list, hlf_g4, Y_train = get_E_truth(args.input_file, mode='voxel', normalise=args.normalise) # Etot_list has Evox data inside
        else:
            categories, Etot_list = get_E_truth(args.input_file, normalise=args.normalise)
    truth_time = time.time() - start_time
    start_time = time.time()
    if args.showers is not None:
        categories, Egan_list, hlf_gan = get_E_gan(model_i=model_i, input_file_name=args.input_file, train_path=args.train_path, eta_slice=args.eta_slice, mode='voxel', preprocess=args.preprocess, suffix=suffix, normalise_by=(Y_train if args.normalise else None)) # Egan_list has Evox data inside
    else:
        categories, Egan_list = get_E_gan(model_i=model_i, input_file_name=args.input_file, train_path=args.train_path, eta_slice=args.eta_slice, preprocess=args.preprocess, suffix=suffix, normalise_by=(Y_train if args.normalise else None))
    gan_time = time.time() - start_time
    start_time = time.time()

    eta_min, eta_max = tuple(args.eta_slice.split('_'))
    ax_text = particle_latex_name(particle)+ ", " + str("{:.2f}".format(int(eta_min) / 100, 2)) + r"$<|\eta|<$" + str("{:.2f}\n".format((int(eta_max)) / 100, 2)) + "Iter: {}".format(int(model_i)*1000)
    #if args.showers:
    #    g4_WidthPhis = hlf_g4.GetWidthPhis()
    #    gan_WidthPhis = hlf_gan.GetWidthPhis()
    if args.showers is not None:
        chosen_shower_var = args.showers.split("_")[0]
        if chosen_shower_var == "ECEtas":
            g4_shower_var_getter = hlf_g4.GetECEtas
            gan_shower_var_getter = hlf_gan.GetECEtas
        elif chosen_shower_var == "ECPhis":
            g4_shower_var_getter = hlf_g4.GetECPhis
            gan_shower_var_getter = hlf_gan.GetECPhis
        elif chosen_shower_var == "WidthEtas":
            g4_shower_var_getter = hlf_g4.GetWidthEtas
            gan_shower_var_getter = hlf_gan.GetWidthEtas
        elif chosen_shower_var == "WidthPhis":
            g4_shower_var_getter = hlf_g4.GetWidthPhis
            gan_shower_var_getter = hlf_gan.GetWidthPhis
        else:
            raise NotImplementedError(f"Shower shape variable {chosen_shower_var} is not implemented or spelled wrong.")
        g4_shower_var = g4_shower_var_getter()
        gan_shower_var = gan_shower_var_getter()
        print("Evaluation running on shower shape variable", chosen_shower_var)
    config = {
        'plot_chi2': True,
        'ax_text': ax_text,
        'output_name': plot_name, 
        'nbins': 30,
    }
    #if args.showers:
    #    g4_WidthPhis_rearranged = np.asarray([g4_WidthPhis.get(2)[Y_train==k] for k in np.unique(Y_train)])
    #    gan_WidthPhis_rearranged = np.asarray([gan_WidthPhis.get(2)[Y_train==k] for k in np.unique(Y_train)])
    #    chi2_results = plot_Etot(categories, g4_WidthPhis_rearranged, gan_WidthPhis_rearranged, config)
    if args.showers is not None:
        chosen_layer = int(args.showers.split("_")[1])
        try:
            g4_shower_var_rearranged = np.asarray([g4_shower_var.get(chosen_layer)[Y_train==k] for k in np.unique(Y_train)])
            gan_shower_var_rearranged = np.asarray([gan_shower_var.get(chosen_layer)[Y_train==k] for k in np.unique(Y_train)])
        except TypeError:
            raise TypeError("If you're using --relevant_layers, make sure you've provided to the --showers flag a layer number that is included among them.")
        if chosen_shower_var=="ECEtas":
            chosen_shower_var_latex = r"EC $\eta$"
        elif chosen_shower_var=="ECPhis":
            chosen_shower_var_latex = r"EC $\phi$"
        elif chosen_shower_var=="WidthEtas":
            chosen_shower_var_latex = r"Width $\eta$"
        elif chosen_shower_var=="WidthPhis":
            chosen_shower_var_latex = r"Width $\phi$"
        chi2_results = plot_Etot(categories, g4_shower_var_rearranged, gan_shower_var_rearranged, config, chosen_shower_var_latex)
    else:
        chi2_results = plot_Etot(categories, Etot_list, Egan_list, config)
    plot_time = time.time() - start_time
    #if args.showers:
    #    print('\033[92m[INFO] Evaluate result on WidthPhis (layer 2)\033[0m', 'model', model_i, 'chi2', f'{chi2_results["All"]:.2f}', f'time (truth) {truth_time:.1f}s (gan) {gan_time:.1f}s (plot) {plot_time:.1f}s')
    if args.showers is not None:
        print('\033[92m[INFO] Evaluate result on', chosen_shower_var, 'for layer', chosen_layer, '\033[0m', 'model', model_i, 'chi2', f'{chi2_results["All"]:.2f}', f'time (truth) {truth_time:.1f}s (gan) {gan_time:.1f}s (plot) {plot_time:.1f}s')
    else:
        print('\033[92m[INFO] Evaluate result\033[0m', 'model', model_i, 'chi2', f'{chi2_results["All"]:.2f}', f'time (truth) {truth_time:.1f}s (gan) {gan_time:.1f}s (plot) {plot_time:.1f}s')
    return {f'ckpt': model_i, **chi2_results}

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def best_ckpt(args, df, cache=False, alt='', mask_cache=False):
    suffix = '_load' if args.loading else ''
    particle = args.input_file.split('/')[-1].split('_')[-2][:-1]
    best_folder = os.path.join(args.train_path, f'{particle}s_eta_{args.eta_slice}{suffix}', 'selected{}_{}'.format(alt, args.showers if args.showers is not None else 'energy'))
    chi_name = os.path.join(best_folder, 'chi2.pdf')
    if not (os.path.exists(chi_name) and cache):
        os.makedirs(best_folder, exist_ok=True)
        best_x = int(df[df[f'All{alt}'] == df[f'All{alt}'].min()]['ckpt'].iloc[0] * 1000)
        best_y = float(df[f'All{alt}'].min())
        x = df['ckpt'] * 1000
        y = df[f'All{alt}']

        remove = f' MeV{alt}'
        categories = [float(i.replace(remove, '')) for i in df if remove in i and alt in i]
        categories = [round(num) for num in categories if str(num)[-1] == '0']
        chi2_list = [df[f'{c} MeV{alt}'].values for c in categories]
        fig, axes = plot_frame(categories + ['All energies'], xlabel="Iterations", ylabel="$\chi^{2}$/NDF", label_pos='right', add_summary_panel=False)
        leg_loc, leg_size = (0.52, 0.72), 12
        markersize = 10
        for index, energy in enumerate(categories):
            ax = axes[index]
            ax.scatter(x, chi2_list[index], s=markersize, facecolor='none', edgecolors="k", alpha=0.9)
            best_x_i = int(df[df[f'{energy} MeV{alt}'] == df[f'{energy} MeV{alt}'].min()]['ckpt'].iloc[0] * 1000)
            best_y_i = df[f'{energy} MeV{alt}'].min()
            try:
                best_y_j = float(df[df['ckpt']==int(best_x/1000)][f'{energy} MeV{alt}'].iloc[0])
            except:
                set_trace()
            ax.scatter(best_x_i, best_y_i, s=markersize*4, c="orange", label="Local min.")
            ax.scatter(best_x, best_y_j, s=markersize*4, c="r", label="Selected")
            #ax.text(0.98, 0.97, "Iter {}\n$\chi^2$ = {:.1f}\nSel. iter {}\n$\chi^2$ = {:.1f}".format(best_x_i, best_y_i, best_x, best_y_j), transform=ax.transAxes, va="top", ha="right", fontsize=10, bbox=dict(facecolor='w', alpha=0.8, edgecolor='w'))
            #ax.legend(loc=leg_loc, frameon=True, fontsize=leg_size)
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(max(0, ymin), min(50, ymax))

        ax = axes[index+1]
        ax.scatter(x, y, s=markersize, facecolor='none', edgecolors="k", alpha=0.9)
        ax.scatter(best_x, best_y, s=markersize*4, c="orange", label="Local min.")
        ax.scatter(best_x, best_y, s=markersize*4, c="r", label='Selected')
        eta_min, eta_max = tuple(args.eta_slice.split('_'))
        #ax.text(0.98, 0.98, particle_latex_name(particle) + "\n" + str("{:.2f}".format(int(eta_min) / 100, 2)) + r"$<|\eta|<$" + str("{:.2f}\n".format((int(eta_min) + 5) / 100, 2)) + "Iter {}\n$\chi^2$ = {:.1f}".format(best_x, best_y), transform=ax.transAxes, va="top", ha="right", fontsize=15, bbox=dict(facecolor='w', alpha=0.8, edgecolor='w'))
        ax.legend(loc=leg_loc, frameon=True, fontsize=leg_size)

        ymin, ymax = ax.get_ylim()
        ax.set_ylim(max(0, ymin), min(50, ymax))
        plt.tight_layout()
        plt.savefig(chi_name)
        print('\033[92m[INFO] Save to\033[0m', chi_name)
        fig.clear()
        plt.close(fig)

    csv_name = os.path.join(best_folder, f'chi2{alt}.csv')
    best_df = df[df[f'All{alt}'] == df[f'All{alt}'].min()]
    if not (os.path.exists(csv_name) and cache):
        best_df.to_csv(csv_name, index=False)

        models = glob.glob(os.path.join(args.train_path, f'{particle}s_eta_{args.eta_slice}{suffix}', 'checkpoints', f'model-{int(best_df["ckpt"].iloc[0])}*'))
        for model in models:
            os.system(f'cp {model} {best_folder}')  
        plot_name = os.path.join(args.train_path, f'{particle}s_eta_{args.eta_slice}{suffix}', '{}_{}'.format(os.path.splitext(os.path.basename(__file__))[0], args.showers if args.showers is not None else 'energy'), f'plot_{particle}_{args.eta_slice}_{int(best_df["ckpt"].iloc[0])}.*')
        os.system(f'cp {plot_name} {best_folder}')

    vox_name = os.path.join(best_folder, 'mask', f'mask_{particle}_{args.eta_slice}_{int(best_df["ckpt"].iloc[0])}_all.pdf')
    if not (os.path.exists(vox_name) and mask_cache):
        # Plot 'masking' distribution; 'masking' means to remove voxel energies below a threshold of 1keV or 1MeV
        categories, E_gan_list, E_gan_vox = get_E_gan(model_i=int(best_df["ckpt"].iloc[0]), input_file_name=args.input_file, train_path=args.train_path, eta_slice=args.eta_slice, preprocess=args.preprocess, mode='voxel', suffix=suffix, return_E_vox=True, istiming=args.istiming)
        categories, E_tru_list, E_tru_vox, E_incident = get_E_truth(args.input_file, mode='voxel', return_E_vox=True)
        #kin, particle = get_kin(args.input_file) # for DS1
        kin = get_kin(args.input_file, label=True) # added in DS2
        kin = filter_energy(particle, h5py.File(f'{args.input_file}', 'r')['incident_energy'][:], args.split_energy_position, kin)
        categories, kin_list = split_energy(kin, kin)
        xlabel = f"Energy of voxel [MeV]"
        plot_energy_vox(categories, [E_tru_list, E_gan_list], label_list=['Geant4', 'GAN'], nvox='all', \
                logx=False, particle=particle, output=vox_name, xlabel=xlabel)
        plot_energy_vox(categories, [E_tru_list, E_gan_list], label_list=['Geant4', 'GAN'], nvox='all', \
                logx=True, particle=particle, output=vox_name.replace('.pdf', '_logx.pdf'), xlabel="$-$" + f"Log({xlabel})")
        xlabel += " / kinematics"
        plot_energy_vox(categories, [E_tru_list, E_gan_list], label_list=['Geant4', 'GAN'], kin_list=kin_list, nvox='all', \
                logx=False, particle=particle, output=vox_name.replace('.pdf', '_normkin.pdf'), draw_ref=False, xlabel=xlabel)
        plot_energy_vox(categories, [E_tru_list, E_gan_list], label_list=['Geant4', 'GAN'], kin_list=kin_list, nvox='all', \
                logx=True, particle=particle, output=vox_name.replace('.pdf', '_normkin_logx.pdf'), draw_ref=False, xlabel="$-$" + f"Log({xlabel})")
    
        layer_folder = os.path.join(args.train_path, f'{particle}s_eta_{args.eta_slice}{suffix}', 'selected_{}'.format(args.showers if args.showers is not None else 'energy'), 'layer')
        #if not os.path.exists(layer_folder) or len(os.listdir(layer_folder)) == 0:
        #    plot_energy_layer(particle=particle, model_i=int(best_df["ckpt"]), input_file_name=args.input_file, train_path=args.train_path, eta_slice=args.eta_slice)
        plot_energy_layer(particle=particle, model_i=int(best_df["ckpt"].iloc[0]), input_file_name=args.input_file, train_path=args.train_path, eta_slice=args.eta_slice)

        if args.save_h5:
            output_h5 = os.path.join(best_folder, 'h5', 'gan.h5')
            if not os.path.exists(output_h5):
                os.makedirs(os.path.dirname(output_h5), exist_ok=True)
                print('Save to h5', E_gan_vox.shape, E_tru_vox.shape)
                gen_h5(E_incident, E_gan_vox, output_h5)
            else:
                print('Skip', output_h5)
        if args.convert:
            config = json.load(open(os.path.join(args.train_path, f'{particle}s_eta_{args.eta_slice}{suffix}', 'train', 'config.json')))
            if args.preprocess in ['normlayer1', 'normlayer2', 'normlayer3', 'normlayerMichele', 'normlayerMichele2']:
                layer_boundaries = get_layer_boundaries(input_data, args.relevant_layers)
                bin_number = layer_boundaries[1:] - layer_boundaries[:-1]
                config_string = f'normlayer__{len(args.relevant_layers)}__{":".join([ str(x) for x in bin_number if x > 0 ])}'
                if args.preprocess in ['normlayer3']:
                    config_string += '__mergelayer'
            else:
                config_string = None
            wgan = WGANGP(job_config=config['job_config'], hp_config=config['hp_config'], logger=__file__, config_string=config_string, toggleConditionEtaPhi=args.eta_phi_conditioning)
            wgan.convert_model(int(best_x/1000))

def gen_h5(energies, showers, output):
    dataset_file = h5py.File(output, 'w')
    energies = np.array(energies)
    showers  = np.array(showers)
    dataset_file.create_dataset('incident_energy', data=energies.reshape(len(energies), -1), compression='gzip')
    dataset_file.create_dataset('showers', data=showers.reshape(len(showers), -1), compression='gzip')
    print('Save h5 file to', output)
    dataset_file.close()

##### auc_model_i() is to evaluate model by training a binary classifier, taken from https://github.com/CaloChallenge/homepage/blob/main/code/evaluate.py
def auc_model_i(args, model_i):
    start_time = time.time()
    input_file_name = args.input_file
    particle = args.input_file.split('/')[-1].split('_')[-2][:-1]
    suffix = '_load' if args.loading else ''

    parser_replacement = {
        'save_mem': False,
        'cls_n_layer': 2,
        'cls_n_hidden': 50,
        'cls_dropout_probability': 0,
        'cls_batch_size': 1000,
        'cls_n_epochs': 50,
        'device': 'cpu',
        'cls_lr': 2e-4,
        'mode': 'cls-high',
        'dataset': input_file_name.split('/')[-1].split('_')[-1].split('.')[0],
        'output_dir': os.path.join(args.train_path, f'{particle}s_eta_{args.eta_slice}{suffix}', '{}_{}'.format(os.path.splitext(os.path.basename(__file__))[0], args.showers if args.showers is not None else 'energy'), f'evaluate_classifier'),
        'ckpt': model_i,
    }
    parser_args = Namespace(**parser_replacement)

    if os.path.exists(os.path.join(parser_args.output_dir, f'loss_{parser_args.ckpt}.json')):
        with open(os.path.join(parser_args.output_dir, f'loss_{parser_args.ckpt}.json'), 'r') as f:
            classifer_results = json.load(f)
        if 'AUC' in classifer_results:
            classifer_results.pop('loss_history')
            print('\033[92m[INFO] Cache\033[0m', 'model', model_i, 'classifer_results', classifer_results['AUC'])
            return classifer_results

    categories, Etru_list, Etru_vox, E_incident = get_E_truth(input_file_name, mode='voxel', return_E_vox=True)
    truth_time = time.time() - start_time
    start_time = time.time()
    categories, Egan_list, Egan_vox = get_E_gan(model_i=model_i, input_file_name=input_file_name, train_path=args.train_path, eta_slice=args.eta_slice, preprocess=args.preprocess, suffix=suffix, mode='voxel', return_E_vox=True)
    gan_time = time.time() - start_time
    start_time = time.time()
    hlf_class = HighLevelFeatures(particle, filename=args.input_file, relevant_layers=args.relevant_layers)
    tru_array = prepare_high_data_for_classifier(hlf_class, Etru_vox, 0, E_incident)
    hlf_class = HighLevelFeatures(particle, filename=args.input_file, relevant_layers=args.relevant_layers)
    gan_array = prepare_high_data_for_classifier(hlf_class, np.array(Egan_vox), 1, E_incident)
    train_data, test_data, val_data = ttv_split(tru_array, gan_array)
    eval_acc, eval_auc, eval_JSD = train_evaluate_classifier(parser_args, train_data, val_data, test_data)
    plot_time = time.time() - start_time

    classifer_results = {
        'ckpt': model_i,
        'Accuracy': eval_acc,
        'AUC': eval_auc,
        'JSD': eval_JSD,
    }
    with open(os.path.join(parser_args.output_dir, f'loss_{parser_args.ckpt}.json'), 'r') as f:
        loss_history = json.load(f)
    with open(os.path.join(parser_args.output_dir, f'loss_{parser_args.ckpt}.json'), 'w') as f:
        json.dump({**classifer_results, 'loss_history': loss_history}, f, indent=2)
    print('\033[92m[INFO] Evaluate result\033[0m', 'model', model_i, 'AUC', f'{classifer_results["AUC"]:.2f}', f'time (truth) {truth_time:.1f}s (gan) {gan_time:.1f}s (classify) {plot_time:.1f}s')
    return classifer_results

def main(args):
    particle = args.input_file.split('/')[-1].split('_')[-2][:-1]
    suffix = '_load' if args.loading else ''
    models = glob.glob(os.path.join(args.train_path, f'{particle}s_eta_{args.eta_slice}{suffix}', 'checkpoints', 'model-*.index'))
    models = [int(m.split('/')[-1].split('-')[-1].split('.')[0]) for m in models]
    models.sort(reverse = True)

    if not models:
        print('\033[91m[ERROR] No model is found at\033[0m', os.path.join(args.train_path, f'{particle}s_eta_{args.eta_slice}{suffix}', 'checkpoints', 'model-*.index'))
        return
    else:
        print('\033[92m[INFO] Evaluate\033[0m', particle, args.input_file, f'| {len(models)} models')

    if args.checkpoint:
        size = 101
        chunks = [models[x:x+size] for x in range(0, len(models), size)]
        if args.islice is not None:
            chunks = [chunks[args.islice]]
    else:
        chunks = [models]

    for models in chunks:
        arguments = (repeat(args), models)
        if 'dataset1' in args.input_file:
            results = execute_multi_tasks(plot_model_i, *arguments, parallel=0 if (args.debug and not args.istiming) else -1)
            filename = f'chi2.csv'
        elif 'dataset2' in args.input_file:
            results = execute_multi_tasks(plot_model_i, *arguments, parallel=0 if (args.debug and not args.istiming) else -1)
            filename = f'classifier.csv'
        df = pd.DataFrame(results).sort_values(by=['ckpt'])
        df_name = os.path.join(args.train_path, f'{particle}s_eta_{args.eta_slice}{suffix}', '{}_{}'.format(os.path.splitext(os.path.basename(__file__))[0], args.showers if args.showers is not None else 'energy'), filename)
        if os.path.exists(df_name):
            df_old = pd.read_csv(df_name)
            df = pd.concat([df, df_old]).drop_duplicates(subset=['ckpt']).reset_index(drop=True)
        df.sort_values(by=['ckpt']).drop_duplicates().reset_index(drop=True).to_csv(df_name, index=False)
        print('\033[92m[INFO] Save to\033[0m', df_name, df.shape)

    best_ckpt(args, df, cache=False)
    

if __name__ == '__main__':

    """Get arguments from command line."""
    parser = ArgumentParser(description="\033[92mConfig for training.\033[0m")
    parser.add_argument('-i', '--input_file', type=str, required=False, default='', help='Training h5 file name (default: %(default)s)')
    parser.add_argument('--relevant_layers', type=int, required=True, nargs='+', help='Relevant layers (Photons [0, 1, 2, 3, 12])')
    parser.add_argument('-t', '--train_path', type=str, required=True, default='../output/dataset1/v1', help='--out_path from train.py (default: %(default)s)')
    parser.add_argument('-e', '--eta_slice', type=str, required=False, default='20_25', help='--out_path from train.py (default: %(default)s)')
    parser.add_argument('--debug', required=False, action='store_true', help='Debug mode (default: %(default)s)')
    parser.add_argument('-p', '--preprocess', type=str, required=False, default=None, help='Preprocessing name (default: %(default)s)')
    parser.add_argument('--checkpoint', required=False, action='store_true', help='Split evaluation into chunks (default: %(default)s)')
    parser.add_argument('--islice', required=False, type=int, default=None, help='Split evaluation into chunks and only run one slice (default: %(default)s)')
    parser.add_argument('-l', '--loading', type=str, required=False, default=None, help='Load model (default: %(default)s)')
    parser.add_argument('--normalise', required=False, action='store_true', help='Plot E_gan/E_truth (default: %(default)s)')
    parser.add_argument('--split_energy_position', type=str, required=False, default='', choices=['', 'le12', 'ge12', 'ge12le18', 'ge18'], help='Load model (default: %(default)s)')
    parser.add_argument('--save_h5', required=False, action='store_true', help='Save H5 https://calochallenge.github.io/homepage/ (default: %(default)s)')
    parser.add_argument('--istiming', required=False, nargs='+', default=False, help='Measure timing: a tuple of three: batch, Ekin, trials (default: %(default)s)')
    parser.add_argument('--convert', required=False, action='store_true', help='Convert best model to lwtnn (default: %(default)s)')
    #parser.add_argument('--showers', required=False, action='store_true', help='Use shower shape to choose best training iteration (default: %(default)s)')
    parser.add_argument('--showers', type=str, required=False, default=None, help='Use shower shape to choose best training iteration. Provided string must be of the format VARIABLE_LAYER, where VARIABLE is a shower shape variable among ECEtas, ECPhis, WidthEtas and WidthPhis, and LAYER is the number of the layer where to evaluate the shower shape variable (e.g., ECEtas_2 runs evaluation on shower shape variable ECEtas for layer no. 2) (default: %(default)s)')
    parser.add_argument('--center_eta_conditioning', required=False, action='store_true', help='To be specified if conditioning was also done on center_eta (default: %(default)s)')
    parser.add_argument('--phi_mod_conditioning', required=False, action='store_true', help='To be specified if conditioning was also done on phi_mod (default: %(default)s)')
    parser.add_argument('--eta_phi_conditioning', required=False, action='store_true', help='To be specified if conditioning was also done on eta and phi (default: %(default)s)')
    args = parser.parse_args()
    print(args)
    main(args)
