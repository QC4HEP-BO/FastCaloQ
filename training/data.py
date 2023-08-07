import numpy as np
import re
from pdb import set_trace

def preprocessing(X_train, kin, name=None, reverse=False, input_file=None, xml=None):
    if not reverse: # train
        if name is None:
            X_train /= kin
        elif name in ['concatlayer', 'normlayer1']:
            X_train /= kin
            bin_edges = xml.GetBinEdges()
            E_layers = []
            for layer in xml.GetRelevantLayers():
                E_layers.append(X_train[:, bin_edges[layer]:bin_edges[layer+1]].mean(axis=-1).reshape(-1, 1))
            E_layers = np.concatenate(E_layers, axis=1)
            X_train = np.concatenate([X_train, E_layers], axis=1)
            return X_train
        elif name in ['normlayer2', 'normlayerMichele2']:
            # https://docs.google.com/presentation/d/e/2PACX-1vTqNjAM0DMe7gM7E6zBIeT4JaIP31S_5ELiGPeOGQ0ORRH0zQHygyY3cIYGkBv0Xwjd3B1cs3oXfjEI/pub?start=false&loop=false&delayms=3000&slide=id.g24b23d90052_0_366
            import tensorflow as tf
            X_train[X_train == 0] = 0.0001
            #X_train[X_train <= 1e-6] = 1e-6
            bin_edges = xml.GetBinEdges()
            E_layers = []
            for layer in xml.GetRelevantLayers():
                E_layers.append(X_train[:, bin_edges[layer]:bin_edges[layer+1]].sum(axis=-1).reshape(-1, 1))
                # normalise voxel energy by layer energy; afterwards, by definition X_train[:, bin_edges[layer]:bin_edges[layer+1].sum(axis=-1) = 1
                X_train[:, bin_edges[layer]:bin_edges[layer+1]] = tf.math.divide_no_nan(X_train[:, bin_edges[layer]:bin_edges[layer+1]], E_layers[-1])
            E_layers = np.concatenate(E_layers, axis=1)
            E_shower = E_layers.sum(axis=-1).reshape(-1, 1)

            # normalise layer energy by shower energy; afterwards, by definition E_layers.sum(axis=-1) = 1
            E_layers = tf.math.divide_no_nan(E_layers, E_layers.sum(axis=-1).reshape(-1, 1))

            # construct E_shower / kin
            E_truth = np.full((X_train.shape[0], 1), E_shower/kin)
            X_train = np.concatenate([X_train, E_layers, E_truth], axis=1)
            return X_train
        elif name in ['normlayerMichele']: # normlayerMichele is to reproduce Michele's model
            import tensorflow as tf
            X_train[X_train == 0] = 0.0001
            #X_train[X_train <= 1e-6] = 1e-6
            bin_edges = xml.GetBinEdges()
            E_layers = []
            for layer in xml.GetRelevantLayers():
                E_layers.append(X_train[:, bin_edges[layer]:bin_edges[layer+1]].sum(axis=-1).reshape(-1, 1))
                # normalise voxel energy by layer energy; afterwards, by definition X_train[:, bin_edges[layer]:bin_edges[layer+1].sum(axis=-1) = 1
                X_train[:, bin_edges[layer]:bin_edges[layer+1]] = tf.math.divide_no_nan(X_train[:, bin_edges[layer]:bin_edges[layer+1]], E_layers[-1])
            E_layers = np.concatenate(E_layers, axis=1)
            E_shower = E_layers.sum(axis=-1).reshape(-1, 1)

            # normalise layer energy by shower energy; afterwards, by definition E_layers.sum(axis=-1) = 1
            E_layers = tf.math.divide_no_nan(E_layers, E_layers.sum(axis=-1).reshape(-1, 1))

            # construct E_shower / kin
            E_truth = np.full((X_train.shape[0], 1), E_shower/kin)
            X_train = np.concatenate([X_train, E_truth, E_layers], axis=1)
            return X_train
        elif name in ['normlayer3']:
            # https://docs.google.com/presentation/d/e/2PACX-1vTqNjAM0DMe7gM7E6zBIeT4JaIP31S_5ELiGPeOGQ0ORRH0zQHygyY3cIYGkBv0Xwjd3B1cs3oXfjEI/pub?start=false&loop=false&delayms=3000&slide=id.g24b23d90052_0_366
            import tensorflow as tf
            X_train[X_train == 0] = 0.0001
            bin_edges = xml.GetBinEdges()
            E_layers = []
            for layer in xml.GetRelevantLayers():
                E_layers.append(X_train[:, bin_edges[layer]:bin_edges[layer+1]].sum(axis=-1).reshape(-1, 1))
                # normalise voxel energy by layer energy; afterwards, by definition X_train[:, bin_edges[layer]:bin_edges[layer+1].sum(axis=-1) = 1
                X_train[:, bin_edges[layer]:bin_edges[layer+1]] = tf.math.divide_no_nan(X_train[:, bin_edges[layer]:bin_edges[layer+1]], E_layers[-1])

            # further normalise by number of layers such that we can perform softmax on all voxels (not in individual layers)
            X_train /= len(xml.GetRelevantLayers())

            E_layers = np.concatenate(E_layers, axis=1)
            E_shower = E_layers.sum(axis=-1).reshape(-1, 1)

            # normalise layer energy by shower energy; afterwards, by definition E_layers.sum(axis=-1) = 1
            E_layers = tf.math.divide_no_nan(E_layers, E_layers.sum(axis=-1).reshape(-1, 1))

            # construct E_shower / kin
            E_truth = np.full((X_train.shape[0], 1), E_shower/kin)
            X_train = np.concatenate([X_train, E_layers, E_truth], axis=1)
            return X_train
        elif name == 'neglog10plus1':
            X_train = - np.log10((X_train + 1) / kin)
        elif re.compile("^log10.([0-9.]+)+$").match(name): # log10.x
            from common import split_energy, get_energies
            X_train = np.log10((X_train / kin) + 1)
            _, xtrain_list = split_energy(input_file, X_train)
            _, kin_list = split_energy(input_file, kin)
            high = float(re.compile("^log10.([0-9.]+)+$").match(name).groups()[0])
            print('scale to', high)
            scale = []
            for k, v in zip(kin_list, xtrain_list):
                if high/np.sort(v.flatten())[-3] < 1:
                    scale.append((k[0].item(), float(high/np.sort(v.flatten())[-3])))
                else:
                    scale.append((k[0].item(), int(high/np.sort(v.flatten())[-3])))
            scale = dict(scale)
            for k,s in scale.items():
                mask = (kin == k)
                X_train[mask.flatten(), :] *= s
            return X_train, scale
        elif re.compile("^scale.([0-9.]+)+$").match(name): # scale.x
            from common import split_energy, get_energies
            X_train /= kin
            _, xtrain_list = split_energy(input_file, X_train)
            _, kin_list = split_energy(input_file, kin)
            high = float(re.compile("^scale.([0-9.]+)+$").match(name).groups()[0])
            print('scale to', high)
            scale = []
            for k, v in zip(kin_list, xtrain_list):
                if high/np.sort(v.flatten())[-3] < 1:
                    scale.append((k[0].item(), float(high/np.sort(v.flatten())[-3])))
                else:
                    scale.append((k[0].item(), int(high/np.sort(v.flatten())[-3])))
            scale = dict(scale)
            for k,s in scale.items():
                mask = (kin == k)
                X_train[mask.flatten(), :] *= s
            return X_train, scale
        elif re.compile("^slope.([0-9.]+)+$").match(name): # slope.x
            from common import split_energy, get_energies
            X_train /= kin
            _, xtrain_list = split_energy(input_file, X_train)
            _, kin_list = split_energy(input_file, kin)
            high = float(re.compile("^slope.([0-9.]+)+$").match(name).groups()[0])
            scale = []
            scale_list = [-10.0] * 15 
            assert(len(scale_list) >= len(kin_list))
            for k, v, s in zip(kin_list, xtrain_list, scale_list):
                if s < 0:
                    scale.append((k[0].item(), -s))
                else:
                    scale.append((k[0].item(), float(s/np.sort(v.flatten())[-3])))
            scale = dict(scale)
            for k,s in scale.items():
                mask = (kin == k)
                X_train[mask.flatten(), :] *= s
            return X_train, scale
        else:
            raise NotImplementedError
    else: # evaluate
        if name is None:
            X_train *= kin
        elif name in ['concatlayer', 'normlayer1']:
            X_train = X_train[:, :X_train.shape[1] - len(xml.GetRelevantLayers())] # drop the last xml.GetRelevantLayers() columns
            X_train *= kin
            return X_train
        elif name in ['normlayer2', 'normlayerMichele2']:
            import tensorflow as tf
            E_shower = tf.reshape(X_train[:, -1], (-1, 1))
            E_shower *= kin

            E_layers = X_train[:, -1-len(xml.GetRelevantLayers()) : -1].numpy()
            E_layers *= E_shower

            X_train = X_train[:, :-1-len(xml.GetRelevantLayers())].numpy() # drap the last xml.GetRelevantLayers() + 1 columns
            bin_edges = xml.GetBinEdges()
            for num, layer in enumerate(xml.GetRelevantLayers()):
                X_train[:, bin_edges[layer]:bin_edges[layer+1]] *= (E_layers[:, num].numpy().reshape(-1, 1))

            return tf.convert_to_tensor(X_train)
        elif name in ['normlayer3']:
            import tensorflow as tf
            # E_shower = predicted shower * kin
            E_shower = tf.reshape(X_train[:, -1], (-1, 1))
            E_shower *= kin

            # E_layer = predicted layer * E_shower
            E_layers = X_train[:, -1-len(xml.GetRelevantLayers()) : -1].numpy()
            E_layers *= E_shower

            # E_voxel = predicted voxel * E_layer * numberOfLayer
            X_train = X_train[:, :-1-len(xml.GetRelevantLayers())].numpy() # drap the last xml.GetRelevantLayers() + 1 columns
            bin_edges = xml.GetBinEdges()
            for num, layer in enumerate(xml.GetRelevantLayers()):
                X_train[:, bin_edges[layer]:bin_edges[layer+1]] *= (E_layers[:, num].numpy().reshape(-1, 1))
            X_train *= len(xml.GetRelevantLayers())

            return tf.convert_to_tensor(X_train)
        elif name in ['normlayerMichele']: # normlayerMichele vs normlayerMichele2: position of total energy is different. [voxE, layerE, showerE] vs [voxE, showerE, layerE]
            import tensorflow as tf
            E_shower = tf.reshape(X_train[:, -1-len(xml.GetRelevantLayers())], (-1, 1))
            E_shower *= kin

            E_layers = X_train[:, -len(xml.GetRelevantLayers()) : ].numpy()
            E_layers *= E_shower

            X_train = X_train[:, :-1-len(xml.GetRelevantLayers())].numpy() # drap the last xml.GetRelevantLayers() + 1 columns
            bin_edges = xml.GetBinEdges()
            for num, layer in enumerate(xml.GetRelevantLayers()):
                X_train[:, bin_edges[layer]:bin_edges[layer+1]] *= (E_layers[:, num].numpy().reshape(-1, 1))

            return tf.convert_to_tensor(X_train)
        elif name == 'neglog10plus1':
             X_train = np.power(10, -X_train) * kin - 1
        elif re.compile("^log10.([0-9.]+)+$").match(name): # log10.x
            import json, tensorflow as tf
            with open(input_file, 'r') as fp:
                scale = json.load(fp)
            scale = dict([(float(k), v) for k,v in scale.items()])
            X_train = X_train.numpy()
            for k,s in scale.items():
                mask = (kin == k)
                X_train[mask.flatten(), :] /= s
            X_train = (np.power(10, X_train) - 1) * kin
            X_train = tf.convert_to_tensor(X_train)
        elif (re.compile("^scale.([0-9.]+)+$").match(name) or \
                re.compile("^slope.([0-9.]+)+$").match(name)
            ): # scale.x
            import json, tensorflow as tf
            with open(input_file, 'r') as fp:
                scale = json.load(fp)
            scale = dict([(float(k), v) for k,v in scale.items()])
            X_train = X_train.numpy()
            for k,s in scale.items():
                mask = (kin == k)
                X_train[mask.flatten(), :] /= s
            X_train *= kin
            X_train = tf.convert_to_tensor(X_train)
        else:
            raise NotImplementedError
    return X_train

def filter_energy(particle, incident_energies, split_energy_position, X_train):
    if split_energy_position == '' or split_energy_position is None:
        return X_train

    if 'photon' in particle:
        if split_energy_position == 'le12':
            positions = (-1, np.power(2,12))
            mask = (incident_energies >= positions[0]) & (incident_energies <= positions[1])
        elif split_energy_position == 'ge12':
            positions = (np.power(2,12), max(incident_energies)*2)
            mask = (incident_energies >= positions[0]) & (incident_energies <= positions[1])
        elif split_energy_position == 'ge12le18':
            positions = (np.power(2,12), np.power(2,18))
            mask = (incident_energies >= positions[0]) & (incident_energies <= positions[1])
        elif split_energy_position == 'ge18':
            positions = (np.power(2,18), max(incident_energies)*2)
            mask = (incident_energies >= positions[0]) & (incident_energies <= positions[1])
        else:
            assert(0)
    else:
        pass

    X_train = X_train[mask.flatten()]
    return X_train
