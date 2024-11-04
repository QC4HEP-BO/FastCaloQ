# From https://gitlab.cern.ch:8443/atlas-simulation-fastcalosim/fastcaloml/CaloINN/-/blob/main/src/plotting_helpers/observables.py?ref_type=heads

import numpy as np
from pdb import set_trace

# Compute layer_boundaries (number of voxels in current and preceding layers)
# layer_boundaries = [0, size_layer_0, size_layer_0 + size_layer_1, ...]
def calc_shower_mean(layer_energy, layer_boundaries, layer, coordinates, direction):
    """Computes the mean of the shower in eta or phi direction for a given layer."""
    
    eta = coordinates[0][layer_boundaries[layer]:layer_boundaries[layer+1]]
    phi = coordinates[1][layer_boundaries[layer]:layer_boundaries[layer+1]]
    
    eta_mean = np.sum(layer_energy * eta, axis=-1) / (layer_energy.sum(axis=-1)+1.e-16)
    phi_mean = np.sum(layer_energy * phi, axis=-1) / (layer_energy.sum(axis=-1)+1.e-16)

    # Compute the mean eta    
    if direction == "eta":
        return eta_mean
    
    elif direction == "phi":
        return phi_mean
    
    elif direction == "both":
        return eta_mean, phi_mean

    else:
        raise ValueError("Invalid direction")
    
def calc_shower_std(layer_energy, layer_boundaries, layer, coordinates, direction):
    """Computes the std of the shower in eta or phi direction for a given layer."""
    
    mean = calc_shower_mean(layer_energy, layer_boundaries, layer, coordinates, 'both')
    
    eta = coordinates[0][layer_boundaries[layer]:layer_boundaries[layer+1]]
    phi = coordinates[1][layer_boundaries[layer]:layer_boundaries[layer+1]]
    
    discriminant = np.sum(layer_energy * eta * eta, axis=-1) / (layer_energy.sum(axis=-1)+1.e-16) - mean[0]**2
    discriminant[discriminant < 0] = 0
    eta_std = np.sqrt(discriminant)

    discriminant = np.sum(layer_energy * phi * phi, axis=-1) / (layer_energy.sum(axis=-1)+1.e-16) - mean[1]**2
    discriminant[discriminant < 0] = 0
    phi_std = np.sqrt(discriminant)

    if direction == "eta":
        return eta_std
    
    elif direction == "phi":
        return phi_std
    
    elif direction == "both":
        return eta_std, phi_std
    
    else:
        raise ValueError("Invalid direction")

def calc_flat_energy_distribution(x, c, layer_boundaries, layer=None):
    """Computes the energy distribution of the shower"""
    
    if layer is None:
        return x.flatten()
    
    return x[:, layer_boundaries[layer]:layer_boundaries[layer+1]].flatten()
 
def calc_energy(x, c, layer_boundaries, layer=None):
    """Computes the energy of a given layer or of the whole shower"""
    
    if layer is None:
        return np.sum(x, axis=-1)
    
    return np.sum(x[:, layer_boundaries[layer]:layer_boundaries[layer+1]], axis=-1)

def calc_etot_over_einc(x, c, layer_boundaries):
    """Computes the total energy of the shower over the incident energy"""
    
    return calc_energy(x, c, layer_boundaries, layer=None) / c[:, 0]

