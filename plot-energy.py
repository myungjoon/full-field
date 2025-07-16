import numpy as np
import matplotlib.pyplot as plt
from src.util import make_3d_animation

input_type = 'mode'
waveguide_radius = 450e-6
precision = 'single'
position = 'off2'
num_grids = 4096
beam_radius = 50e-6  # in meters
total_power = 1600
dz = 1e-5

propagation_length = 0.15  # in meters

filename1 = f'energies_custom_2.5e-05_on_500000_double_2048_1e-05_0.15.npy'
filename2 = f'energies_custom_2.5e-05_on_200000_double_2048_1e-05_0.15.npy'
filename3 = f'energies_custom_2.5e-05_on_250000_double_2048_1e-05_0.15.npy'
filename4 = f'energies_custom_2.5e-05_on_300000_double_2048_1e-05_0.15.npy'
filenames = [filename1, filename2, filename3, filename4]
# filenames = [filename1]


energies_arr = []
for filename in filenames:
    energies = np.load(filename)
    energies_arr.append(energies)
energies_arr = np.array(energies_arr)

# plot the energies
# plt.figure(figsize=(10, 6))
for i, energies in enumerate(energies_arr):
    energy_loss = energies[0] - energies[-1]
    print(f'Energy loss for run {i+1}: {energy_loss / energies[0]:.4f}')

    plt.figure(figsize=(10, 6))
    z = np.arange(energies.shape[0])
    plt.plot(z, energies, label=f'Run {i+1}', marker='o')

plt.show()