import numpy as np
import matplotlib.pyplot as plt

from src.util import *
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

input_type = 'mode_mixing'
position = 'on'
total_power = 16000000
ds_factor = 4

dz=1e-6


fiber_index = np.load('./GRIN_rod_indices.npy')
# input_field = np.load(f'./data/input_rod_{input_type}_{position}_{int(total_power)}.npy', )
# output_field = np.load(f'./data/output_rod_{input_type}_{position}_{int(total_power)}.npy', )
fields = np.load(f'./fields_rod_{input_type}_{position}_{int(total_power)}.npy', )
# energies = np.load(f'./data/energies_rod_{input_type}_{position}_{int(total_power)}.npy', )
# modes = np.load(f'./modes_rod_{input_type}_{position}_{int(total_power)}.npy', )
# fields = np.load('./data/fields_rod_mode_mixing_on_100.npy')

fiber_index_ds = fiber_index[::ds_factor, ::ds_factor]

x_window = fiber_index_ds.shape[0] // 4
y_window = fiber_index_ds.shape[1] // 4
fiber_index_ds = fiber_index_ds[(fiber_index_ds.shape[0]//2 - x_window):(fiber_index_ds.shape[0]//2 + x_window), (fiber_index_ds.shape[1]//2 - y_window):(fiber_index_ds.shape[1]//2 + y_window)]

animation_filename = f'rod_{input_type}_{position}_{int(total_power)}'
# plot_input_and_output_beam(input_field, output_field, indices=fiber_index, interpolation="bilinear")
make_3d_animation(fields, indices=fiber_index_ds, filename=animation_filename)
# plot_energy_evolution(energies, dz=dz)
# plot_mode_evolution(modes, dz=dz)
# plot_3d_profile(fields)


# plot_input_and_output_modes(input_field, output_field)

plt.show()