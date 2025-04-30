import numpy as np
import matplotlib.pyplot as plt

from src.util import make_3d_animation
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


num_samples = 100

total_length = 0.10
dz=total_length / num_samples

dz_mode = 5e-6
ds_factor = 2

total_power = 63000
input_type = "mode_mixing"
position = "on"
waveguide_type = "fiber"
seed = 2

Lx, Ly = 100e-6, 100e-6
unit = 1e-6

extent = [-Lx/2/unit, Lx/2/unit, -Ly/2/unit, Ly/2/unit]
extent = [-Lx/2/unit, Lx/2/unit, -Ly/2/unit, Ly/2/unit]
radius = 26e-6 / unit

fiber_index = np.load(f'./data/GRIN_{waveguide_type}_indices.npy')
fields = np.load(f'./initial_field_arr_21.npy', )


fiber_index_ds = fiber_index[::4, ::4]

x_window = fiber_index_ds.shape[0] // 4
y_window = fiber_index_ds.shape[1] // 4
fiber_index_ds = fiber_index_ds[(fiber_index_ds.shape[0]//2 - x_window):(fiber_index_ds.shape[0]//2 + x_window), (fiber_index_ds.shape[1]//2 - y_window):(fiber_index_ds.shape[1]//2 + y_window)]

animation_filename = f'{waveguide_type}_{input_type}_{position}_{int(total_power)}'
# plot_input_and_output_beam(input_field, output_field, radius=radius, indices=fiber_index, extent=extent, interpolation="bilinear")
make_3d_animation(fields, indices=fiber_index_ds, radius=radius, filename=animation_filename, extent=extent, interpolation="bilinear")
# plot_energy_evolution(energies, dz=dz)
# plot_mode_evolution(modes, dz=dz_mode)
# plot_3d_profile(fields)
# plot_input_and_output_modes(input_field, output_field)

plt.show()