import numpy as np
import matplotlib.pyplot as plt

from src.util import make_3d_animation, plot_3d_profile, plot_input_and_output_beam, plot_mode_evolution
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


num_samples = 100

total_length = 0.15
dz=total_length / num_samples

dz_mode = 1e-5
ds_factor = 2

total_power = 160000
input_type = "mode_mixing"
position = "off"
waveguide_type = "rod"
seed = 30

Lx, Ly = 2000e-6, 2000e-6
unit = 1e-6

extent = [-Lx/unit, Lx/unit, -Ly/unit, Ly/unit]
radius = 450e-6 / unit

fiber_index = np.load(f'./data/GRIN_{waveguide_type}_indices.npy')
input_field = np.load(f'./data/input_{waveguide_type}_{input_type}_{position}_{int(total_power)}_{seed}_{dz_mode}.npy', )
output_field = np.load(f'./data/output_{waveguide_type}_{input_type}_{position}_{int(total_power)}_{seed}_{dz_mode}.npy', )
fields = np.load(f'./data/fields_{waveguide_type}_{input_type}_{position}_{int(total_power)}_{seed}_{dz_mode}.npy', )
energies = np.load(f'./data/energies_{waveguide_type}_{input_type}_{position}_{int(total_power)}_{seed}_{dz_mode}.npy', )
modes = np.load(f'./data/modes_{waveguide_type}_{input_type}_{position}_{int(total_power)}_{seed}_{dz_mode}.npy', )

fiber_index_ds = fiber_index[::1, ::1]
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