import numpy as np
import matplotlib.pyplot as plt

from src.util import plot_3d_profile
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


num_samples = 100

total_length = 0.10
dz=total_length / num_samples

dz_mode = 6e-6
ds_factor = 4

total_power = 5000
input_type = "gaussian"
position = "on"
waveguide_type = "rod"
seed = 2
unit = 1e-6


radius = 450e-6
input_radius =100e-6
Lx, Ly = 8 * radius, 8 * radius

extent = [-Lx/2/unit, Lx/2/unit, -Ly/2/unit, Ly/2/unit]


fiber_index = np.load(f'./data/GRIN_{waveguide_type}_indices.npy')
power = 1500000
fields = np.load(f'./claude_gaussian_256pixel_{radius}_{input_radius}_{power}_fields_on.npy', )
fiber_index_ds  = fiber_index[::1, ::1]
x_window = fiber_index_ds.shape[0] // 4
y_window = fiber_index_ds.shape[1] // 4
fiber_index_ds = fiber_index_ds[(fiber_index_ds.shape[0]//2 - x_window):(fiber_index_ds.shape[0]//2 + x_window), (fiber_index_ds.shape[1]//2 - y_window):(fiber_index_ds.shape[1]//2 + y_window)]

animation_filename = f'256pixel_{waveguide_type}_{input_type}_{position}_{radius}_{input_radius}_{int(power)}'
plot_3d_profile(fields, indices=fiber_index_ds, radius=radius, filename=animation_filename, extent=extent, interpolation="bilinear")
plt.show()