import numpy as np
import matplotlib.pyplot as plt
from src.util import make_3d_animation


input_type = 'mode'
fiber_radius = 25e-6
propagation_length = 1.0  # in meters
precision = 'single'
position = 'off2'
num_grids = 4096
beam_radius = 25e-6  # in meters
total_power = 1600
dz = 5e-6

# gaussian_trajectory = np.load(f'trajectory_{input_type}_{beam_radius}_{position}_{total_power}_{precision}_{num_grids}_{dz}.npy')

filename = f'fields_mode_5e-05_on_160000_single_256_1e-05.npy'
print(f'filename : {filename}')

# plt.figure(figsize=(10, 6))
# z = np.arange(gaussian_trajectory.shape[0])
# plt.plot(z, gaussian_trajectory[:, 1], c='blue', marker='o', label='Max Intensity Trajectory')
# plt.grid()
# plt.legend()
# plt.show()

fields = np.load(filename)
animation_filename = f'./results/{filename}'.replace('.npy', '.mp4')
make_3d_animation(fields, radius=fiber_radius, propagation_length=propagation_length*100,
                   filename=animation_filename, roi=None, interpolation="bilinear")
