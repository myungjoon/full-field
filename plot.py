import numpy as np
import matplotlib.pyplot as plt
from src.util import make_3d_animation

input_type = 'mode'
waveguide_radius = 450e-6
propagation_length = 1.0  # in meters
precision = 'single'
position = 'off2'
num_grids = 4096
beam_radius = 50e-6  # in meters
total_power = 1600
dz = 1e-5



filename = f'fields_custom_2.5e-05_on_160000_double_2048_1e-05.npy'
print(f'filename : {filename}')


# gaussian_trajectory = np.load(f'trajectory_{input_type}_{beam_radius}_{position}_{total_power}_{precision}_{num_grids}.npy')
# plt.figure(figsize=(10, 6))
# z = np.arange(gaussian_trajectory.shape[0])
# plt.plot(z, gaussian_trajectory[:, 1], c='blue', marker='o', label='Max Intensity Trajectory')
# plt.grid()
# plt.legend()
# plt.show()

fields = np.load(filename)
animation_filename = f'./results/{filename}'.replace('.npy', '.mp4')
make_3d_animation(fields, waveguide_radius=waveguide_radius, propagation_length=propagation_length*100,
                   filename=animation_filename, roi=beam_radius*3, interpolation="bilinear")
