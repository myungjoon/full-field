import numpy as np
import matplotlib.pyplot as plt
from src.util import make_3d_animation

input_type = 'mode'
waveguide_radius = 450e-6
precision = 'single'
position = 'off2'
num_grids = 4096
beam_radius = 50e-6  # in metersss
total_power = 1600
dz = 1e-5

propagation_length = 0.15  # in meters

filename1 = f'fields_custom_2.5e-05_off_160000_double_2048_1e-05_0.15.npy'
filename2 = f'fields_custom_2.5e-05_off_250000_double_2048_1e-05_1.0.npy'
filename3 = f'fields_custom_2.5e-05_off_300000_double_2048_1e-05_1.0.npy'
filename4 = f'fields_custom_2.5e-05_off_160000_double_2048_1e-05_1.0.npy'
# filenames = [filename1, filename2, filename3, filename4]
filenames = [filename1]

for filename in filenames:
    fields = np.load(filename)
    animation_filename = f'./results/{filename}'.replace('.npy', '.mp4')
    make_3d_animation(fields, waveguide_radius=waveguide_radius, propagation_length=propagation_length*100,
                    filename=animation_filename, roi=beam_radius*3, interpolation="bilinear")
