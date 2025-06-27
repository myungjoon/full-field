import numpy as np
import matplotlib.pyplot as plt

from src.util import plot_3d_profile
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

num_samples = 2000
total_length = 1.0

input_type = "mode"
position = "on"
waveguide_type = "rod"
precision = 'single'
num_grids = 2048

radius = 450e-6
beam_radius = 50e-6
total_power = 500000

# fields = np.load(f'fields_{input_type}_{beam_radius}_{position}_{total_power}_{precision}_{num_grids}.npy')
fields = np.load(f'fields_mode_10_on_1600000.npy')
# fields = fields[:len(fields)//2,:,:]  # Take only the first half of the fields
# fields = fields[::2,:,:]
plot_3d_profile(fields,)
plt.show()