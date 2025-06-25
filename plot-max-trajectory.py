import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fields = np.load(f'fields_mode_8_off_1000000_True_1.0_33.npy')
# fields = np.load(f'fields_mode_8_off_500000_True_1.0_33.npy')

print(f'fields shape: {fields.shape}')

# fields = fields[::2]
fields = fields[:fields.shape[0]//2]  # Take only the first half of the data    
intensities = np.abs(fields)**2

# find the indices of the maximum intensity along the last two axes
max_indices = np.zeros((intensities.shape[0], 2), dtype=int)
for i in range(intensities.shape[0]):
    max_indices[i] = np.unravel_index(np.argmax(intensities[i],), intensities[i].shape)
    # print(f'max indices for trajectory {i}: {max_indices}')


# Plotting the maximum trajectory
# Only plot 2d trajectory, ignore x-axis

z = np.arange(intensities.shape[0])

fig = plt.figure(figsize=(18, 6))
plt.plot(z, max_indices[:, 1], c='blue', marker='o', label='Max Intensity Trajectory')
plt.show()

