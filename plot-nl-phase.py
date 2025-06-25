import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fields = np.load(f'nl_phase_mode_6_off_160000_4.0_35.npy')

print(f'fields shape: {fields.shape}')


plt.imshow(fields, cmap='turbo', interpolation='bilinear')
plt.show()

