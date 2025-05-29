import numpy as np
import matplotlib.pyplot as plt

modes_arr = np.load('modes_mode_5_on_500000.npy')

plt.figure(figsize=(15, 6))
for i in range(modes_arr.shape[1]):
    plt.plot(modes_arr[:,i], label=f'Mode {i+1}')

plt.legend()
plt.show()