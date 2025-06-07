import numpy as np
import matplotlib.pyplot as plt

# default fontsize as 15
plt.rcParams.update({'font.size': 15})

modes_arr = np.load('modes_mode_5_off_160000_False_1.0_100.npy')
modes_arr = modes_arr**2


z = np.linspace(0, 50.0, modes_arr.shape[0])  # Assuming modes_arr has shape (num_samples, num_modes)

plt.figure(figsize=(7, 6))
for i in range(10):
    plt.plot(z, modes_arr[:,i], label=f'Mode {i+1}')

# plt.ylim(0, 1.0)
plt.xlabel('Propagation Distance (cm)')
plt.ylabel('Mode Energy')
plt.legend()
plt.show()

