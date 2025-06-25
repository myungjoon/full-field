import numpy as np
import matplotlib.pyplot as plt

def plot_nl_phase(nl_phase, interpolation="bilinear"):
    plt.figure()
    plt.imshow(nl_phase, cmap='turbo', interpolation=interpolation)
    plt.colorbar()

position = 'on'

phases_scale_1 = np.load(f'./nl_phase_mode_6_{position}_160000_1.0_35.npy')
phases_scale_2 = np.load(f'./nl_phase_mode_6_{position}_160000_2.0_35.npy')
phases_scale_4 = np.load(f'./nl_phase_mode_6_{position}_160000_4.0_35.npy')

vmax = max(np.max(phases_scale_1), np.max(phases_scale_2), np.max(phases_scale_4))
vmin = min(np.min(phases_scale_1), np.min(phases_scale_2), np.min(phases_scale_4))

# Check the integration
print(f'summation of phases_scale_1: {np.sum(phases_scale_1)}')
print(f'summation of phases_scale_2: {np.sum(phases_scale_2)}')
print(f'summation of phases_scale_4: {np.sum(phases_scale_4)}')

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(phases_scale_1, cmap='turbo', vmin=vmin, vmax=vmax, interpolation='bilinear')

plt.subplot(1, 3, 2)
plt.imshow(phases_scale_2, cmap='turbo', vmin=vmin, vmax=vmax, interpolation='bilinear')

plt.subplot(1, 3, 3)
plt.imshow(phases_scale_4, cmap='turbo', vmin=vmin, vmax=vmax, interpolation='bilinear')

plt.colorbar()




plt.show()