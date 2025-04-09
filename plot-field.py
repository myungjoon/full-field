import numpy as np
import matplotlib.pyplot as plt

   
data = np.load('fields_rod_off_power_1600000.0.npy')
output = np.load('output_rod_off_power_1600000.0.npy')
Knls = np.load('Knls_rod_off_power_1600000.0.npy')
Kins = np.load('Kins_rod_off_power_1600000.0.npy')


print(data.shape)

def plot_field_intensity(field, indices=None, interpolation="bilinear"):
    plt.figure()
    plt.imshow(np.abs(field**2), cmap='turbo', interpolation=interpolation)
    if indices is not None:
        plt.imshow(indices, cmap='gray', alpha=0.5, interpolation=interpolation)
    plt.colorbar()

plt.figure()
plt.plot(np.abs(Knls), label='Knls')
plt.plot(np.abs(Kins), label='Kins')
plt.title('Knls vs. Kins')
plt.legend()

plot_field_intensity(data[0, :, :], indices=None, interpolation="bilinear")
plot_field_intensity(data[-1, :, :], indices=None, interpolation="bilinear")
plot_field_intensity(data[-2, :, :], indices=None, interpolation="bilinear")
plot_field_intensity(data[-3, :, :], indices=None, interpolation="bilinear")

# plt.imshow(np.abs(data[-5, :, :]**2), cmap='turbo', interpolation='bilinear')
plt.show()