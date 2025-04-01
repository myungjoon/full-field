import numpy as np
import matplotlib.pyplot as plt

fields = np.load('E_array_NL_512_512_1e-05.npy')

N = 5
a = 26e-6
L = 160e-6
extent = [-L/2, L/2, -L/2, L/2]


theta = np.linspace(0, 2*np.pi, 100)
x_boundary = a * np.cos(theta)
y_boundary = a * np.sin(theta)


fig, axes = plt.subplots(2, N, figsize=(17, 9))

for i in range(2*N,N,-1):
    axes[0, i-6].imshow(np.abs(fields[-1*i])**2, cmap='jet', extent=extent)
    axes[0, i-6].plot(x_boundary, y_boundary, 'w--', linewidth=1.5)

for i in range(N,0,-1):
    axes[1, i-1].imshow(np.abs(fields[-1*i])**2, cmap='jet', extent=extent)
    axes[1, i-1].plot(x_boundary, y_boundary, 'w--', linewidth=1.5)

plt.show()
