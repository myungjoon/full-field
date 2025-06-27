import numpy as np
import matplotlib.pyplot as plt

def create_phase_map(N_grids, N_pixels=2, phases=None):
    phase_map = np.zeros((N_grids, N_grids))

    N_half = N_grids // 2
    # macropixel size
    macropixel_size = N_half // N_pixels
    core_start = N_half // 2

    phases = np.random.random((N_pixels, N_pixels)) *  2 * np.pi
    
    for i in range(N_pixels):
        for j in range(N_pixels):
            row_start = core_start + i * macropixel_size
            row_end = row_start + macropixel_size
            col_start = core_start + j * macropixel_size
            col_end = col_start + macropixel_size
            
            phase_map[row_start:row_end, col_start:col_end] = phases[i, j]
    
    return phase_map


def gaussian_beam(w, X, Y, N_grids, cx=0, cy=0,):
    R = np.sqrt((X - cx)**2 + (Y - cy)**2)
    field = np.exp(-R**2 / (w**2))
    # field = normalize_field_to_power(field.to(self.device), dx, dy, self.power)
    return field

N_grids = 512
beam_radius = 50e-6
cx, cy = 0., 0.
N_pixels = 16

Lx, Ly = 4 * beam_radius, 4 * beam_radius
Nx, Ny = N_grids, N_grids

x = np.linspace(-Lx/2, Lx/2, Nx,)
y = np.linspace(-Ly/2, Ly/2, Ny,)
X, Y = np.meshgrid(x, y, indexing='ij')

gaussian = gaussian_beam(beam_radius, X, Y, N_grids, cx=cx, cy=cy)
phase_map = create_phase_map(N_grids, N_pixels=N_pixels)
phases = np.exp(1j * phase_map)

input_beam = gaussian * phases
input_beam = np.fft.fftshift(np.fft.fft2(input_beam))

intensity = np.abs(input_beam)**2

plt.figure(1)
plt.imshow(phase_map, cmap='turbo', interpolation='nearest')
plt.colorbar()

plt.figure(2)
plt.imshow(np.abs(gaussian)**2, cmap='turbo', interpolation='nearest')

plt.figure(3)
plt.imshow(intensity, cmap='turbo', interpolation='nearest')
plt.colorbar()

plt.show()