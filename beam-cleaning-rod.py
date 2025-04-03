import numpy as np
import matplotlib.pyplot as plt
import torch
import os, time

from src.util import plot_index_profile, plot_beam_intensity, plot_energy_evolution
from src.simulation import Domain, Fiber, Input, run
from src.modes import calculate_modes, decompose_modes, n_to_lm

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


np.random.seed(45)

c = 299792458

NA = 0.25
n_clad = 1.45
n_core = np.sqrt(NA**2 + n_clad**2)
n2 = 2.3e-20

wvl0 = 775e-9

radius = 450e-6
Lx, Ly = 2000e-6, 2000e-6
Nx, Ny = 4096, 4096
ds_factor = 4

total_z = 0.15
dz = 1e-6

intensity = 4.07e13
total_power = 144e3
# intensity = 4.07e1
amp = np.sqrt(intensity)

print(f'The total power is {total_power} W')

num_samples = 200


cx = 400e-6
cy = 0
domain = Domain(Lx, Ly, Nx, Ny, device=device)
input_beam = Input(domain, wvl0, n_core, n_clad, type="gaussian", noise=True, beam_radius=50e-6, cx=cx, cy=cy, amp=amp, radius=radius, device=device)
fiber = Fiber(domain, n_core, n_clad, total_z, dz, n2=n2, radius=radius, disorder=False, device=device)

indices = fiber.n.cpu().numpy()
input_field = input_beam.field.cpu().numpy()


start_time = time.time()

output, fields, energies = run(domain, input_beam, fiber, wvl0, dz=dz, mode_decompose=False, nonlinear=nonlinear)
print(f'Elapsed time: {time.time() - start_time}')

output = output.cpu().numpy()
fields = fields[:, ::ds_factor, ::ds_factor]
energies = energies[:, ::ds_factor, ::ds_factor]

x_window = fields.shape[1] // 4
y_window = fields.shape[2] // 4

fields = fields[:, (fields.shape[1]//2 - x_window):(fields.shape[1]//2 + x_window), (fields.shape[2]//2 - y_window):(fields.shape[2]//2 + y_window)]
fields = fields[:, (fields.shape[1]//2 - x_window):(fields.shape[1]//2 + x_window), (fields.shape[2]//2 - y_window):(fields.shape[2]//2 + y_window)]
fields = fields.cpu().numpy()
fiber_index = fiber.n.cpu().numpy()

np.save('GRIN_rod_indices.npy', fiber_index)
np.save(f'output_rod_off_power_{total_power}.npy', output)
np.save(f'fields_rod_off_power_{total_power}.npy', fields)

plot_index_profile(fiber_index)
plot_beam_intensity(input_field, indices=indices, interpolation="bilinear")
plot_beam_intensity(output, indices=indices, interpolation="bilinear")
plot_energy_evolution(energies, dz=dz)
plt.show()
