import numpy as np
import matplotlib.pyplot as plt
import torch
import os, time

from util import plot_index_profile, plot_beam_intensity, plot_mode_evolution
from simulation import Domain, Fiber, Input, run
from modes import calculate_modes, decompose_modes, n_to_lm

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
np.random.seed(40)

c = 299792458

radius = 450e-6
Lx, Ly = 3000e-6, 3000e-6
Nx, Ny = 2048, 2048
total_z = 0.15
dz=5e-6

NA = 0.25
n_clad = 1.45
n_core = np.sqrt(NA**2 + n_clad**2)
n2 = 2.3e-20

cx, cy = 0, 350e-6
beam_radius = 50e-6

wvl0 = 775e-9 # Wavelength (meter)

num_samples = 200

domain = Domain(Lx, Ly, Nx, Ny, device=device)

input_beam = Input(domain, wvl0, n_core, n_clad, type="gaussian", noise=False, beam_radius=beam_radius, amp=1.0, radius=radius, device=device)

fiber = Fiber(domain, n_core, n_clad, total_z, dz, n2=n2, radius=radius, device=device)

indices = fiber.n.cpu().numpy()
input_field = input_beam.field.cpu().numpy()

# Run simulaton
nonlinear = False
if nonlinear:
    print(f'Running Nonlinear simulation')
else:
    print(f'Running Linear simulation')
start_time = time.time()

output, fields, energies = run(domain, input_beam, fiber, wvl0, dz=dz, mode_decompose=False, nonlinear=nonlinear)
print(f'Elapsed time: {time.time() - start_time}')
fields = fields[:, ::4, ::4]
output = output.cpu().numpy()
fields = fields.cpu().numpy()
#energies = energies.cpu().numpy()
rod_index = fiber.n.cpu().numpy()

np.save('rod_GRIN.npy', rod_index)
np.save('self-imaging_fields.npy', fields)


plot_beam_intensity(input_field, indices=indices)
plot_beam_intensity(output, indices=indices)
plt.show()
