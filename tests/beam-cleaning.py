import numpy as np
import matplotlib.pyplot as plt
import torch
import os, time

from util import plot_index_profile, plot_beam_intensity, plot_mode_evolution
from simulation import Domain, Fiber, Input, run
from modes import calculate_modes, decompose_modes, n_to_lm

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# np.random.seed(44)
np.random.seed(105)

c = 299792458

n_core = 1.47
n_clad = 1.457
print(f'NA = {np.sqrt(n_core**2 - n_clad**2)}')

wvl0 = 1064e-9 # Wavelength (meter)


radius = 26e-6
Lx, Ly = 150e-6, 150e-6
Nx, Ny = 256, 256
total_z = 1.0
dz = 1e-6

n2 = 2.3e-20
# intensity = 5e13
intensity = 5e14
amp = np.sqrt(intensity)

num_samples = 500

num_mode_input = 10
num_mode_total = 10
coefficients = np.zeros((num_mode_total,2), dtype=complex)
for n in range(num_mode_input):
    l, m = n_to_lm(n+1)
    if l == 0:
        coefficients[n,0] = 1.0 * np.exp(1j * np.random.random() * 0.2 * np.pi)
    else:
        # alpha = np.random.random()
        # coefficients[n,0] = alpha * np.exp(1j * np.random.random() * 0.5 * np.pi)
        coefficients[n,1] = np.exp(1j * np.random.random() * 0.2 * np.pi)

# for n in range(num_mode_input, num_mode_total):
#     l, m = n_to_lm(n+1)
#     if l == 0:
#         coefficients[n,0] = 0.1 * np.exp(1j * np.random.random() * 0.3 * np.pi)
#     else:
#         coefficients[n,0] = 0.1 * np.exp(1j * np.random.random() * 0.3 * np.pi)
#         coefficients[n,1] = 0.1 * np.exp(1j * np.random.random() * 0.3 * np.pi)

input_type = "gaussian"

domain = Domain(Lx, Ly, Nx, Ny, device=device)

if input_type == "random":
    input_beam = Input(domain, wvl0, n_core, n_clad, type=input_type, amp=amp, radius=radius,
                        num_mode=num_mode_input, coefficients=coefficients, device=device)
elif input_type == "gaussian":
    input_beam = Input(domain, wvl0, n_core, n_clad, type="gaussian", noise=True, beam_radius=15e-6, amp=amp, radius=radius, device=device)
else:
    raise ValueError('Invalid Input Type')

fiber = Fiber(domain, n_core, n_clad, total_z, dz, n2=n2, radius=radius, disorder=False, device=device)

indices = fiber.n.cpu().numpy()
input_field = input_beam.field.cpu().numpy()

total_power = np.sum(np.abs(input_field)**2)
print(f'Total power: {total_power:.4e}')

# Run simulaton
nonlinear = True
if nonlinear:
    print(f'Running Nonlinear simulation')
else:
    print(f'Running Linear simulation')
start_time = time.time()

output, modes, fields, energies = run(domain, input_beam, fiber, wvl0, dz=dz, mode_decompose=True, nonlinear=nonlinear)
print(f'Elapsed time: {time.time() - start_time}')

output = output.cpu().numpy()
fields = fields.cpu().numpy()
fiber_index = fiber.n.cpu().numpy()

np.save('fiber_GRIN.npy', fiber_index)

if nonlinear:
    np.save(f'fiber_modes_coeffs_nonlinear_{input_type}_high.npy', modes)
    np.save(f'fiber_output_nonlinear_{input_type}_high.npy', output)
    np.save(f'fiber_fields_nonlinear_{input_type}_high.npy', fields)
else:
    np.save('fiber_modes_linear.npy', modes)
    np.save('fiber_output_linear.npy', output)
    np.save('fiber_fields_linear.npy', fields)

plot_index_profile(fiber_index)
plot_beam_intensity(input_field, indices=indices, interpolation="bilinear")
plot_beam_intensity(output, indices=indices, interpolation="bilinear")
plot_mode_evolution(modes)
plt.show()
