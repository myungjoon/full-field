import numpy as np
import matplotlib.pyplot as plt
import torch
import os, time

from src.util import plot_index_profile, plot_beam_intensity, plot_energy_evolution
from src.simulation import Domain, Fiber, Input, run
from src.modes import calculate_modes, decompose_modes, n_to_lm

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


np.random.seed(46)

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

total_z = 0.50
dz = 1e-5

total_power = 6.0e6
input_type = "mode_mixing"
position = "off"

mode_decompose = True

print(f'The total power is {total_power} W')
print(f'The input beam is the combination of several modes')

num_samples = 500

num_modes = 6
target_modes = (0, 2, 4, 5, 7, 8)
coefficients = np.zeros((num_modes, 2), dtype=complex)
for i, n in enumerate(target_modes):
    l, m = n_to_lm(n+1)
    print(f'The mode {n+1} is LP{l}{m}')
    if l == 0:
        coefficients[n,0] = 0.1 * np.exp(1j * np.random.random() * 1.0 * np.pi)
    else:
        alpha = np.random.random()
        coefficients[i,0] = alpha * np.exp(1j * np.random.random() * 1.0 * np.pi)
        coefficients[i,1] = (1-alpha) * np.exp(1j * np.random.random() * 1.0 * np.pi)

num_l = 5
num_m = 5
num_total_modes = num_l * num_m
input_mode = np.zeros((num_l, num_m), dtype=float)
for i, mode in enumerate(target_modes):
    l, m = n_to_lm(mode+1)
    input_mode[l, m] = coefficients[i, 0].real


domain = Domain(Lx, Ly, Nx, Ny, device=device)


if position == "off":
    cx = 300e-6
    cy = 0
else:
    cx = 0
    cy = 0

if input_type == "mode_mixing":
    input_beam = Input(domain, wvl0, n_core, n_clad, type=input_type, power=total_power, 
                       radius=radius, num_mode=num_modes, cx=cx, cy=cy, coefficients=coefficients, device=device)
elif input_type == "gaussian":

    noise = True
    beam_radius = 50e-6
    input_beam = Input(domain, wvl0, n_core, n_clad, 
                       type="gaussian", cx=cx, cy=cy, 
                       power=total_power, noise=noise, beam_radius=beam_radius, radius=radius, device=device)
else:
    raise ValueError('Invalid Input Type')

fiber = Fiber(domain, n_core, n_clad, total_z, dz, n2=n2, radius=radius, disorder=False, device=device)

index_distribution = fiber.n.cpu().numpy()
input_field = input_beam.field.cpu().numpy()

start_time = time.time()
print(f'The simulation starts.')
if mode_decompose:
    output_field, modes, fields, energies, Knls, Kins  = run(domain, input_beam, fiber, wvl0, dz=dz, mode_decompose=mode_decompose,)
else:
    output_field, fields, energies, Knls, Kins  = run(domain, input_beam, fiber, wvl0, dz=dz, mode_decompose=mode_decompose,)
print(f'Elapsed time: {time.time() - start_time}')

output_field = output_field.cpu().numpy()
fields = fields[:, ::ds_factor, ::ds_factor]

x_window = fields.shape[1] // 4
y_window = fields.shape[2] // 4

fields = fields[:, (fields.shape[1]//2 - x_window):(fields.shape[1]//2 + x_window), (fields.shape[2]//2 - y_window):(fields.shape[2]//2 + y_window)]
fields = fields[:, (fields.shape[1]//2 - x_window):(fields.shape[1]//2 + x_window), (fields.shape[2]//2 - y_window):(fields.shape[2]//2 + y_window)]
fields = fields.cpu().numpy()
fiber_index = fiber.n.cpu().numpy()
# modes = modes.cpu().numpy()
energies = energies.cpu().numpy()


np.save('GRIN_rod_indices.npy', fiber_index)
np.save(f'input_rod_{input_type}_{position}_{int(total_power)}.npy', input_field)
np.save(f'output_rod_{input_type}_{position}_{int(total_power)}.npy', output_field)
np.save(f'fields_rod_{input_type}_{position}_{int(total_power)}.npy', fields)
np.save(f'energies_rod_{input_type}_{position}_{int(total_power)}.npy', energies)
np.save(f'modes_rod_{input_type}_{position}_{int(total_power)}.npy', modes)
np.save(f'Knls_rod_{input_type}_{position}_{int(total_power)}.npy', Knls)
np.save(f'Kins_rod_{input_type}_{position}_{int(total_power)}.npy', Kins)

# plot_index_profile(fiber_index)

# plot_beam_input_and_output(input_field, output_field, indices=index_distribution, interpolation="bilinear")
# plot_input_and_output_modes(input_field, indices=index_distribution, interpolation="bilinear")
# make_3d_animation(fields, fiber_index, indices=index_distribution, interpolation="bilinear", filename=f'animation_rod_{input_type}_{position}_{int(total_power)}.mp4')
# plot_3d_profile(fields, fiber_index, indices=index_distribution, interpolation="bilinear", filename=f'3d_profile_rod_{input_type}_{position}_{int(total_power)}.png')
# plot_energy_evolution(energies, dz=dz)
# plot_mode_evolution(modes, dz=dz)
# plot_beam_intensity(input_field, indices=index_distribution, interpolation="bilinear")
# plot_beam_intensity(output, indices=index_distribution, interpolation="bilinear")


plt.show()
