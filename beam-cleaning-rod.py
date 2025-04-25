import numpy as np
import matplotlib.pyplot as plt
import torch
import os, time, itertools

from src.util import plot_index_profile, plot_beam_intensity, plot_energy_evolution
from src.simulation import Domain, Fiber, Input, run
from src.modes import calculate_modes, decompose_modes, n_to_lm

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


seed = 30
np.random.seed(seed)

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

total_z = 0.20
dz = 5e-6

# total_power = 1.6e3
# input_type = "mode_mixing"
# position = "off"
# waveguide_type = "rod"

mode_decompose = True

input_type = "mode_mixing"
waveguide_type = "rod"
positions = ["off"]
total_powers = [1.6e6]

num_samples = 500
target_modes = (0, 2, 4, 5, 7, 8, 9, 10, 11, 12, 15, 17, 18, 20)
num_modes = len(target_modes)
coefficients = np.zeros((num_modes, 2), dtype=complex)
for i, n in enumerate(target_modes):
    l, m = n_to_lm(n+1)
    print(f'The mode {n+1} is LP{l}{m}')
    if l == 0:
        coefficients[i,0] = 0.1 * np.exp(1j * np.random.random() * 1.0 * np.pi)
    else:
        alpha = np.random.random()
        coefficients[i,0] = alpha * np.exp(1j * np.random.random() * 1.0 * np.pi)
        coefficients[i,1] = (1-alpha) * np.exp(1j * np.random.random() * 1.0 * np.pi)


domain = Domain(Lx, Ly, Nx, Ny, device=device)

combinations = list(itertools.product(positions, total_powers))
# loop for each combination
for position, total_power in combinations:
    print(f'The total power is {total_power} W')
    print(f'The input beam is the combination of several modes')

    if position == "off":
        cy = 300e-6
        cx = 0
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
    elif input_type == "mode":
        l = 0
        m = 1
        input_beam = Input(domain, wvl0, n_core, n_clad, type="mode", power=total_power, radius=radius, l=l, m=m, device=device)
        
    else:
        raise ValueError('Invalid Input Type')

    fiber = Fiber(domain, n_core, n_clad, total_z, dz, n2=n2, radius=radius, disorder=False, device=device)

    index_distribution = fiber.n.cpu().numpy()
    input_field = input_beam.field.cpu().numpy()


    plot_beam_intensity(input_field, indices=index_distribution, interpolation="bilinear")
    plt.show()


    start_time = time.time()
    print(f'The simulation starts.')
    if mode_decompose:
        output, modes, fields, energies, Knls, Kins  = run(domain, input_beam, fiber, wvl0, dz=dz, mode_decompose=mode_decompose,)
    else:
        output, fields, energies, Knls, Kins  = run(domain, input_beam, fiber, wvl0, dz=dz, mode_decompose=mode_decompose,)
    print(f'Elapsed time: {time.time() - start_time}')

    output = output.cpu().numpy()
    fields = fields[:, ::ds_factor, ::ds_factor]

    x_window = fields.shape[1] // 4
    y_window = fields.shape[2] // 4

    fields = fields[:, (fields.shape[1]//2 - x_window):(fields.shape[1]//2 + x_window), (fields.shape[2]//2 - y_window):(fields.shape[2]//2 + y_window)]
    fields = fields[:, (fields.shape[1]//2 - x_window):(fields.shape[1]//2 + x_window), (fields.shape[2]//2 - y_window):(fields.shape[2]//2 + y_window)]
    fields = fields.cpu().numpy()
    fiber_index = fiber.n.cpu().numpy()

    np.save(f'./data/GRIN_{waveguide_type}_indices.npy', fiber_index)
    np.save(f'./data/modes_{waveguide_type}_{input_type}_{position}_{int(total_power)}_{seed}_{dz}.npy', modes)
    np.save(f'./data/input_{waveguide_type}_{input_type}_{position}_{int(total_power)}_{seed}_{dz}.npy', input_field)
    np.save(f'./data/output_{waveguide_type}_{input_type}_{position}_{int(total_power)}_{seed}_{dz}.npy', output)
    np.save(f'./data/fields_{waveguide_type}_{input_type}_{position}_{int(total_power)}_{seed}_{dz}.npy', fields)
    np.save(f'./data/energies_{waveguide_type}_{input_type}_{position}_{int(total_power)}_{seed}_{dz}.npy', energies)
    np.save(f'Knls_{waveguide_type}_{input_type}_{position}_{int(total_power)}_{seed}_{dz}.npy', Knls)
    np.save(f'Kins_{waveguide_type}_{input_type}_{position}_{int(total_power)}_{seed}_{dz}.npy', Kins)

# plot_index_profile(fiber_index)
# plot_beam_intensity(input_field, indices=index_distribution, interpolation="bilinear")
# plot_beam_intensity(output, indices=index_distribution, interpolation="bilinear")
# plot_energy_evolution(energies, dz=dz)

# plt.show()