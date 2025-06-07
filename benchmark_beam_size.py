import numpy as np
import matplotlib.pyplot as plt
import torch
import os, time, argparse

from src.util import plot_index_profile, plot_beam_intensity, plot_beam_intensity_and_phase, make_3d_animation, print_total_power
from src.simulation import Domain, Fiber, Input, run
from src.modes import calculate_modes, decompose_modes, n_to_lm, calculate_effective_mode_area
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# arguments for total_power, beam_radius, position
parser = argparse.ArgumentParser(description='Simulation parameters')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--input_type', type=str, default='mode', choices=['gaussian', 'mode', 'fundamental'], help='Input type')
parser.add_argument('--num_modes', type=int, default=20, help='Number of input modes')
parser.add_argument('--total_power', type=float, default=5e3, help='Total power (W)')
parser.add_argument('--beam_radius', type=float, default=50e-6, help='Beam radius (m)')
parser.add_argument('--position', type=str, default='on', choices=['on', 'off'], help='Beam position')
parser.add_argument('--disorder', type=bool, default=False, help='Disorder in the fiber')
parser.add_argument('--precision', type=str, default='single', choices=['single', 'double'], help='Precision of the simulation')
parser.add_argument('--num_pixels', type=int, default=32, help='Number of pixels for the phase map')
parser.add_argument('--scale', type=float, default=8.0, help='Scale factor for the input beam')
parser.add_argument('--in_phase', type=bool, default=False, help='Input beam in phase')
parser.add_argument('--device_id', type=int, default=1, help='Device ID for CUDA')

args = parser.parse_args()
print(f'Input type: {args.input_type}')
if args.input_type == 'mode':
    print('Using mode input')
    print(f'Number of modes: {args.num_modes}')
elif args.input_type == 'gaussian':
    print('Using Gaussian beam input')
    print(f'Beam radius: {args.beam_radius * 1e6} um')
elif args.input_type == 'fundamental':
    print('Using fundamental mode input')
else:
    raise ValueError('Invalid input type. Choose either "gaussian" or "mode".')


print(f'Total power: {args.total_power * 1e-3} kW')
print(f'Beam position: {args.position}')
print(f'Disorder: {args.disorder}')
print(f'Precision: {args.precision}')
print(f'Number of pixels: {args.num_pixels}x{args.num_pixels}')
print(f'Input in phase: {args.in_phase}')
print(f'Scale factor: {args.scale}')
seed = args.seed
input_type = args.input_type
total_power = args.total_power
position = args.position
disorder = args.disorder
precision = args.precision
num_pixels = args.num_pixels
beam_radius = args.beam_radius
num_modes = args.num_modes
in_phase = args.in_phase
scale = args.scale
device_id = args.device_id

np.random.seed(seed)

device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')

c = 299792458
wvl0 = 775e-9

# Fiber parameters
NA = 0.25
n_clad = 1.45
n_core = np.sqrt(NA**2 + n_clad**2)
n2 = 2.3e-20
fiber_radius = 20e-6
propagation_length = 1.0

# Simulation domain parameters
Lx, Ly = 4*fiber_radius, 4*fiber_radius
unit = 1e-6
Nx, Ny = 512, 512
print(f'The grid size is {Nx}x{Ny}')
ds_factor = 4 if Nx >= 2048 else 1 # downsampling factor for memory efficiency
dz = 1e-5

mode_decompose = False
domain = Domain(Lx, Ly, Nx, Ny, precision=precision, device=device)

fiber = Fiber(domain, n_core, n_clad, propagation_length, dz, n2=n2, radius=fiber_radius, precision=precision, device=device)
fiber_indices = fiber.n.cpu().numpy()

extent = [-Lx/2/unit, Lx/2/unit, -Ly/2/unit, Ly/2/unit]

fiber_index = fiber.n.cpu().numpy()
num_field_sample = 500
num_mode_sample = 2000

if position == "off":
    cy = fiber_radius/4 
    cx = 0
else:
    cx = 0
    cy = 0

pixels = (num_pixels, num_pixels)

if input_type in ['mode', 'gaussian']:
    input_beam = Input(domain, wvl0, n_core, n_clad, 
                        type=input_type, cx=cx, cy=cy, pixels=pixels, in_phase=in_phase,
                        power=total_power, precision=precision, scale=scale,
                        beam_radius=beam_radius, num_modes=num_modes, fiber_radius=fiber_radius, device=device)
elif input_type == 'fundamental':
    input_beam = Input(domain, wvl0, n_core, n_clad, 
                        type=input_type, power=total_power, precision=precision, fiber_radius=fiber_radius, device=device)
else:
    raise ValueError('Invalid input type. Choose either "gaussian", "mode", or "fundamental".')


input_field = input_beam.field.cpu().numpy()

dx = domain.Lx / domain.Nx
dy = domain.Ly / domain.Ny

A_eff = calculate_effective_mode_area(input_field, dx, dy)
nl_phase_analytic = 2 * np.pi * n2 * total_power / (wvl0 * A_eff) * propagation_length
print(f'The effective mode area is {A_eff * 1e12:.2f} um^2')


# plot_beam_intensity_and_phase(input_field, indices=fiber_indices, extent=extent, interpolation="bilinear")
# plt.show()
print_total_power(domain, input_field)

print(f'The simulation starts.')
fields, energies, modes, nl_phase  = run(domain, input_beam, fiber, wvl0, dz=dz, num_field_sample=num_field_sample,
                                          num_mode_sample=num_mode_sample, precision=precision,
                                            disorder=disorder, ds_factor=ds_factor, calculate_nl_phase=True)

print(f'The analytical nonlinear phase is {nl_phase_analytic:.2f} rad')
print(f'The numerical nonlinear phase is {nl_phase:.2f} rad')

fields = fields.cpu().numpy()
fiber_index_ds = fiber.n[::ds_factor, ::ds_factor]
np.save(f'fiber_{fiber_radius}_indices.npy', fiber_index)

# if mode_decompose:
#     np.save(f'modes_{fiber_radius}_{beam_radius}_{position}_{int(total_power)}.npy', modes)

# plot_beam_intensity(input_field, indices=fiber_index, interpolation="bilinear")
if input_type == 'mode':
    np.save(f'input_{input_type}_{num_modes}_{position}_{int(total_power)}_{in_phase}_{scale}_{seed}.npy', input_field)
    np.save(f'fields_{input_type}_{num_modes}_{position}_{int(total_power)}_{in_phase}_{scale}_{seed}.npy', fields)
    np.save(f'modes_{input_type}_{num_modes}_{position}_{int(total_power)}_{in_phase}_{scale}_{seed}.npy', modes)
    animation_filename = f'./results/{input_type}_{num_modes}_{position}_{int(total_power)}_{in_phase}_{scale}_{seed}'
else:
    np.save(f'input_{fiber_radius}_{beam_radius}_{position}_{int(total_power)}.npy', input_field)
    np.save(f'fields_{fiber_radius}_{beam_radius}_{position}_{int(total_power)}.npy', fields)

    if disorder:
        animation_filename = f'./results/{input_type}_{beam_radius}_{position}_{num_pixels}x{num_pixels}_disorder_{int(total_power)}'
    else:
        animation_filename = f'./results/{input_type}_{beam_radius}_{position}_{num_pixels}x{num_pixels}_{int(total_power)}'


make_3d_animation(fields, radius=fiber_radius, filename=animation_filename, interpolation="bilinear")