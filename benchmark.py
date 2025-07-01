import numpy as np
import matplotlib.pyplot as plt
import torch
import os, time, argparse

from src.util import plot_index_profile, plot_beam_intensity, plot_beam_intensity_and_phase, make_3d_animation, print_total_power
from src.simulation import Domain, Waveguide, Input, run
from src.modes import calculate_modes, decompose_modes, n_to_lm

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# arguments for total_power, beam_radius, position
parser = argparse.ArgumentParser(description='Simulation parameters')
parser.add_argument('--seed', type=int, default=61, help='Random seed for reproducibility')
parser.add_argument('--input_type', type=str, default='gaussian', choices=['gaussian', 'mode'], help='Input type')
parser.add_argument('--num_modes', type=int, default=8, help='Number of input modes')
parser.add_argument('--total_power', type=float, default=160e3, help='Total power (W)')
parser.add_argument('--beam_radius', type=float, default=25e-6, help='Beam radius (m)')
parser.add_argument('--position', type=str, default='on', choices=['on', 'off'], help='Beam position')
parser.add_argument('--disorder', type=bool, default=False, help='Disorder in the fiber')
parser.add_argument('--precision', type=str, default='double', choices=['single', 'double'], help='Precision of the simulation')
parser.add_argument('--num_pixels', type=int, default=32, help='Number of pixels for the phase map')
parser.add_argument('--scale', type=float, default=2.0, help='Scale factor for the input beam')
parser.add_argument('--in_phase', type=bool, default=True, help='Input beam in phase')
parser.add_argument('--device_id', type=int, default=1, help='Device ID for CUDA')

args = parser.parse_args()
print(f'Input type: {args.input_type}')
if args.input_type == 'mode':
    print(f'Number of modes: {args.num_modes}')
elif args.input_type == 'gaussian':
    print(f'Beam radius: {args.beam_radius * 1e6} um')    
else:
    raise ValueError('Invalid input type. Choose either "gaussian" or "mode".')


print(f'Precision: {args.precision}')
print(f'Total power: {args.total_power * 1e-3} kW')
print(f'Beam position: {args.position}')
print(f'Disorder: {args.disorder}')
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
scale = args.scale
device_id = args.device_id

np.random.seed(seed)
device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')

c = 299792458
wvl0 = 775e-9

# Fiber parameters
NA = 0.25
n_clad = 1.457
n_core = np.sqrt(NA**2 + n_clad**2)
n2 = 3.2e-20 * 2 # factor 2 for the estimantion of GRIN rod material property
waveguide_radius = 450e-6
propagation_length = 1.0

# Simulation domain parameters
Lx, Ly = 3 * waveguide_radius, 3 * waveguide_radius
unit = 1e-6
Nx, Ny = 2048, 2048
print(f'The grid size is {Nx}x{Ny}')
ds_factor = round(Nx / 512) if Nx >= 512 else 1 # downsampling factor for memory efficiency
dz = 1e-5

mode_decompose = False
domain = Domain(Lx, Ly, Nx, Ny, propagation_length, device=device)
waveguide = Waveguide(domain, n_core, n_clad, n2=n2, radius=waveguide_radius, device=device)
waveguide_indices = waveguide.n.cpu().numpy()

extent = [-Lx/2/unit, Lx/2/unit, -Ly/2/unit, Ly/2/unit]

num_field_sample = 1000
num_mode_sample = 1000

if position == 'off':
    cy = waveguide_radius / 4
    cx = 0
else:
    cx = 0
    cy = 0


print(f'filename : fields_{input_type}_{beam_radius}_{position}_{int(total_power)}_{precision}_{Nx}_{dz}.npy')
print(f'trajectory_{input_type}_{beam_radius}_{position}_{int(total_power)}_{precision}_{Nx}_{dz}.npy')

# modes = np.load('modes_2048_1.npy')
# modes = torch.tensor(modes, dtype=torch.complex64, device=device)
# num_mode = 6

#coefficients with random phase
# input_type = 'custom'
# coefficients = torch.tensor([0.3, 0.2, 0.2, 0.2, 0.1, 0.1])
# coefficients = coefficients.reshape((num_mode,1,1)) * np.exp(1j * np.random.uniform(0, 1.0 * np.pi, (num_mode, 1, 1)))
# coefficients = coefficients.to(device)


# fields = torch.sum(coefficients * modes[:num_mode], dim=0)

cx1 = 0
cy1 = waveguide_radius / 4

cx2 = waveguide_radius / 4
cy2 = 0

gaussian_beam_1 = Input(domain, wvl0, n_core, n_clad, 
                    input_type=input_type, beam_radius=beam_radius, cx=cx1, cy=cy1, 
                    power=total_power, device=device)

gaussian_beam_2 = Input(domain, wvl0, n_core, n_clad,
                    input_type=input_type, beam_radius=beam_radius, cx=cx2, cy=cy2, 
                    power=total_power, device=device)

gaussian_1 = gaussian_beam_1.field
gaussian_2 = gaussian_beam_2.field

fields = gaussian_1 + gaussian_2

input_type = 'custom'
input_beam = Input(domain, wvl0, n_core, n_clad, phase_modulation=False, pixels=(2,2),
                    input_type=input_type, custom_fields=fields, cx=cx, cy=cy, scale=scale, beam_radius=beam_radius,
                    power=total_power, num_modes=num_modes, waveguide_radius=waveguide_radius, device=device)
# input_beam.field = fields

input_field = input_beam.field.cpu().numpy()
plot_beam_intensity_and_phase(input_field, indices=waveguide_indices, extent=extent, interpolation=None)
plt.show()

print(f'The simulation starts.')
fields, energies, nl_phase, max_trajectory  = run(domain, waveguide, input_beam, dz=dz, num_field_sample=num_field_sample,
                                num_mode_sample=num_mode_sample, precision=precision, 
                                disorder=disorder, ds_factor=ds_factor, calculate_nl_phase=True, trace_modes=False,)

fields = fields.cpu().numpy()
nl_phase = nl_phase.cpu().numpy()
max_trajectory = max_trajectory.cpu().numpy()

np.save(f'fields_{input_type}_{beam_radius}_{position}_{int(total_power)}_{precision}_{Nx}_{dz}.npy', fields)
np.save(f'nl_phase_{input_type}_{beam_radius}_{position}_{int(total_power)}.npy', nl_phase)
np.save(f'trajectory_{input_type}_{beam_radius}_{position}_{int(total_power)}_{precision}_{Nx}_{dz}.npy', max_trajectory)