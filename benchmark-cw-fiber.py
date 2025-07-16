import numpy as np
import matplotlib.pyplot as plt
import torch
import os, time, argparse

from src.util import plot_index_profile, plot_beam_intensity, plot_beam_intensity_and_phase, print_total_power
from src.simulation import Domain, Waveguide, Input, run
from src.modes import calculate_modes, decompose_modes, n_to_lm

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

parser = argparse.ArgumentParser(description='Simulation parameters')

parser.add_argument('--precision', type=str, default='double', choices=['single', 'double'], help='Precision of the simulation')
parser.add_argument('--input_type', type=str, default='mode', choices=['gaussian', 'mode'], help='Input type')
parser.add_argument('--num_modes', type=int, default=1, help='Number of input modes')
parser.add_argument('--total_power', type=float, default=3, help='Total power (W)')
parser.add_argument('--beam_radius', type=float, default=1e-6, help='Beam radius (m)')
parser.add_argument('--scale', type=float, default=1.0, help='Scale factor for the input beam')
parser.add_argument('--device_id', type=int, default=0, help='Device ID for CUDA')

args = parser.parse_args()
print(f'Input type: {args.input_type}')
if args.input_type == 'mode':
    print(f'Number of modes: {args.num_modes}')
elif args.input_type == 'gaussian':
    print(f'Beam radius: {args.beam_radius * 1e6} um')    
else:
    raise ValueError('Invalid input type. Choose either "gaussian" or "mode".')

print(f'Precision: {args.precision}')
print(f'Total power: {args.total_power} nJ')

input_type = args.input_type
total_power = args.total_power
precision = args.precision
beam_radius = args.beam_radius
num_modes = args.num_modes
scale = args.scale
device_id = args.device_id

device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')

c = 299792458
wvl0 = 775e-9

# Fiber parameters
NA = 0.25
n_clad = 1.457
n_core = np.sqrt(NA**2 + n_clad**2)
n2 = 2.3e-20
waveguide_radius = 25e-6
propagation_length = 0.01

# Simulation domain parameters
Lx, Ly = 3 * waveguide_radius, 3 * waveguide_radius
unit = 1e-6
Nx, Ny = 256, 256
print(f'The grid size is {Nx}x{Ny}')
ds_factor = round(Nx / 512) if Nx >= 512 else 1 # downsampling factor for memory efficiency
dz = 1e-5


mode_decompose = False
domain = Domain(Lx, Ly, Nx, Ny, propagation_length, device=device)
waveguide = Waveguide(domain, n_core, n_clad, n2=n2, radius=waveguide_radius, device=device)
waveguide_indices = waveguide.n.cpu().numpy()


extent = [-Lx/2/unit, Lx/2/unit, -Ly/2/unit, Ly/2/unit]

num_field_sample = 500
num_mode_sample = 500


print(f'filename : fields_{input_type}_{beam_radius}_{int(total_power)}_{precision}_{Nx}_{dz}.npy')


input_beam = Input(domain, wvl0, n_core, n_clad, 
                   input_type=input_type, power=total_power, beam_radius=beam_radius,
                    num_modes=num_modes, waveguide_radius=waveguide_radius, device=device)

input_field = input_beam.field.cpu().numpy()
plot_beam_intensity_and_phase(input_field, indices=waveguide_indices, extent=extent, interpolation=None)
plt.show()

print(f'The simulation starts.')
fields, energies, = run(domain, waveguide, input_beam, dz=dz, num_field_sample=num_field_sample,
                                precision=precision, ds_factor=ds_factor, )

fields = fields.cpu().numpy()
energies = energies.cpu().numpy()

final_field = fields[-1]

plt.figure()
plt.plot(energies, label='Energy')
plt.xlabel('Propagation step')
plt.ylabel('Energy (nJ)')
plt.title('Energy vs Propagation Step')
plt.legend()

plt.figure()
plot_beam_intensity(final_field**2, indices=waveguide_indices, extent=extent, interpolation=None)
plt.show()