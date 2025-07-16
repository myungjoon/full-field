import numpy as np
import matplotlib.pyplot as plt
import torch
import os, time, argparse

from src.util import plot_index_profile, plot_beam_intensity, plot_beam_intensity_and_phase, print_total_power
from src.simulation_st import Domain, Waveguide, Input, run
from src.modes import calculate_modes, decompose_modes, n_to_lm

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

parser = argparse.ArgumentParser(description='Simulation parameters')

parser.add_argument('--precision', type=str, default='double', choices=['single', 'double'], help='Precision of the simulation')
parser.add_argument('--input_type', type=str, default='mode', choices=['gaussian', 'mode'], help='Input type')
parser.add_argument('--num_modes', type=int, default=1, help='Number of input modes')
parser.add_argument('--total_energy', type=float, default=3, help='Total power (W)')
parser.add_argument('--beam_radius', type=float, default=25e-6, help='Beam radius (m)')
parser.add_argument('--position', type=str, default='on', choices=['on', 'off'], help='Beam position')
parser.add_argument('--scale', type=float, default=1.0, help='Scale factor for the input beam')
parser.add_argument('--seed', type=int, default=61, help='Random seed for reproducibility')
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
print(f'Total power: {args.total_energy} nJ')
print(f'Beam position: {args.position}')

seed = args.seed
input_type = args.input_type
total_energy = args.total_energy
position = args.position
precision = args.precision
beam_radius = args.beam_radius
num_modes = args.num_modes
scale = args.scale
device_id = args.device_id

np.random.seed(seed)
device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')

c = 299792458
wvl0 = 1030e-9

# Fiber parameters
NA = 0.13
n_clad = 1.45
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

# Time domain
Nt = 2**9
Lt = 10 
dt = Lt / Nt  # time step in seconds
t = np.linspace(-Lt/2, Lt/2, Nt)
omega = 2 * np.pi * np.fft.fftfreq(Nt, dt)
tfwhm = 1  # ps
# beta2 = 16.55e-27  # s^2/m, dispersion parameter
beta2 = -0.0123

t0 = tfwhm / (2 * np.sqrt(np.log(2)))
pulse = np.sqrt(total_energy / (t0*np.sqrt(np.pi)) * 1000) * np.exp(-t**2 / (2 * t0**2))  # Gaussian pulse in nJ
plt.plot(t * 1e12, pulse)
plt.xlabel('Time (ps)')
plt.show()


mode_decompose = False
domain = Domain(Lx, Ly, Lt, Nx, Ny, Nt, propagation_length, device=device)
waveguide = Waveguide(domain, n_core, n_clad, n2=n2, beta2=beta2, radius=waveguide_radius, device=device)
waveguide_indices = waveguide.n.cpu().numpy()


extent = [-Lx/2/unit, Lx/2/unit, -Ly/2/unit, Ly/2/unit]

num_field_sample = 500
num_mode_sample = 500

if position == 'off':
    cy = waveguide_radius / 4
    cx = 0
else:
    cx = 0
    cy = 0

print(f'filename : fields_{input_type}_{beam_radius}_{position}_{int(total_energy)}_{precision}_{Nx}_{dz}.npy')


pulse = torch.tensor(pulse, dtype=torch.float64, device=device)
input_beam = Input(domain, wvl0, n_core, n_clad, pulse=pulse, 
                   input_type=input_type, energy=total_energy, 
                    cx=cx, cy=cy, num_modes=num_modes, waveguide_radius=waveguide_radius, device=device)

input_field = input_beam.field.cpu().numpy()
plot_beam_intensity_and_phase(input_field[Nt//2], indices=waveguide_indices[Nt//2], extent=extent, interpolation=None)
plt.show()

def get_spectral_intensity(field, dt):
    """Calculate the spectral intensity of the field."""
    field_ft = np.fft.fftshift(np.fft.fft(field, axis=0))
    freq = np.fft.fftfreq(field.shape[0], dt)
    spectral_intensity = np.abs(field_ft)**2
    return freq, spectral_intensity


print(f'The simulation starts.')
fields, times, energies = run(domain, waveguide, input_beam, dz=dz, num_field_sample=num_field_sample,
                                precision=precision, ds_factor=ds_factor, )

fields = fields.cpu().numpy()
times = times.cpu().numpy()
energies = energies.cpu().numpy()


plt.figure()
plot_beam_intensity(fields[Nt//2], indices=waveguide_indices[Nt//2], extent=extent, interpolation=None)

plt.figure()
plt.plot(np.arange(501), energies, label='Energy')
plt.xlabel('Propagation step')
plt.ylabel('Energy (nJ)')


plt.figure()
plt.plot(t, times[0], label='t=0')
plt.plot(t, times[Nt//2], label='t=middle')
plt.plot(t, times[-10], label='t=last10')
plt.plot(t, times[-1], label='t=last')
plt.legend()
plt.xlabel('Time (s)')
plt.show()



np.save(f'st_fields_{input_type}_{beam_radius}_{position}_{int(total_energy)}_{precision}_{Nx}_{dz}.npy', fields)
np.save(f'st_times_{input_type}_{beam_radius}_{position}_{int(total_energy)}_{precision}_{Nx}_{dz}.npy', times)