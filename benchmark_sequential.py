import numpy as np
import matplotlib.pyplot as plt
import torch
import os, time, argparse

from src.util import plot_index_profile, plot_beam_intensity, plot_beam_intensity_and_phase, make_3d_animation, print_total_power
from src.simulation import Domain, Fiber, Input, run
from src.modes import calculate_modes, decompose_modes, n_to_lm

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

seed = 1
np.random.seed(seed)

# arguments for total_power, beam_radius, position
parser = argparse.ArgumentParser(description='Simulation parameters')
parser.add_argument('--input_type', type=str, default='gaussian', choices=['gaussian', 'mode'], help='Input type')
parser.add_argument('--num_modes', type=int, default=5, help='Number of input modes')
parser.add_argument('--total_power', type=float, default=500e3, help='Total power (W)')
parser.add_argument('--beam_radius', type=float, default=20e-6, help='Beam radius (m)')
parser.add_argument('--position', type=str, default='on', choices=['on', 'off'], help='Beam position')
parser.add_argument('--disorder', type=bool, default=False, help='Disorder in the fiber')
parser.add_argument('--precision', type=str, default='single', choices=['single', 'double'], help='Precision of the simulation')
parser.add_argument('--num_pixels', type=int, default=8, help='Number of pixels for the phase map')
parser.add_argument('--num_levels', type=int, default=8, help='Number of levels for each phase map')
parser.add_argument('--device_id', type=int, default=0, help='Device ID for CUDA')

args = parser.parse_args()
print(f'Input type: {args.input_type}')
if args.input_type == 'mode':
    print(f'Number of modes: {args.num_modes}')
elif args.input_type == 'gaussian':
    print(f'Beam radius: {args.beam_radius * 1e6} um')
    
else:
    raise ValueError('Invalid input type. Choose either "gaussian" or "mode".')
print(f'Total power: {args.total_power * 1e-3} kW')
print(f'Beam position: {args.position}')
print(f'Disorder: {args.disorder}')
print(f'Precision: {args.precision}')
print(f'Number of pixels: {args.num_pixels}x{args.num_pixels}')

input_type = args.input_type
total_power = args.total_power
position = args.position
disorder = args.disorder
precision = args.precision
num_pixels = args.num_pixels
beam_radius = args.beam_radius
num_modes = args.num_modes
num_levels = args.num_levels
device_id = args.device_id

device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')


def compute_mode_overlap_objective(output, target):
    target_norm = target / torch.sqrt(torch.sum(torch.abs(target)**2))
    output_norm = output / torch.sqrt(torch.sum(torch.abs(output)**2))
    
    # 중첩 계산 (내적의 절대값 제곱)
    overlap = torch.abs(torch.sum(output_norm.conj() * target_norm))**2
    
    # 최대화를 위해 음수 반환
    return overlap

c = 299792458
wvl0 = 1064e-9

# Fiber parameters
NA = 0.1950
n_clad = 1.457
n_core = np.sqrt(NA**2 + n_clad**2)
n2 = 2.3e-20
fiber_radius = 25e-6
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
# # fundamental_mode = Input(domain, wvl0, n_core, n_clad, type="mode", power=total_power, radius=fiber_radius, l=0, m=1, device=device)
# # fundamental_mode_field = fundamental_mode.field.cpu().numpy()
# # plot_beam_intensity(fundamental_mode_field, indices=fiber_indices, extent=extent, interpolation="bilinear")
# plt.savefig(f'./results/fundamental_mode_{fiber_radius}.png',)


fiber_index = fiber.n.cpu().numpy()
num_sample = 100

if position == "off":
    cy = fiber_radius/4 
    cx = 0
else:
    cx = 0
    cy = 0

phase_modulation = True
pixels = (num_pixels, num_pixels)
input_beam = Input(domain, wvl0, n_core, n_clad, 
                    type=input_type, cx=cx, cy=cy, pixels=pixels, 
                    power=total_power, phase_modulation=phase_modulation, precision=precision,
                    beam_radius=beam_radius, num_modes=num_modes, fiber_radius=fiber_radius, device=device)
input_field = input_beam.field.cpu().numpy()

target_beam = Input(domain, wvl0, n_core, n_clad, fiber_radius=fiber_radius, precision=precision,
                    type="single mode", l=1, m=2, device=device)

target = target_beam.field.cpu().numpy()
plot_beam_intensity_and_phase(target, indices=fiber_indices, extent=extent, interpolation="bilinear")
plt.show()

print(f'Sequential optimization starts.')




fobj_best = 0
field = input_beam.field
phases = np.zeros((num_pixels, num_pixels), dtype=np.float32)

fobj_list = []

for i in range(num_pixels):
    for j in range(num_pixels):
        for k in range(num_levels):
            phases[i, j] += 2 * np.pi * (k / num_levels)
            input_beam.update_beam_phase(field, pixels, phases)
            output_fields, _  = run(domain, input_beam, fiber, wvl0, dz=dz, n_sample=num_sample, precision=precision, disorder=disorder, ds_factor=ds_factor)     
            output = output_fields[-1].cpu().numpy()
            fobj_current = compute_mode_overlap_objective(output, target)
            if fobj_current > fobj_best:
                best_fobj = fobj_current
                best_input = output_fields[0].cpu().numpy()
                best_output = output_fields[-1].cpu().numpy()
                     

input_field = output_fields[0].cpu().numpy()
fields = output_fields.cpu().numpy()


np.save(f'optimized_fiber_{fiber_radius}_indices.npy', fiber_index)
np.save(f'optimized_input_{fiber_radius}_{beam_radius}_{position}_{int(total_power)}.npy', input_field)
np.save(f'optimized_fields_{fiber_radius}_{beam_radius}_{position}_{int(total_power)}.npy', fields)