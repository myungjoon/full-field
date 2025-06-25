import numpy as np
import matplotlib.pyplot as plt
import torch
import os, time, argparse

from src.util import plot_index_profile, plot_beam_intensity, plot_beam_intensity_and_phase, make_3d_animation, print_total_power
from src.simulation import Domain, Fiber, Input, run
from src.modes import calculate_modes, decompose_modes, n_to_lm

device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

c = 299792458
wvl0 = 775e-9

# Fiber parameters
NA = 0.25
n_clad = 1.45
n_core = np.sqrt(NA**2 + n_clad**2)
n2 = 2.3e-20
fiber_radius = 450e-6
propagation_length = 0.5

# Simulation domain parameters
Lx, Ly = 4*fiber_radius, 4*fiber_radius
unit = 1e-6
Nx, Ny = 4096, 4096
print(f'The grid size is {Nx}x{Ny}')
ds_factor = 4 if Nx >= 2048 else 1 # downsampling factor for memory efficiency
dz = 1e-5

mode_decompose = False
domain = Domain(Lx, Ly, Nx, Ny, propagation_length, device=device)
fiber = Fiber(domain, n_core, n_clad, n2=n2, radius=fiber_radius, device=device)
fiber_indices = fiber.n.cpu().numpy()

extent = [-Lx/2/unit, Lx/2/unit, -Ly/2/unit, Ly/2/unit]

fiber_index = fiber.n.cpu().numpy()
num_field_sample = 500
num_mode_sample = 2000

input_type = 'mode'  # 'mode' or 'gaussian'

cx, cy = 0, 0
in_phase = False
total_power=160e3
num_modes = 6
# coefficients = torch.ones(num_modes, dtype=complex)
coefficients = torch.tensor([0.3, 0.3, 0.3, 0.3, 0.3, 0.3])

input_beam = Input(domain, wvl0, n_core, n_clad, 
                    input_type=input_type, cx=cx, cy=cy, in_phase=in_phase,
                    power=total_power, num_modes=num_modes, coefficients=coefficients, fiber_radius=fiber_radius, device=device)

input_field = input_beam.field.cpu().numpy()
plot_beam_intensity(input_field, indices=fiber_indices, extent=extent, interpolation="bilinear")
plt.show()
