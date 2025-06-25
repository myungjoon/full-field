import numpy as np
import matplotlib.pyplot as plt
import torch

from src.util import plot_index_profile, plot_beam_intensity, plot_beam_intensity_and_phase, make_3d_animation, print_total_power, plot_3d_profile
from src.simulation import Domain, Fiber, KerrMedium, Input, run
from src.modes import calculate_modes, decompose_modes, n_to_lm, calculate_effective_mode_area

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

wvl0 = 775e-9
NA = 0.25
n0 = 1.45
nc = np.sqrt(NA**2 + n0**2)
n2 = 3.2e-20

fiber_radius = 400e-6

Lx = 1200e-6
Ly = 1200e-6
Nx = 4096
Ny = 4096
dz = 1e-5
total_z = 0.5

P_middle = 32e3
P_high = 160e3

coefficients1 = np.array([9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
coefficients2 = np.array([7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0])
coefficients3 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

coefficient1 = np.sqrt(coefficients1)
coefficient2 = np.sqrt(coefficients2)
coefficient3 = np.sqrt(coefficients3)

precision = 'double'
domain_5 = Domain(Lx, Ly, Nx, Ny, total_z, precision=precision, device=device)

fiber_radius_4 = 200e-6
medium_5 = Fiber(domain=domain_5, nc=nc, n0=n0, n2=n2, radius=fiber_radius_4, precision=precision, device=device)

input_beam_middle4 = Input(
    domain_5, wvl0, nc, n0, input_type='mode', num_modes=20, fiber_radius=fiber_radius_4, coefficients=coefficient1, power=P_middle, precision=precision, device=device)
input_beam_high4 = Input(
    domain_5, wvl0, nc, n0, input_type='mode', num_modes=20, fiber_radius=fiber_radius_4, coefficients=coefficient1, power=P_high, precision=precision, device=device)


fields_middle4, _, modes_middle4 = run(domain_5, medium_5, input_beam_middle4, dz=dz, precision=precision, trace_modes=True)
fields_high4, _, modes_high4 = run(domain_5, medium_5, input_beam_high4, dz=dz, precision=precision, trace_modes=True)
