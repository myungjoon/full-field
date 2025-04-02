import numpy as np
import matplotlib.pyplot as plt
import torch

from src.util import plot_index_profile, plot_beam_intensity, plot_beam_phase, plot_mode_evolution
from src.simulation import Domain, Fiber, Input, run
from src.modes import calculate_modes, decompose_modes, n_to_lm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
np.random.seed(40)

c = 299792458
n_core = 1.47
n_clad = 1.457

print(f'NA = {np.sqrt(n_core**2 - n_clad**2)}')

wvl0 = 1064e-9 # Wavelength (meter)
radius = 26e-6
Lx, Ly = 150e-6, 150e-6
Nx, Ny = 256, 256
total_z = 0.01
dz = 1e-6
n2 = 2.3e-20

intensity = 1.0
amp = np.sqrt(intensity)

# test mode
l=0
m=1

domain = Domain(Lx, Ly, Nx, Ny, device=device)
fiber = Fiber(domain, n_core, n_clad, total_z, dz, n2=n2, radius=radius, device=device)
input_beam = Input(domain, wvl0, n_core, n_clad, type="mode", amp=1.0, l=l, m=m, radius=radius, device=device)


output, fields, energies = run(domain, input_beam, fiber, wvl0, dz=dz, mode_decompose=False, nonlinear=False)

input_beam_field = input_beam.field.cpu().numpy()

for i in range(100):
    output_analytic = input_beam_field * np.exp(1j * input_beam.beta * dz * i * 100)
    midpoint = int(Nx / 2)
    angle_sim = np.angle(fields[i, midpoint-3, midpoint])
    angle_anal = np.angle(output_analytic[midpoint-3, midpoint])

    print(f"Angle difference at z={i*100*dz:3e}: {np.abs(angle_sim - angle_anal)% 2*np.pi:.3f} rad")

indices = fiber.n.cpu().numpy()
output_beam_field = output.cpu().numpy()
output_analytic = input_beam_field * np.exp(1j * input_beam.beta * total_z)
# plot_beam_intensity(input_beam_field, indices)
# plot_beam_phase(input_beam_field, indices)
plot_beam_intensity(output_beam_field, indices)
plot_beam_phase(output_beam_field, indices)
plot_beam_intensity(output_analytic, indices)
plot_beam_phase(output_analytic, indices)

plt.show()