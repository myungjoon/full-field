import numpy as np
import matplotlib.pyplot as plt
import torch

from src.util import plot_index_profile, plot_beam_intensity, plot_energy_evolution, make_3d_animation
from src.simulation import Domain, Fiber, Input, run
from src.modes import calculate_modes, decompose_modes, n_to_lm

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

seed = 1
np.random.seed(seed)

c = 299792458

NA = 0.25
n_clad = 1.45
n_core = np.sqrt(NA**2 + n_clad**2)
n2 = 2.3e-20
wvl0 = 775e-9

fiber_radius = 450e-6
Lx, Ly = 4*fiber_radius, 4*fiber_radius
unit = 1e-6
Nx, Ny = 4096, 4096
print(f'The grid size is {Nx}x{Ny}')

ds_factor = 4 if Nx >= 2048 else 1

total_z = 0.15
dz = 1e-5

total_power = 100e4

mode_decompose = False
# Check the fundamental mode
input_type = "gaussian"
position = "on"
waveguide_type = "fiber"

dx, dy = Lx/Nx, Ly/Ny

domain = Domain(Lx, Ly, Nx, Ny, device=device)
fiber = Fiber(domain, n_core, n_clad, total_z, dz, n2=n2, radius=fiber_radius, disorder=False, device=device)
fiber_indices = fiber.n.cpu().numpy()

extent = [-Lx/2/unit, Lx/2/unit, -Ly/2/unit, Ly/2/unit]
fundamental_mode = Input(domain, wvl0, n_core, n_clad, type="mode", power=total_power, radius=fiber_radius, l=0, m=1, device=device)
fundamental_mode_field = fundamental_mode.field.cpu().numpy()

fundamental_mode_area = np.sum(np.abs(fundamental_mode_field)**2 * (dx * dy))**2 / np.sum(np.abs(fundamental_mode_field)**4 * (dx * dy))
print(f"Mode area: {fundamental_mode_area:.2e} m^2")
plot_beam_intensity(fundamental_mode_field, indices=fiber_indices, extent=extent, interpolation="bilinear")
plt.show()