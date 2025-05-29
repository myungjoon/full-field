import numpy as np
import matplotlib.pyplot as plt
import torch
import os, time

from src.util import plot_index_profile, plot_beam_intensity, plot_energy_evolution, make_3d_animation
from src.simulation import Domain, Fiber, Input, run
from src.modes import calculate_modes, decompose_modes, n_to_lm

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

seed = 1
np.random.seed(seed)





c = 299792458

NA = 0.25
n_clad = 1.45
n_core = np.sqrt(NA**2 + n_clad**2)
n2 = 2.3e-20
wvl0 = 775e-9

fiber_radius = 50e-6
Lx, Ly = 4*fiber_radius, 4*fiber_radius
unit = 1e-6
Nx, Ny = 512, 512
print(f'The grid size is {Nx}x{Ny}')

ds_factor = 4 if Nx >= 2048 else 1

total_z = 0.30
dz = 1e-5

total_power = 100e3

mode_decompose = False
# Check the fundamental mode
input_type = "gaussian"
position = "on"
waveguide_type = "fiber"

domain = Domain(Lx, Ly, Nx, Ny, device=device)
fiber = Fiber(domain, n_core, n_clad, total_z, dz, n2=n2, radius=fiber_radius, disorder=False, device=device)
fiber_indices = fiber.n.cpu().numpy()

extent = [-Lx/2/unit, Lx/2/unit, -Ly/2/unit, Ly/2/unit]
fundamental_mode = Input(domain, wvl0, n_core, n_clad, type="mode", power=total_power, radius=fiber_radius, l=0, m=1, device=device)
fundamental_mode_field = fundamental_mode.field.cpu().numpy()
plot_beam_intensity(fundamental_mode_field, indices=fiber_indices, extent=extent, interpolation="bilinear")
plt.savefig(f'./results/fundamental_mode_{fiber_radius}.png',)

fiber_index = fiber.n.cpu().numpy()
num_sample = 200
powers = [1e4, 1e5, 5e5, 1e6]
beam_radius = 25e-6
for total_power in powers:
    print(f'The total power is {total_power} W')

    if position == "off":
        cy = fiber_radius/4 
        cx = 0
    else:
        cx = 0
        cy = 0


    input_beam = Input(domain, wvl0, n_core, n_clad, type="gaussian", power=total_power, noise=True,
                        radius=fiber_radius, cx=cx, cy=cy,beam_radius=beam_radius, device=device)
    input_field = input_beam.field.cpu().numpy()
              
    # elif input_type == "gaussian":
    #     noise = True
    #     beam_radius = 50e-6
    #     input_beam = Input(domain, wvl0, n_core, n_clad, 
    #                        type="gaussian", cx=cx, cy=cy, 
    #                        power=total_power, noise=noise, beam_radius=beam_radius, radius=fiber_radius, device=device)
        

    print(f'The simulation starts.')
    if mode_decompose:
        modes, fields, energies = run(domain, input_beam, fiber, wvl0, dz=dz, n_sample=num_sample, mode_decompose=mode_decompose,)
    else:
        fields, energies  = run(domain, input_beam, fiber, wvl0, dz=dz, n_sample=num_sample, mode_decompose=mode_decompose,)
    fields = fields[:, ::ds_factor, ::ds_factor]
    fields = fields.cpu().numpy()
    fiber_index_ds = fiber.n[::ds_factor, ::ds_factor]
    np.save(f'fiber_{fiber_radius}_indices.npy', fiber_index)
    np.save(f'input_{fiber_radius}_{position}_{int(total_power)}.npy', input_field)
    np.save(f'fields_{fiber_radius}_{position}_{int(total_power)}.npy', fields)
    np.save(f'energies_{fiber_radius}_{position}_{int(total_power)}.npy', energies)

    if mode_decompose:
        np.save(f'modes_{fiber_radius}_{position}_{int(total_power)}.npy', modes)

    # plot_beam_intensity(input_field, indices=fiber_index, interpolation="bilinear")
    animation_filename = f'./results/{input_type}_{fiber_radius}_{position}_{int(total_power)}'
    make_3d_animation(fields, indices=fiber_index_ds, radius=fiber_radius/unit, filename=animation_filename, extent=extent, interpolation="bilinear")

# plt.show()
