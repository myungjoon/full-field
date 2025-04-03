import numpy as np
import matplotlib.pyplot as plt
import torch
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from src.util import plot_index_profile, plot_beam_intensity, plot_mode_evolution
from src.simulation import Domain, Fiber, Input, run
from src.modes import calculate_modes, decompose_modes, n_to_lm

from mmfsim.coupling import GrinFiberCoupler, GrinFiberDegenCoupler


def combine_modes(coefficients, modes, propagation_constants=None, propagation_length=0.):
    shape = (modes[0].shape[0], modes[0].shape[1])
    combined_field = torch.zeros(shape, dtype=torch.complex128)

    # if propagation_constants is not None:
    #     for n in range(len(coefficients)):
    #         combined_field += coefficients[n,0] * modes[n][:, :, 0] * np.exp(-1j * propagation_constants[n] * propagation_length) + coefficients[n,1] * modes[n][:, :, 1] * np.exp(-1j * propagation_constants[n] * propagation_length)
    #         # combined_field += modes[n][:, :, 1] * np.exp(1j * propagation_constants[n] * propagation_length)
    #     return combined_field
    
    for n in range(len(coefficients)):
        combined_field += coefficients[n,0] * modes[n][:, :, 0] + coefficients[n,1] * modes[n][:, :, 1]
        # combined_field += coefficients[n,1] * modes[n][:, :, 1]
    return combined_field


def calculate_effective_indices(modes, fiber_index):
    """
    Calculate the effective indices (n_eff) for each mode.

    Args:
        modes: List of mode fields (numpy arrays).
        fiber_index: Refractive index profile (numpy array).

    Returns:
        List of effective indices (n_eff) for each mode.
    """
    n_eff_list = []
    for mode in modes:
        # Compute the numerator: ∫∫ |mode|^2 * n(x, y) dx dy
        numerator = torch.sum(torch.abs(mode[:, :, 0])**2 * fiber_index)

        # Compute the denominator: ∫∫ |mode|^2 dx dy
        denominator = torch.sum(np.abs(mode[:, :, 0])**2)

        # Effective index for the mode
        n_eff = numerator / denominator
        n_eff_list.append(n_eff)

    return n_eff_list

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device('cpu')
np.random.seed(1)

np.set_printoptions(precision=3)

c = 299792458

n_core = 1.47
n_clad = 1.457
print(f'NA = {np.sqrt(n_core**2 - n_clad**2)}')

wvl0 = 1064e-9 # Wavelength (meter)


radius = 26e-6
Lx, Ly = 150e-6, 150e-6
Nx, Ny = 256, 256
total_z = 1.0
dz = 1e-6

n2 = 2.3e-20
# intensity = 5e13
intensity = 1.0
amp = np.sqrt(intensity)


input_type = "mode_mixing"
beam_radius = 10e-6
cx = 0
cy = 0
noise=True
domain = Domain(Lx, Ly, Nx, Ny, device=device)

num_samples = 500
num_modes = 30

for t in range(5):
    coefficients = np.zeros((num_modes, 2), dtype=complex)
    num_random_modes = 5
    for n in [0, 2, 4, 5, 7, 8, ]:
        l, m = n_to_lm(n+1)
        if l == 0:
            coefficients[n,0] = 1.0 * np.exp(1j * np.random.random() * 1.0 * np.pi)
        else:
            alpha = np.random.random()
            coefficients[n,0] = alpha * np.exp(1j * np.random.random() * 1.0 * np.pi)
            coefficients[n,1] = (1-alpha) * np.exp(1j * np.random.random() * 1.0 * np.pi)


    if input_type == "mode_mixing":
        input_beam = Input(domain, wvl0, n_core, n_clad, type=input_type, amp=amp, radius=radius,
                            num_mode=num_modes, coefficients=coefficients, device=device)
    elif input_type == "gaussian":
        input_beam = Input(domain, wvl0, n_core, n_clad, type="gaussian", cx=cx, cy=cy, noise=noise, beam_radius=beam_radius, amp=amp, radius=radius, device=device)
    else:
        raise ValueError('Invalid Input Type')

    fiber = Fiber(domain, n_core, n_clad, total_z, dz, n2=n2, radius=radius, disorder=False, device=device)
    indices = fiber.n.numpy()
    input_field = input_beam.field.numpy()

    fiber_index = fiber.n.numpy()

    modes = calculate_modes(domain, fiber, input_beam, num_modes=num_modes, device=device)
    mode_coefficients = decompose_modes(input_field, modes, num_modes=num_modes, dtype='complex')

    combined_field = combine_modes(mode_coefficients, modes)
    combined_field = combined_field.numpy()

    plot_beam_intensity(input_field, indices=indices, interpolation="bilinear")
plt.show()

effective_indices = calculate_effective_indices(modes, fiber_index)
effective_indices = np.array(effective_indices)
propagation_constants = 2 * np.pi * effective_indices / wvl0
propagation_length = 1e-2


# input_gt = np.load('./data/fiber_input_field.npy').squeeze()



radial_modes = []
nonradial_modes = []
for n in range(num_modes):
    l, m = n_to_lm(n+1)
    if l == 0:
        radial_modes.append(n)
    else:
        nonradial_modes.append(n)
    mode_energy = np.abs(mode_coefficients[n,0])**2 + np.abs(mode_coefficients[n,1])**2
    print(f'LP{l} {m} : {mode_energy}')
radial_modes = np.array(radial_modes)
nonradial_modes = np.array(nonradial_modes)
plt.figure()
# normalize the mode coefficients
mode_coefficients = mode_coefficients / np.sqrt(np.sum(np.abs(mode_coefficients)**2))
plt.bar(radial_modes, np.abs(mode_coefficients[radial_modes][:,0])**2, color='r', label='l=0')
plt.bar(nonradial_modes, np.abs(mode_coefficients[nonradial_modes][:,0])**2 + np.abs(mode_coefficients[nonradial_modes][:,1])**2, color='b', label='l>0')
plt.ylim(0, 1.01)
plt.xlabel('Mode index')
plt.ylabel('Mode energy')


plt.show()

combined_field_prop = combine_modes(mode_coefficients, modes, propagation_constants=propagation_constants, propagation_length=propagation_length)
combined_field_prop = combined_field_prop.numpy()

# Comparison
output_gt = np.load('./data/fiber_output_field.npy')
output_gt = output_gt.squeeze()

# difference = np.abs(input_gt - combined_field)
# difference_prop = np.abs(output_gt - combined_field_prop)

output_mode_coefficients = decompose_modes(output_gt, modes, num_modes=num_modes, dtype='complex')
output_combined_field = combine_modes(output_mode_coefficients, modes)
# phase = (propagation_constants[0] * propagation_length)# % (2 * np.pi)
# print(f'phase : {phase}')
# print(f'phase_sim : {np.angle(output_mode_coefficients[0,0])}')
# difference_prop = np.abs(output_field - combined_field_prop)

# plt.plot(effective_indices, 'o-')
# plt.xlabel('Mode index')
# plt.ylabel('Effective index')
# plt.grid()
fields = np.load('./data/fiber_fields.npy')
fields = fields.squeeze()

# plot_index_profile(fiber_index)
# plot_beam_intensity(input_field, indices=indices, interpolation="bilinear")
phase = 0
dz = 1e-8

# for i in range(100):
#     phase = -1 * propagation_constants[1] * (i * dz)
#     phase = phase % (2 * np.pi)
#     if phase > np.pi:
#         phase = phase - 2 * np.pi

#     combined_field_prop = combine_modes(mode_coefficients, modes, propagation_constants=propagation_constants, propagation_length=(i*dz))
#     combined_field_prop = combined_field_prop.numpy()

#     output_mode_coefficients = decompose_modes(fields[i], modes, num_modes=num_modes, dtype='complex')

#     print(f'phase : {phase}, output_mode_coefficients : {np.angle(output_mode_coefficients[1])}')

plot_beam_intensity(input_field, indices=indices, interpolation="bilinear")
plot_beam_intensity(combined_field, indices=indices, interpolation="bilinear")
# plot_beam_intensity(output_gt, indices=indices, interpolation="bilinear")
# plot_beam_intensity(output_combined_field, indices=indices, interpolation="bilinear")
# plot_beam_intensity(difference, indices=indices, interpolation="bilinear")
# plot_beam_intensity(difference_prop, indices=indices, interpolation="bilinear")


plt.show()

total_power = np.sum(np.abs(input_field)**2)
print(f'Total power: {total_power:.4e}')
