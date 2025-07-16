import numpy as np

filename = f'fields_custom_2.5e-05_on_500000_double_2048_1e-05_0.15.npy'
fields = np.load(filename)

test = np.any(np.isnan(fields))
print(f"Contains NaN values: {test}")


intensity = np.abs(fields) ** 2
n2 = 6.4e-20  # m^2/W
k0 = 2 * np.pi / 775e-9  # wvl0 = 550 nm
print(f"maximum intensity value: {np.max(intensity)}")
print(f"maximum Knl value: {np.max(n2 * intensity * k0)}")
print(f"maximum nonlinear phase value: {np.max(n2 * intensity * k0) * 1e-5}")
