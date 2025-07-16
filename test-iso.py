import numpy as np
import matplotlib.pyplot as plt

total_power = 300000
fields = np.load(f'fields_custom_2.5e-05_off_{total_power}_double_2048_1e-05_0.15.npy')

print(f"fields.shape: {fields.shape}")

for i in range(fields.shape[0]):
    # normalization
    fields[i] = fields[i] / np.max(np.abs(fields[i]))  # Normalize each field

np.save(f'normalized_fields_custom_2.5e-05_off_{total_power}_double_2048_1e-05_0.15.npy', fields)

intensity = np.abs(fields) ** 2

for i in range(fields.shape[0]):
    print(f'max intensity at {i}: {np.max(intensity[i])}')



