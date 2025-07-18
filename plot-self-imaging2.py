import numpy as np
import matplotlib.pyplot as plt

# self-imaging
def get_spot_size(fields):
    """
    2D field array로부터 spot size를 계산하는 함수
    
    Parameters:
    -----------
    fields : 2D numpy array
        Electric field 또는 intensity distribution
        
    Returns:
    --------
    float
        Spot size (w) - ISO 11146 standard에 따른 1/e^2 radius
    """

    intensity = np.abs(fields)**2

    
    # Total power 계산
    total_power = np.sum(intensity)
    
    if total_power == 0:
        return 0.0
    
    # Grid coordinates 생성 (pixel 기준)
    ny, nx = intensity.shape
    y_indices, x_indices = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    
    # Center of mass 계산 (weighted mean)
    x_center = np.sum(x_indices * intensity) / total_power
    y_center = np.sum(y_indices * intensity) / total_power
    
    # Second moment 계산
    sigma_x_sq = np.sum(((x_indices - x_center)**2) * intensity) / total_power
    sigma_y_sq = np.sum(((y_indices - y_center)**2) * intensity) / total_power
    
    w_x = 2 * np.sqrt(sigma_x_sq)
    w_y = 2 * np.sqrt(sigma_y_sq)
    
    spot_size = (w_x + w_y) / 2
    
    return spot_size

    

z = np.linspace(0, 0.15, 2001)  # Propagation distance in meters
wvl0 = 775e-9
n0 = 1.45
NA = 0.25
nc = np.sqrt(n0**2 + NA**2)
delta = (nc**2 - n0**2) / (2 * nc**2)
n2 = 2.6e-20*2
a = 450e-6
zp = np.pi * a / np.sqrt(2 * delta)
beta = nc * 2 * np.pi / wvl0
w0 = 100e-6
P1 = 10e3
P2 = 100e3
P3 = 500e3
P_cr = 2 * np.pi * nc / n2 / beta**2

C1 = np.sqrt(1 - (P1/P_cr)) * zp / beta / np.pi / w0**2
C2 = np.sqrt(1 - (P2/P_cr)) * zp / beta / np.pi / w0**2
C3 = np.sqrt(1 - (P3/P_cr)) * zp / beta / np.pi / w0**2

wz1 = w0 * np.sqrt(np.cos(np.pi / zp * z)**2 + C1**2 * np.sin(np.pi / zp * z)**2)
wz2 = w0 * np.sqrt(np.cos(np.pi / zp * z)**2 + C2**2 * np.sin(np.pi / zp * z)**2)
wz3 = w0 * np.sqrt(np.cos(np.pi / zp * z)**2 + C3**2 * np.sin(np.pi / zp * z)**2)

fields1 = np.load('fields_gaussian_0.0001_on_1000.npy')
# fields2 = np.load('fields_gaussian_0.0001_on_10000.npy')
# fields3 = np.load('fields_gaussian_0.0001_on_100000.npy')
fields4 = np.load('fields_gaussian_0.0001_on_100000_3.npy')
fields5 = np.load('fields_gaussian_0.0001_on_500000_3.npy')

wz1_sim = np.zeros(fields1.shape[0])
wz2_sim = np.zeros(fields1.shape[0])
wz3_sim = np.zeros(fields1.shape[0])
wz4_sim = np.zeros(fields1.shape[0])
wz5_sim = np.zeros(fields1.shape[0])


for i in range(fields1.shape[0]):
    wz1_sim[i] = get_spot_size(np.abs(fields1[i])**2) * (a * 4) / 512
    # wz2_sim[i] = get_spot_size(fields2[i]) * (450e-6 * 4) / 512
    # wz3_sim[i] = get_spot_size(fields3[i]) * (450e-6 * 4) / 512
    wz4_sim[i] = get_spot_size(np.abs(fields4[i])**2) * (a * 4) / 512
    wz5_sim[i] = get_spot_size(np.abs(fields5[i])**2) * (a * 4) / 512


# wz_max, wz_min = np.max(wz5_sim), np.min(wz5_sim)
# wz5_sim /= wz_max
# wz4_sim /= wz_max

print(f' Critical power P_cr: {P_cr / 1e3:.2f} kW')
print(f' self-imaging period zp: {zp * 1e3:.2f} mm')

print(f' C1: {C1:.4f}, C2: {C2:.4f}, C3: {C3:.4f}')

z_sim = np.linspace(0, 0.15, fields1.shape[0])

plt.figure(figsize=(8, 6))
plt.xlabel('Propagation distance (cm)')
plt.ylabel('Spot size (um)')
# plt.plot(z * 1e2, wz1 * 1e6, label='Theory, 1 kW')
plt.plot(z * 1e2, wz2 * 1e6, label='Theory, 100 kW')
plt.plot(z * 1e2, wz3 * 1e6, label='Theory, 500 kW')

# plt.plot(z_sim * 1e2, wz1_sim * 1e6, label='Sim, 1 kW')
# plt.plot(z_sim * 1e2, wz2_sim * 1e6, label='Sim, 10 kW')
# plt.plot(z_sim * 1e2, wz3_sim * 1e6, label='Sim, 100 kW')
plt.plot(z_sim * 1e2, wz4_sim * 1e6, label='Sim, 100 kW')
plt.plot(z_sim * 1e2, wz5_sim * 1e6, label='Sim, 500 kW')
# plt.plot(z * 1e2, wz2 * 1e6, label='P2 = 10 kW')
# plt.plot(z * 1e2, wz3 * 1e6, label='P3 = 100 kW')

plt.legend()
plt.show()