import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

# 기존 필드 데이터 로드
fields = np.load('custom_fields.npy')
intensities = np.abs(fields) ** 2

print(f"원본 크기: {intensities.shape}")

# 2048x2048 -> 4096x4096로 interpolation (zoom factor = 2)
intensities_upsampled = zoom(intensities, zoom=2, order=3)  # order=3은 cubic interpolation

print(f"업샘플링 후 크기: {intensities_upsampled.shape}")

# 시각화
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# 원본 2048x2048
ax1.imshow(intensities, cmap='turbo')
ax1.set_title('Original 2048x2048')
ax1.axis('off')

# 업샘플링된 4096x4096
ax2.imshow(intensities_upsampled, cmap='turbo')
ax2.set_title('Interpolated 4096x4096')
ax2.axis('off')

plt.tight_layout()
plt.show()

# 4096x4096만 따로 그리기
plt.figure(figsize=(12, 12))
plt.imshow(intensities_upsampled, cmap='turbo')
plt.title('4096x4096 Interpolated Field Intensity')
plt.axis('off')
plt.show()