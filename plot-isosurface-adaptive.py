import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure
from scipy.interpolate import RegularGridInterpolator


def analyze_your_data(data, percentile=50):
    isovalue = np.percentile(data, percentile)
    
    return isovalue

def analyze_data_adaptive(data, percentile=95):
    """각 x 슬라이스별로 적응적 threshold 적용"""
    Nx, Ny, Nz = data.shape
    adaptive_data = np.zeros_like(data)
    
    for i in range(Nx):
        slice_data = data[i, :, :]  # x=i에서의 yz 평면
        if slice_data.max() > 0:  # 데이터가 있는 경우만
            threshold = np.percentile(slice_data, percentile)
            # threshold 이상인 값만 유지, 나머지는 0
            adaptive_data[i, :, :] = np.where(slice_data >= threshold, slice_data, 0)
    
    return adaptive_data

def interpolation_only_isosurface(X, Y, Z, data, isovalue=0.5, upscale_factor=2):
    x_orig = X[:, 0, 0] if X.ndim == 3 else np.arange(data.shape[0])
    y_orig = Y[0, :, 0] if Y.ndim == 3 else np.arange(data.shape[1])
    z_orig = Z[0, 0, :] if Z.ndim == 3 else np.arange(data.shape[2])
    
    x_high = np.linspace(x_orig.min(), x_orig.max(), 
                        len(x_orig) * upscale_factor)
    y_high = np.linspace(y_orig.min(), y_orig.max(), 
                        len(y_orig) * upscale_factor)
    z_high = np.linspace(z_orig.min(), z_orig.max(), 
                        len(z_orig) * upscale_factor)
    
    interpolator = RegularGridInterpolator(
        (x_orig, y_orig, z_orig), data, 
        method='linear',  
        bounds_error=False, 
        fill_value=data.min()
    )
    
    X_high, Y_high, Z_high = np.meshgrid(x_high, y_high, z_high, indexing='ij')
    points = np.column_stack([X_high.ravel(), Y_high.ravel(), Z_high.ravel()])
    data_high = interpolator(points).reshape(X_high.shape)
    
    verts, faces, _, _ = measure.marching_cubes(data_high, level=isovalue)
    
    verts[:, 0] = x_high[0] + verts[:, 0] * (x_high[1] - x_high[0])
    verts[:, 1] = y_high[0] + verts[:, 1] * (y_high[1] - y_high[0])
    verts[:, 2] = z_high[0] + verts[:, 2] * (z_high[1] - z_high[0])
    
    return verts, faces, data_high


def add_cylinder(ax, radius, z_min, z_max, alpha=0.2, color='gray'):
    """원통 추가"""
    # 원통의 매개변수
    theta = np.linspace(0, 2*np.pi, 30)
    z_cyl = np.linspace(z_min, z_max, 20)
    
    # 원통 표면 생성
    THETA, Z_CYL = np.meshgrid(theta, z_cyl)
    X_CYL = radius * np.cos(THETA)
    Y_CYL = radius * np.sin(THETA)
    
    # 원통 그리기
    ax.plot_surface(Z_CYL, Y_CYL, X_CYL, 
                   alpha=alpha, color=color, 
                   linewidth=0, antialiased=False)


def add_axis_boundaries(ax, x_range, y_range, z_range, linewidth=2, color='black'):
    """XY 축 경계선 추가"""
    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range
    
    # XY 평면의 경계선들 (z = z_min에서)
    # X축 경계선
    ax.plot([z_min, z_max], [y_min, y_min], [x_min, x_min], 
            color=color, linewidth=linewidth)
    ax.plot([z_min, z_max], [y_max, y_max], [x_min, x_min], 
            color=color, linewidth=linewidth)
    ax.plot([z_min, z_max], [y_min, y_min], [x_max, x_max], 
            color=color, linewidth=linewidth)
    ax.plot([z_min, z_max], [y_max, y_max], [x_max, x_max], 
            color=color, linewidth=linewidth)
    
    # Y축 경계선
    ax.plot([z_min, z_min], [y_min, y_max], [x_min, x_min], 
            color=color, linewidth=linewidth)
    ax.plot([z_max, z_max], [y_min, y_max], [x_min, x_min], 
            color=color, linewidth=linewidth)
    ax.plot([z_min, z_min], [y_min, y_max], [x_max, x_max], 
            color=color, linewidth=linewidth)
    ax.plot([z_max, z_max], [y_min, y_max], [x_max, x_max], 
            color=color, linewidth=linewidth)


def visualize_isosurface(X, Y, Z, data, isovalue=0.5, 
                        filename='isosurface.png', radius=450, use_adaptive=False, adaptive_percentile=75):
    
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.size': 10
    })
    
    # 적응적 threshold 사용 여부 결정
    if use_adaptive:
        processed_intensities = analyze_data_adaptive(data, adaptive_percentile)
        # 적응적 데이터에서는 0보다 큰 모든 값을 표시
        isovalue = processed_intensities[processed_intensities > 0].min() * 1.01
        print(f"Using adaptive threshold (percentile={adaptive_percentile}) with isovalue={isovalue:.2e}")
    else:
        processed_intensities = data
        print(f"Using global threshold with isovalue={isovalue:.2e}")
    
    # interpolation_only_isosurface
    verts, faces, _ = interpolation_only_isosurface(
        X, Y, Z, processed_intensities, isovalue, upscale_factor=3
    )

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111, projection='3d')

    # Isosurface 렌더링
    surf = ax.plot_trisurf(
        verts[:, 2], verts[:, 1], faces, verts[:, 0],
        alpha=0.8,
        color='crimson',
        linewidth=0,
        antialiased=True,
        shade=True,
        edgecolor='none',
        rasterized=True
    )
    
    # 축 범위 계산
    x_range = (X.min(), X.max())
    y_range = (Y.min(), Y.max()) 
    z_range = (Z.min(), Z.max())
    
    # 원통 추가
    add_cylinder(ax, radius, z_range[0], z_range[1], alpha=0.15, color='lightblue')
    
    ax.set_zlabel(r'$x (\mu m)$', fontsize=12)
    ax.set_ylabel(r'$y (\mu m)$', fontsize=12) 
    ax.set_xlabel(r'$z (cm) $', fontsize=12)
    ax.set_box_aspect([4, 1, 1])

    # x, y, z ticks 설정
    ax.set_zticks(np.linspace(x_range[0], x_range[1], 3), )
    ax.set_yticks(np.linspace(y_range[0], y_range[1], 3), )
    ax.set_xticks(np.linspace(z_range[0], z_range[1], 6), )

    # set tick labels
    ax.set_yticklabels([f'{int(tick)}' for tick in np.linspace(450, -450, 3)])

    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=10)

    ax.set_xlim(z_range[0], z_range[1])
    ax.set_ylim(y_range[0], y_range[1])
    ax.set_zlim(x_range[0], x_range[1])

    ax.invert_xaxis()
    ax.view_init(elev=12, azim=37, roll=0)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig, ax


# 메인 실행 부분
linearity = 'nonlinear'  # 오타 수정: liniearity -> linearity
if linearity == 'linear':
    filename = 'fields_custom_2.5e-05_off_1_double_2048_1e-05_0.15.npy'
elif linearity == 'nonlinear':
    filename = 'fields_custom_2.5e-05_off_160000_double_2048_1e-05_0.15.npy'
fields = np.load(filename)

fields = fields.astype(np.complex64)
ds = 4
fields = fields[:, ::ds, ::ds]  # Downsample by a factor of 4
# only look at the middle as roi
fields = fields[:, fields.shape[1]//6 : (fields.shape[2] // 6)*5, fields.shape[2]//6 : (fields.shape[2] // 6)*5]
fields = fields[::5,:,:]

intensities = np.abs(fields)**2
percentile = 99.5

intensities = np.transpose(intensities, (1, 2, 0))

# 기존 방법
isovalue = analyze_your_data(intensities, percentile)

Nx = intensities.shape[0]
Ny = intensities.shape[1]
Nz = intensities.shape[2]

radius = 450
propation_length = 15  # in centimeter

x = np.linspace(-1*radius, 1*radius, Nx)
y = np.linspace(-1*radius, 1*radius, Ny)
z = np.linspace(0, propation_length, Nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')


# 새로운 방법 (적응적 threshold)
print("\n=== Adaptive Threshold Method ===")
visualize_isosurface(X, Y, Z, intensities, isovalue=None, 
                     filename=f'{linearity}_adaptive_75.png', 
                     radius=radius, use_adaptive=True, adaptive_percentile=75)