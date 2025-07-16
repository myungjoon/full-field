import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure
from scipy.interpolate import RegularGridInterpolator
import matplotlib as mpl
import matplotlib.font_manager as fm

fm.fontManager.__init__()
mpl.rcParams['font.family'] = ['Arial']
mpl.rcParams['font.size'] = 9

def analyze_your_data(data, percentile=50):
    isovalue = np.percentile(data, percentile)
    
    return isovalue

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


def visualize_isosurface_multiple_lines(X, Y, Z, data, isovalue=0.5, 
                                    filename='isosurface.png', radius=450):
    
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.size': 10
    })
    
    # 여러 isosurface 값 및 색상 설정
    isovalues = [0.1, 0.2, 0.4, 0.8]  # 4개 레벨
    colors = ['blue', 'green', 'orange', 'red']  # 각 레벨에 대한 색상
    alphas = [0.6, 0.5, 0.4, 0.3]  # 각 레벨의 투명도 (높은 값일수록 더 투명)
    
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111, projection='3d')

    # 여러 isosurface 렌더링
    for i, (iso_val, color, alpha) in enumerate(zip(isovalues, colors, alphas)):
        # 각 isosurface 계산
        verts, faces, processed_data = interpolation_only_isosurface(
            X, Y, Z, data, iso_val, upscale_factor=3
        )
        
        # 해당 isosurface 렌더링
        surf = ax.plot_trisurf(
            verts[:, 2], verts[:, 0], faces, verts[:, 1],
            alpha=alpha,
            color=color,
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
    
    # 축 경계선 추가
    # add_axis_boundaries(ax, x_range, y_range, z_range)
    
    # 원통 추가
    add_cylinder(ax, radius, z_range[0], z_range[1], alpha=0.15, color='lightblue')
    
    # ax.set_zlabel(r'$x (\mu m)$', fontsize=12)
    # ax.set_ylabel(r'$y (\mu m)$', fontsize=12) 
    # ax.set_xlabel(r'$z (cm) $', fontsize=12)
    ax.set_box_aspect([5, 1, 1])

    # x, y, z ticks 설정
    ax.set_zticks(np.linspace(x_range[0], x_range[1], 3), )
    ax.set_yticks(np.linspace(y_range[0], y_range[1], 3), )
    ax.set_xticks(np.linspace(z_range[0], z_range[1], 6), )

    # set tick labels
    ax.set_yticklabels([f'{int(tick)}' for tick in np.linspace(450, -450, 3)])

    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='both', which='minor', labelsize=8)

    ax.set_xlim(z_range[0], z_range[1])
    ax.set_ylim(y_range[0], y_range[1])
    ax.set_zlim(x_range[0], x_range[1])

    ax.invert_xaxis()

    ax.margins(0)
    # ax.view_init(elev=30, azim=80, roll=0)
    # ax.view_init(elev=12, azim=37, roll=0)
    # ax.view_init(elev=28, azim=64, roll=0)
    ax.view_init(elev=8, azim=20+90, roll=0)
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10  
    ax.zaxis.labelpad = 20
    # ax.grid(True, alpha=0.3)
    # ax.xaxis.pane.fill = False
    # ax.yaxis.pane.fill = False
    # ax.zaxis.pane.fill = False
    # ax.xaxis.pane.set_alpha(0.0)
    # ax.yaxis.pane.set_alpha(0.0)
    # ax.zaxis.pane.set_alpha(0.0)
    fig.tight_layout(pad=0)
    # plt.tight_layout()
    plt.savefig(f'{filename}.svg', format='svg', bbox_inches='tight', dpi=300)
    plt.show()
    
    return fig, ax


def visualize_isosurface(X, Y, Z, data, isovalue=0.5, 
                                    filename='isosurface.png', radius=450):
    
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.size': 10
    })
    
    # interpolation_only_isosurface
    verts, faces, processed_data = interpolation_only_isosurface(
        X, Y, Z, data, isovalue, upscale_factor=3
    )

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111, projection='3d')

    # Isosurface 렌더링
    surf = ax.plot_trisurf(
        verts[:, 2], verts[:, 0], faces, verts[:, 1],
        alpha=0.5,
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
    
    # 축 경계선 추가
    # add_axis_boundaries(ax, x_range, y_range, z_range)
    
    # 원통 추가
    add_cylinder(ax, radius, z_range[0], z_range[1], alpha=0.15, color='lightblue')
    
    # ax.set_zlabel(r'$x (\mu m)$', fontsize=12)
    # ax.set_ylabel(r'$y (\mu m)$', fontsize=12) 
    # ax.set_xlabel(r'$z (cm) $', fontsize=12)
    ax.set_box_aspect([5, 1, 1])

    # x, y, z ticks 설정
    ax.set_zticks(np.linspace(x_range[0], x_range[1], 3), )
    ax.set_yticks(np.linspace(y_range[0], y_range[1], 3), )
    ax.set_xticks(np.linspace(z_range[0], z_range[1], 6), )

    # set tick labels
    ax.set_yticklabels([f'{int(tick)}' for tick in np.linspace(450, -450, 3)])

    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='both', which='minor', labelsize=8)

    ax.set_xlim(z_range[0], z_range[1])
    ax.set_ylim(y_range[0], y_range[1])
    ax.set_zlim(x_range[0], x_range[1])

    ax.invert_xaxis()

    ax.margins(0)
    # ax.view_init(elev=30, azim=80, roll=0)
    # ax.view_init(elev=12, azim=37, roll=0)
    # ax.view_init(elev=28, azim=64, roll=0)
    ax.view_init(elev=8, azim=20+90, roll=0)
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10  
    ax.zaxis.labelpad = 20
    # ax.grid(True, alpha=0.3)
    # ax.xaxis.pane.fill = False
    # ax.yaxis.pane.fill = False
    # ax.zaxis.pane.fill = False
    # ax.xaxis.pane.set_alpha(0.0)
    # ax.yaxis.pane.set_alpha(0.0)
    # ax.zaxis.pane.set_alpha(0.0)
    fig.tight_layout(pad=0)
    # plt.tight_layout()
    plt.savefig(f'{filename}.svg', format='svg', bbox_inches='tight', dpi=300)
    plt.show()
    
    return fig, ax


# 메인 실행 부분
liniearity = 'nonlinear'
if liniearity == 'linear':
    filename = 'normalized_fields_custom_2.5e-05_off_1_double_2048_1e-05_0.15.npy'
elif liniearity == 'nonlinear':
    filename = 'normalized_fields_custom_2.5e-05_off_160000_double_2048_1e-05_0.15.npy'
fields = np.load(filename)

fields = fields.astype(np.complex64)
ds = 4
fields = fields[:, ::ds, ::ds]  # Downsample by a factor of 4
# only look at the middle as roi
fields = fields[:, fields.shape[1]//6 : (fields.shape[2] // 6)*5, fields.shape[2]//6 : (fields.shape[2] // 6)*5]
fields = fields[::2,:,:]

intensities = np.abs(fields)**2
percentile = 95

intensities = np.transpose(intensities, (1, 2, 0))

# isovalue = analyze_your_data(intensities, percentile)
isovalue = 0.2
print(f'isovalue: {isovalue}')


Nx = intensities.shape[0]
Ny = intensities.shape[1]
Nz = intensities.shape[2]

radius = 450
propation_length = 15  # in centimeter

x = np.linspace(-1*radius, 1*radius, Nx)
y = np.linspace(-1*radius, 1*radius, Ny)
z = np.linspace(0, propation_length, Nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

visualize_isosurface_multiple_lines(X, Y, Z, intensities, isovalue=isovalue, 
                     filename=f'{liniearity}_{isovalue}.png', radius=radius)