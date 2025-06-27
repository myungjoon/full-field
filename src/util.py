import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
from matplotlib.colors import Normalize

import os
import imageio
plt.rcParams['font.size'] = 15

def n_to_lm(n):
    """
    Convert mode index n to (l,m) pair according to GRIN fiber LP mode indexing.
    n starts from 1, with n=1 corresponding to (l,m)=(0,1)
    
    The ordering follows:
    1. Smaller l+m values come first
    2. For equal l+m values, pairs with smaller l come first
    
    Examples:
    n=1 -> (0,1)
    n=2 -> (0,2)
    n=3 -> (1,1)
    n=4 -> (0,3)
    n=5 -> (1,2)
    n=6 -> (2,1)
    n=7 -> (0,4)
    n=8 -> (1,3)
    n=9 -> (2,2)
    n=10 -> (3,1)
    etc.
    """
    if n < 1:
        raise ValueError("Mode index n must be a positive integer")
    
    # Start from group 1 (which has l+m=1)
    group_sum = 1
    count = 0
    
    while True:
        # Number of (l,m) pairs in current group where l+m=group_sum
        group_size = group_sum
        
        # If n is in the current group
        if count + group_size >= n:
            # Calculate position within group (0-indexed)
            pos_in_group = n - count - 1
            
            # For group with sum=group_sum, l goes from 0 to group_sum-1
            l = pos_in_group
            m = group_sum - l
            
            return (l, m)
        
        # Move to next group
        count += group_size
        group_sum += 1

def plot_mode_evolution(modes, dz, num_modes=10):
    plt.figure()
    z = np.arange(0, len(modes)*dz, dz)
    for i in range(num_modes):
        l, m = n_to_lm(i+1)
        plt.plot(z, np.sum(np.abs(modes[:, i])**2, axis=1), label=f'LP{l}{m}')
    plt.xlabel('z (m)')
    plt.ylabel('Amplitude')

    plt.legend()

def plot_index_profile(n):
    fig, ax = plt.subplots()
    im = ax.imshow(n, cmap='Blues')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax)

def plot_beam_intensity(field, indices=None, extent=None, interpolation=None):
    fig, ax = plt.subplots()
    
    # extent = [-150/2, 150/2, -150/2, 150/2]
    xtick = np.linspace(0, field.shape[1], 11)
    ytick = np.linspace(0, field.shape[0], 11)
    xlabel = np.linspace(extent[0], extent[1], 11)
    ylabel = np.linspace(extent[2], extent[3], 11)

    if interpolation is not None:
        im = ax.imshow(np.abs(field)**2, cmap='turbo', interpolation=interpolation)
    else:
        im = ax.imshow(np.abs(field)**2, cmap='turbo',)
    ax.set_xticks(xtick)
    ax.set_yticks(ytick)
    ax.set_xticklabels([f'{x:.0f}' for x in xlabel])
    ax.set_yticklabels([f'{y:.0f}' for y in ylabel])

    ax.set_xlabel(r'x ($\mu m$)')
    ax.set_ylabel(r'y ($\mu m$)')
    # colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax)

    if indices is not None:
        ax.contour(indices, levels=[np.min(indices)], colors='white', linewidths=2)

def plot_beam_intensity_and_phase(field, indices=None, extent=None, interpolation=None):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    eps = 1e-5
    # extent = [-450, 450, -450, 450]
    xtick = np.linspace(0, field.shape[1]+eps, 5)
    ytick = np.linspace(0, field.shape[0]+eps, 5)
    xlabel = np.linspace(extent[0], extent[1], 5)
    ylabel = np.linspace(extent[2], extent[3], 5)

    if interpolation is not None:
        im = ax[0].imshow(np.abs(field)**2, cmap='turbo', interpolation=interpolation)
    else:
        im = ax[0].imshow(np.abs(field)**2, cmap='turbo',)
    ax[0].set_xticks(xtick)
    ax[0].set_yticks(ytick)
    ax[0].set_xticklabels([f'{x:.1f}' for x in xlabel])
    ax[0].set_yticklabels([f'{y:.1f}' for y in ylabel])

    ax[0].set_xlabel(r'x ($\mu m$)')
    ax[0].set_ylabel(r'y ($\mu m$)')
    
    # colorbar
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax)

    if indices is not None:
        ax[0].contour(indices, levels=[np.min(indices)], colors='white', linewidths=2)

    if interpolation is not None:
        im = ax[1].imshow(np.angle(field), cmap='turbo', interpolation=interpolation)
    else:
        im = ax[1].imshow(np.angle(field), cmap='turbo',)
        
    ax[1].set_xticks(xtick)
    ax[1].set_yticks(ytick)
    ax[1].set_xticklabels([f'{x:.1f}' for x in xlabel])
    ax[1].set_yticklabels([f'{y:.1f}' for y in ylabel])

    ax[1].set_xlabel(r'x ($\mu m$)')
    ax[1].set_ylabel(r'y ($\mu m$)')
    
    # colorbar
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax)

    if indices is not None:
        ax[1].contour(indices, levels=[np.min(indices)], colors='white', linewidths=2)

def plot_beam_phase(field, indices=None, extent=None, interpolation=None):
    fig, ax = plt.subplots()
    
    eps = 1e-5
    extent = [-75, 75, -75, 75]
    xtick = np.linspace(0, field.shape[1]+eps, 5)
    ytick = np.linspace(0, field.shape[0]+eps, 5)
    xlabel = np.linspace(extent[0], extent[1], 5)
    ylabel = np.linspace(extent[2], extent[3], 5)

    if interpolation is not None:
        im = ax.imshow(np.angle(field), cmap='turbo', interpolation=interpolation)
    else:
        im = ax.imshow(np.angle(field), cmap='turbo',)
    ax.set_xticks(xtick)
    ax.set_yticks(ytick)
    ax.set_xticklabels([f'{x}' for x in xlabel])
    ax.set_yticklabels([f'{y}' for y in ylabel])

    ax.set_xlabel(r'x ($\mu m$)')
    ax.set_ylabel(r'y ($\mu m$)')
    # colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax)

    if indices is not None:
        ax.contour(indices, levels=[np.min(indices)], colors='white', linewidths=2)

def plot_multiple_beams(fields, indices=None):

    fig, ax = plt.subplots(1, len(fields), figsize=(4*len(fields), 4))
    for i, field in enumerate(fields):
        ax[i].imshow(np.abs(field)**2, cmap='jet')
        if indices is not None:
            ax[i].contour(indices, levels=[np.min(indices)], colors='white', linewidths=2)


def plot_energy_evolution(energies, dz):
    plt.figure()
    plt.plot(np.arange(0, len(energies)*dz, dz), energies)
    plt.xlabel('z (m)')
    plt.ylabel('Energy')
    plt.title('Energy Evolution')

def plot_beam_amplitude(field, indices=None, extent=None, interpolation=None):
    fig, ax = plt.subplots()
    
    eps = 1e-5
    extent = [-75, 75, -75, 75]
    xtick = np.linspace(0, field.shape[1]+eps, 5)
    ytick = np.linspace(0, field.shape[0]+eps, 5)
    xlabel = np.linspace(extent[0], extent[1], 5)
    ylabel = np.linspace(extent[2], extent[3], 5)

    if interpolation is not None:
        im = ax.imshow(field.real, cmap='turbo', interpolation=interpolation)
    else:
        im = ax.imshow(field.real, cmap='turbo',)
    ax.set_xticks(xtick)
    ax.set_yticks(ytick)
    ax.set_xticklabels([f'{x}' for x in xlabel])
    ax.set_yticklabels([f'{y}' for y in ylabel])

    ax.set_xlabel(r'x ($\mu m$)')
    ax.set_ylabel(r'y ($\mu m$)')
    # colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax)

    if indices is not None:
        ax.contour(indices, levels=[np.min(indices)], colors='white', linewidths=2)

def plot_input_and_output_modes(input_modes, output_modes, num_l=5, num_m=5):
    x_size, y_size = num_l, num_m

    xpos, ypos = np.meshgrid(np.arange(x_size), np.arange(y_size), indexing="ij")
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos)
    dx = dy = 0.7
    dz1 = input_modes.flatten()
    dz2 = output_modes.flatten()

    # 색상: 행 index 기반
    colors = ['tomato', 'gold', 'limegreen', 'deepskyblue', 'orchid']
    color_list = [colors[x] for x in xpos]

    # 시점 고정용 각도
    elev = 30
    azim = 45

    fig = plt.figure(figsize=(14, 6))

    # 그래프 1
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.bar3d(xpos, ypos, zpos, dx, dy, dz1, color=color_list, shade=True)
    ax1.set_title("Input", fontsize=15)
    ax1.set_xlabel("m", fontsize=15)
    ax1.set_ylabel("l", fontsize=15)
    ax1.set_zlabel("Energy (%)", fontsize=15)
    ax1.view_init(elev=elev, azim=azim)
    ax1.set_xlim(x_size - 0.5, -0.5)  # ← row 방향 반전

    # 그래프 2
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.bar3d(xpos, ypos, zpos, dx, dy, dz2, color=color_list, shade=True)
    ax2.set_title("Output", fontsize=15)
    ax2.set_xlabel("m", fontsize=15)
    ax2.set_ylabel("l", fontsize=15)
    ax2.set_zlabel("Energy (%)", fontsize=15)
    ax2.view_init(elev=elev, azim=azim)
    ax2.set_xlim(x_size - 0.5, -0.5)  # ← row 방향 반전

    plt.tight_layout()

def plot_input_and_output_beam(input_field, output_field, radius=10, extent=None, indices=None, interpolation=None):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(np.abs(input_field)**2, cmap='turbo', interpolation=interpolation, extent=extent)
    ax[0].set_title('Input Field')

    ax[0].set_xlim([-75, 75])
    ax[0].set_ylim([-75, 75])
    ax[0].set_xticks([-75, 0, 75])
    ax[0].set_yticks([-75, 0, 75])
    ax[0].set_xticklabels([-75, 0, 75])
    ax[0].set_yticklabels([-75, 0, 75])
    ax[0].set_xlabel(r'x ($\mu m$)')
    ax[0].set_ylabel(r'y ($\mu m$)')
    
    fiber0 = Circle((0, 0), radius, fill=False, linestyle='--', edgecolor='white', linewidth=2.0)
    ax[0].add_patch(fiber0)

    ax[1].imshow(np.abs(output_field)**2, cmap='turbo', interpolation=interpolation, extent=extent)
    ax[1].set_title('Output Field')
    fiber1 = Circle((0, 0), radius, fill=False, linestyle='--', edgecolor='white', linewidth=2.0)
    ax[1].add_patch(fiber1)

    ax[1].set_xlim([-75, 75])
    ax[1].set_ylim([-75, 75])
    ax[1].set_xticks([-75, 0, 75])
    ax[1].set_yticks([-75, 0, 75])
    ax[1].set_xticklabels([-75, 0, 75])
    ax[1].set_yticklabels([-75, 0, 75])
    ax[1].set_xlabel(r'x ($\mu m$)')
    ax[1].set_ylabel(r'y ($\mu m$)')

    plt.tight_layout()

def print_total_power(domain, field):
    """
    Print the total power of the field.
    """
    # Calculate the total power
    dx = domain.Lx / domain.Nx
    dy = domain.Ly / domain.Ny
    total_power = np.sum(np.abs(field)**2) * dx * dy
    print(f'Total power: {total_power:.4f} W')

def plot_3d_profile(fields, threshold_ratio=0.99, point_size=3, 
                   alpha=0.9, colormap='turbo', 
                   background_color='black', figsize=(10, 8)):
    
    intensities = np.abs(fields)**2
    nz, nx, ny = intensities.shape
    
    x_indices_list = []
    y_indices_list = []
    z_indices_list = []
    intensity_values = []
    
    for i in range(nz):
        threshold = threshold_ratio * np.max(intensities[i])
        x_idx, y_idx = np.where(intensities[i] > threshold)
        
        if len(x_idx) > 0: 
            z_idx = np.full_like(x_idx, i)
            intensities_slice = intensities[i][x_idx, y_idx]
            
            x_indices_list.append(x_idx)
            y_indices_list.append(y_idx)
            z_indices_list.append(z_idx)
            intensity_values.append(intensities_slice)
    
    if x_indices_list: 
        x_indices = np.concatenate(x_indices_list)
        y_indices = np.concatenate(y_indices_list)
        z_indices = np.concatenate(z_indices_list)
        all_intensities = np.concatenate(intensity_values)
    else:
        print("Warning: No data points above threshold found")
        x_indices = np.array([])
        y_indices = np.array([])
        z_indices = np.array([])
        all_intensities = np.array([])
    
  
    R = min(nx, ny) * 0.45  # 코어 반지름
    core_center_x = nx // 2
    core_center_y = ny // 2
    
    # 그림 생성
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # 배경색 설정
    if background_color == 'black':
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('white')
        ax.yaxis.pane.set_edgecolor('white')
        ax.zaxis.pane.set_edgecolor('white')
        ax.xaxis.pane.set_alpha(0.1)
        ax.yaxis.pane.set_alpha(0.1)
        ax.zaxis.pane.set_alpha(0.1)
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
    

    scatter = ax.scatter(x_indices, y_indices, z_indices, 
                color='white', s=point_size, 
                alpha=0.8)
        
        # 컬러바 추가
        # cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
        # cbar.set_label('Intensity', rotation=270, labelpad=15)
        # if background_color == 'black':
        #     cbar.ax.yaxis.set_tick_params(color='white')
        #     cbar.ax.yaxis.label.set_color('white')
    
    theta = np.linspace(0, 2*np.pi, 100)
    
    z_positions = [0, nz-1]
    for z in z_positions:
        x_circle = core_center_x + R * np.cos(theta)
        y_circle = core_center_y + R * np.sin(theta)
        z_circle = np.full_like(theta, z)
        ax.plot(x_circle, y_circle, z_circle, 'cyan', 
                linewidth=2, alpha=alpha)
    
    n_vertical_lines = 2
    for i in range(0, 100, 100//n_vertical_lines):
        x_line = core_center_x + R * np.cos(theta[i])
        y_line = core_center_y + R * np.sin(theta[i])
        z_line = [0, nz-1]
        ax.plot([x_line, x_line], [y_line, y_line], z_line, 
                'cyan', linewidth=1.5, alpha=alpha)
    

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_zlabel('z', fontsize=12)
    
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_zlim(0, nz)
    
    # only show the center area of the core
    ax.set_xlim(core_center_x - R/16, core_center_x + R/16)
    ax.set_ylim(core_center_y - R/16, core_center_y + R/16)
    ax.set_zlim(0, nz)

    ax.grid(False)
    
    ax.set_box_aspect([nx/max(nx,ny,nz), ny/max(nx,ny,nz), nz/max(nx,ny,nz)])
    
    ax.view_init(elev=20, azim=-60)
    
    if background_color == 'black':
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.zaxis.label.set_color('white')
        ax.title.set_color('white')
    
    plt.tight_layout()
    
    return fig, ax

def make_3d_animation(fields, radius=10, propagation_length=100, filename=None, extent=None, roi=None, interpolation=None):
    intensities = np.abs(fields)**2

    if not os.path.exists('frames'):
        os.makedirs('frames')

    num_frames = intensities.shape[0]
    unit_propagation_length = propagation_length / num_frames
    plt.figure()
    for i in range(num_frames):

        vmin = np.min(intensities[i])
        vmax = np.max(intensities[i])
        norm = Normalize(vmin=vmin, vmax=vmax)

        extent = [-radius/1e-6, radius/1e-6, -radius/1e-6, radius/1e-6]

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(intensities[i], cmap='turbo', norm=norm, origin='lower', extent=extent, interpolation=interpolation)
        # ax.set_xlim([-750, 750])
        # ax.set_ylim([-750, 750])
        plt.xlabel(r'x ($\mu m$)')
        plt.ylabel(r'y ($\mu m$)')

        # Only plot the half of the image
        if roi is not None:
            ax.set_xlim([-roi/1e-6, roi/1e-6])
            ax.set_ylim([-roi/1e-6, roi/1e-6])
        else:
            ax.set_xlim([-radius/1e-6, radius/1e-6])
            ax.set_ylim([-radius/1e-6, radius/1e-6])

        fiber = Circle((0, 0), radius/1e-6, fill=False, linestyle='--', edgecolor='white', linewidth=2.0)
        ax.add_patch(fiber)
        
          # Convert to cm if needed

        # At each frame, place text at the top right corner with the current propagation distance dz*i
        current_z = i * unit_propagation_length  # Adjust this value based on your simulation parameters
        ax.text(0.98, 0.98, f'z = {current_z:.2f} cm', transform=ax.transAxes, ha='right', va='top', fontsize=15, color='white')
        # Save frame
        plt.savefig(f'frames/frame_{i:03d}.png')
        plt.close()

    fps = 10  # Frames per second
    with imageio.get_writer(filename, fps=fps) as writer:
        for i in range(num_frames):
            image = imageio.imread(f'frames/frame_{i:03d}.png')
            writer.append_data(image)




def correlation(simulation, reference, dx=1e-6):
    """
    Calculate the correlation between two fields.
    """

    reference = np.abs(reference)**2
    simulation = np.abs(simulation)**2

    reference = (reference - np.mean(reference)) / np.std(reference)
    simulation = (simulation - np.mean(simulation)) / np.std(simulation)

    corr, p_value = stats.pearsonr(reference.flatten(), simulation.flatten())
    print(f"Pearson Correlation: {corr:.4f}, p-value: {p_value:.4e}")
    return corr, p_value