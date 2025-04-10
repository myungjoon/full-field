import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D

from matplotlib.colors import Normalize

import os
import imageio
plt.rcParams['font.size'] = 13

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

def plot_mode_evolution(modes, dz):
    plt.figure()
    z = np.arange(0, len(modes)*dz, dz)
    for i in range(modes.shape[1]):
        l, m = n_to_lm(i+1)
        plt.plot(z, modes[:, i], label=f'LP{l}{m}')
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
    
    eps = 1e-5
    extent = [-75, 75, -75, 75]
    xtick = np.linspace(0, field.shape[1]+eps, 5)
    ytick = np.linspace(0, field.shape[0]+eps, 5)
    xlabel = np.linspace(extent[0], extent[1], 5)
    ylabel = np.linspace(extent[2], extent[3], 5)

    if interpolation is not None:
        im = ax.imshow(np.abs(field)**2, cmap='turbo', interpolation=interpolation)
    else:
        im = ax.imshow(np.abs(field)**2, cmap='turbo',)
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

def plot_input_and_output_beam(input_field, output_field, indices=None, interpolation=None):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    #plot input_field and output_field
    ax[0].imshow(np.abs(input_field)**2, cmap='turbo', interpolation=interpolation)
    ax[0].set_title('Input Field')
    if indices is not None:
        ax[0].contour(indices, levels=[np.min(indices)], colors='white', linewidths=2)


    ax[1].imshow(np.abs(output_field)**2, cmap='turbo', interpolation=interpolation)
    ax[1].set_title('Output Field')
    if indices is not None:
        ax[1].contour(indices, levels=[np.min(indices)], colors='white', linewidths=2)
    
    plt.tight_layout()

def make_3d_animation(fields, indices=None, filename=None):
    intensities = np.abs(fields)**2

    x_size, y_size = intensities.shape[1], intensities.shape[2]

    if not os.path.exists('frames'):
        os.makedirs('frames')

    num_frames = intensities.shape[0]
    for i in range(num_frames):

        vmin = np.min(intensities[i])
        vmax = np.max(intensities[i])
        norm = Normalize(vmin=vmin, vmax=vmax)

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(intensities[i], cmap='turbo', norm=norm, origin='lower',)
        

        x_tick_positions = [0, x_size/2 , x_size-1]
        x_tick_labels = [-1000, 0, 1000]
        plt.xticks(ticks=x_tick_positions, labels=x_tick_labels)
        plt.xlabel(r'x ($\mu m$)')

        y_tick_positions = [0, y_size/2 , y_size-1]
        y_tick_labels = [-1000, 0, 1000]
        plt.yticks(ticks=y_tick_positions, labels=y_tick_labels)
        plt.ylabel(r'y ($\mu m$)')

        # Add colorbar
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.1)
        # plt.colorbar(im, cax=cax)
        
        # Add frame number
        # ax.set_title(f'Frame {i}')

        # Add fiber structure
        ax.contour(indices, levels=[np.min(indices)], colors='white', linewidths=2)
        
        # Save frame
        plt.savefig(f'frames/frame_{i:03d}.png')
        plt.close()

    # Create MP4 using imageio
    fps = 10  # Frames per second
    with imageio.get_writer(filename + '.mp4', fps=fps) as writer:
        for i in range(num_frames):
            image = imageio.imread(f'frames/frame_{i:03d}.png')
            writer.append_data(image)



def plot_3d_profile(fields):
    intensities = np.abs(fields)**2
    nz, nx, ny = intensities.shape

    for i in range(nz):
        threshold = 0.7 * np.max(intensities[i])    
        x_indices, y_indices = np.where(intensities[i] > threshold)
        z_indices = np.full_like(x_indices, i)
        if i == 0:
            all_x_indices = x_indices
            all_y_indices = y_indices
            all_z_indices = z_indices
        else:
            all_x_indices = np.concatenate((all_x_indices, x_indices))
            all_y_indices = np.concatenate((all_y_indices, y_indices))
            all_z_indices = np.concatenate((all_z_indices, z_indices))

    x_indices = all_x_indices
    y_indices = all_y_indices
    z_indices = all_z_indices

    # Define fiber parameters
    R = min(nx, ny) * 0.15  # Core radius (adjust as needed)
    core_center_x = nx // 2
    core_center_y = ny // 2

    # Create figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot only the points above threshold
    scatter = ax.scatter(x_indices, y_indices, z_indices, c='white', s=3)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set limits
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_zlim(0, nz)

    # Get rid of the grid
    ax.grid(False)
    # Get rid of bounding box
    ax.set_box_aspect([1, 1, 1])
    # Make background black
    ax.set_facecolor('black')

    # Generate and plot cylindrical fiber structure
    theta = np.linspace(0, 2*np.pi, 100)
    z_positions = np.linspace(0, nz-1, 20)  # 20 circles along z-axis

    indices = np.array([0, -1])
    for z in z_positions[indices]:
        x_circle = core_center_x + R * np.cos(theta)
        y_circle = core_center_y + R * np.sin(theta)
        z_circle = np.full_like(theta, z)
        ax.plot(x_circle, y_circle, z_circle, 'b-', linewidth=1.5, alpha=0.7)

    # Add vertical lines to connect circles and complete the cylinder visualization
    for i in range(25, 100, 50):  # Draw 10 vertical lines around the circle
        x_line = core_center_x + R * np.cos(theta[i])
        y_line = core_center_y + R * np.sin(theta[i])
        z_line = z_positions
        ax.plot([x_line]*len(z_line), [y_line]*len(z_line), z_line, 'b-', linewidth=1, alpha=0.9)

    # Fix the camera angle
    ax.view_init(elev=-30, azim=-45)


    # Display the plot
    plt.tight_layout()