import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import imageio
from matplotlib.colors import Normalize

plt.rcParams['font.size'] = 14

# change to current file directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

data = np.load('fiber_fields_linear_noise.npy')
# data = np.load('fiber_fields_nonlinear_gaussian_high.npy')
# data = np.load('self-imaging_fields_off.npy')
data = np.abs(data)**2

output_intensity = data[-1]
max_val = np.max(output_intensity)

ds_factor = 4

fiber = np.load('GRIN_rod_indices.npy')
fiber = fiber[::ds_factor, ::ds_factor]

print(f'index distribution shape : {fiber.shape}')

# Find global min and max for consistent colormap scaling
vmin = np.min(data)*0.8
vmax = np.max(data)*0.8
norm = Normalize(vmin=vmin, vmax=vmax)

# Create individual frames
for i in range(data.shape[0]):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(data[i], cmap='jet', norm=norm, origin='lower',)
    
    x_tick_positions = [0, 127.5, 255.5]
    x_tick_labels = [-75, 0, 75]
    plt.xticks(ticks=x_tick_positions, labels=x_tick_labels)
    plt.xlabel(r'x ($\mu m$)')

    y_tick_positions = [0, 127.5, 255.5]
    y_tick_labels = [-75, 0, 75]
    plt.yticks(ticks=y_tick_positions, labels=y_tick_labels)
    plt.ylabel(r'y ($\mu m$)')

    # Add colorbar
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.1)
    # plt.colorbar(im, cax=cax)
    
    # Add frame number
    # ax.set_title(f'Frame {i}')

    # Add fiber structure
    ax.contour(fiber, levels=[np.min(fiber)], colors='white', linewidths=2)
    
    # Save frame
    plt.savefig(f'frames/frame_{i:03d}.png')
    plt.close()

# Create MP4 using imageio
fps = 10  # Frames per second
with imageio.get_writer('fiber_fields_linear_noise.mp4', mode='I', fps=fps) as writer:
    for i in range(data.shape[0]):
        image = imageio.imread(f'frames/frame_{i:03d}.png')
        writer.append_data(image)

print("MP4 created: field_evolution.mp4")