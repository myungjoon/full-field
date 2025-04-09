import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import imageio
from matplotlib.colors import Normalize

plt.rcParams['font.size'] = 14

# change to current file directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

filename = 'fields_rod_gaussian_on_16000'

field = np.load(f'{filename}.npy')
field = np.abs(field)**2

output_intensity = field[-1]
max_val = np.max(output_intensity)

ds_factor = 4

fiber = np.load('rod_GRIN.npy')
fiber = fiber[::ds_factor, ::ds_factor]

print(f'index distribution shape : {fiber.shape}')

# Create individual 
for i in range(field.shape[0]):

    vmin = np.min(field[i])
    vmax = np.max(field[i])
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(field[i], cmap='turbo', norm=norm, origin='lower',)
    

    x_tick_positions = [0, 255 , 511]
    x_tick_labels = [-1000, 0, 1000]
    plt.xticks(ticks=x_tick_positions, labels=x_tick_labels)
    plt.xlabel(r'x ($\mu m$)')

    # y_tick_positions = [0, 127.5, 255.5]
    x_tick_labels = [-1000, 0, 1000]
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
with imageio.get_writer(filename + '.mp4', fps=fps) as writer:
    for i in range(field.shape[0]):
        image = imageio.imread(f'frames/frame_{i:03d}.png')
        writer.append_data(image)

print(f"MP4 created: {filename}")