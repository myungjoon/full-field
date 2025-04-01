import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

def plot_mode_evolution(modes):
    plt.figure()
    for i in range(modes.shape[1]):
        l, m = n_to_lm(i+1)
        plt.plot(modes[:, i], label=f'LP{l}{m}')
    plt.xlabel('z')
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
        im = ax.imshow(np.abs(field)**2, cmap='jet', interpolation=interpolation)
    else:
        im = ax.imshow(np.abs(field)**2, cmap='jet',)
    ax.set_xticks(xtick)
    ax.set_yticks(ytick)
    ax.set_xticklabels([f'{x}' for x in xlabel])
    ax.set_yticklabels([f'{y}' for y in ylabel])
    # ax.set_xlim(-3001, 3001)
    # ax.set_ylim(-3001, 3001)
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
