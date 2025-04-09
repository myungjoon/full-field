import numpy as np
import torch

import mmfsim.grid as grids
from mmfsim.fiber import GrinFiber
from mmfsim.modes import GrinLPMode

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

class Mode:
    def __init__(self, domain, fiber, input_beam, l=0, m=0, device='cpu'):
        self.Lx = domain.Lx
        self.Ly = domain.Ly
        self.Nx = domain.Nx
        self.Ny = domain.Ny

        self.radius = fiber.radius
        self.n_core = fiber.n_core
        self.n_clad = fiber.n_clad

        self.wvl0 = input_beam.wvl0
        self.device = device
        self.l = l
        self.m = m

        # self.field = self.calculate_field()
       
    def calculate_field(self):       
        grid = grids.Grid(pixel_size=self.Lx/self.Nx, pixel_numbers=(self.Nx, self.Ny))
        grin_fiber = GrinFiber(radius=self.radius, wavelength=self.wvl0, n1=self.n_core, n2=self.n_clad)
        mode = GrinLPMode(self.l, self.m)
        mode.compute(grin_fiber, grid)
        
        field = torch.tensor(mode._fields)
        # summation = torch.sum(torch.abs(field)**2)
        field = field # / torch.sqrt(summation)
        return field.to(self.device)



def calculate_modes(domain, fiber, beam, num_modes=10, device='cpu'):
    modes = []
    for n in range(num_modes):
        l, m = n_to_lm(n+1)
        mode = Mode(domain, fiber, beam, l, m, device).calculate_field()
        modes.append(mode)

    return modes

def decompose_modes(field, modes, num_modes=10, dtype='real'):
    # Decompose the input beam into the modes of the fiber
    if dtype == 'real':
        coefficients = np.zeros((num_modes,2), dtype=float)
    elif dtype == 'complex':
        coefficients = np.zeros((num_modes,2), dtype=complex)
    else:
        raise ValueError('Invalid dtype')
    
    for n in range(num_modes):
        l, m = n_to_lm(n+1)
        
        if l == 0:
            overlap_0 = torch.sum(modes[n][:, :, 0] * field.conj())
            overlap_1 = torch.tensor(0.)
        else:
            overlap_0 = torch.sum(modes[n][:, :, 0] * field.conj())
            overlap_1 = torch.sum(modes[n][:, :, 1] * field.conj())
        
        if dtype == 'real':
            coefficients[n, 0] = torch.abs(overlap_0)
            coefficients[n, 1] = torch.abs(overlap_1)
        elif dtype == 'complex':
            coefficients[n, 0] = overlap_0
            coefficients[n, 1] = overlap_1

    return coefficients
