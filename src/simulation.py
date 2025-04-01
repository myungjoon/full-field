import numpy as np
import torch
import mmfsim.grid as grids
from mmfsim.coupling import GrinFiberCoupler
from mmfsim.fiber import GrinFiber
from mmfsim.modes import GrinLPMode
from modes import calculate_modes, decompose_modes, n_to_lm

from tqdm import tqdm

class Domain:
    def __init__(self, Lx, Ly, Nx, Ny, device='cpu'):
        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny
        self.device = device
        self.X, self.Y = self.generate_grids(Lx, Ly, Nx, Ny,)
        self.KX, self.KY = self.generate_freqs(Lx, Ly, Nx, Ny,)
        

    def generate_grids(self, Lx, Ly, Nx, Ny, ):
        x = torch.linspace(-Lx/2, Lx/2, Nx, device=self.device, dtype=torch.float64)
        y = torch.linspace(-Ly/2, Ly/2, Ny, device=self.device, dtype=torch.float64)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        return X, Y

    def generate_freqs(self, Lx, Ly, Nx, Ny, ):
        kx = torch.fft.fftfreq(Nx, d=Lx/Nx).to(self.device) * 2 * torch.pi
        ky = torch.fft.fftfreq(Ny, d=Ly/Ny).to(self.device) * 2 * torch.pi
        KX, KY = torch.meshgrid(kx, ky, indexing='ij')
        return KX, KY


class Fiber:
    def __init__(self, domain, n_core, n_clad, total_z, dz, radius=5e-6, n2=0, structure_type="GRIN", disorder=False, device='cpu'):
        self.domain = domain
        self.n_core = n_core
        self.n_clad = n_clad
        self.radius = radius
        
        self.total_z = total_z
        self.dz = dz
        self.n_step = int(total_z / dz)
        self.n2 = n2

        self.device = device

        if structure_type == "GRIN":
            self.n = self.GRIN_fiber(disorder)
        elif structure_type == "homogeneous":
            self.n = self.homogeneous_fiber()
    
    def homogeneous_fiber(self,):
        n = torch.fill(self.domain.X, self.n_clad)
        return n.to(self.device)

    def GRIN_fiber(self, disorder=False):
        n = torch.zeros_like(self.domain.X)
        R = torch.sqrt(self.domain.X**2 + self.domain.Y**2)
        delta = (self.n_core**2 - self.n_clad**2) / (2 * self.n_core**2)
        if disorder:
            perturbation = torch.randn_like(n) * 0.0001
            perturbation[torch.where(R > self.radius)] = 0
        else:
            perturbation = 0
        n[torch.where(R > self.radius)] = self.n_clad
        
        n[torch.where(R <= self.radius)] = self.n_core * torch.sqrt(1 - 2 * delta * (R[torch.where(R <= self.radius)]/self.radius)**2) 
        n = n + perturbation

        return n.to(self.device)

class Input:
    def __init__(self, domain, wvl0, n_core, n_clad, radius=0., beam_radius=5e-6, num_mode=0, normalization=True, coefficients=None,
                 amp=1.0, noise=False, cx=0, cy=0, l=0, m=1, type="random", device='cpu'):
        self.domain = domain
        self.wvl0 = wvl0
        self.n_core = n_core
        self.n_clad = n_clad
        self.normalization = normalization
        self.amp = amp
        self.radius = radius
        self.device = device

        if type=="gaussian":
            self.field = self.gaussian_beam(beam_radius, cx=cx, cy=cy, noise=noise)
        elif type=="mode":
            self.field = self.LP_modes(l, m)
        elif type=="random":
            self.field = self.mode_mixing(num_mode, coefficients)            
        else:
            raise ValueError('Invalid Type')

    def gaussian_beam(self, w, cx=0, cy=0, noise=False):
        R = torch.sqrt((self.domain.X-cx)**2 + (self.domain.Y-cy)**2)
        field = torch.exp(-R**2 / w**2)
        field = field / torch.max(torch.abs(field)) * self.amp
        if noise:
            field = field * torch.exp(1j * 0.2 * np.pi * torch.rand_like(field))
        return field.to(self.device)

    def LP_modes(self, l, m,):
        grid = grids.Grid(pixel_size=self.domain.Lx/self.domain.Nx, pixel_numbers=(self.domain.Nx, self.domain.Ny))
        grin_fiber = GrinFiber(radius=self.radius, wavelength=self.wvl0, n1=self.n_core, n2=self.n_clad)
        mode = GrinLPMode(l, m)
        mode.compute(grin_fiber, grid)
        field = torch.tensor(mode._fields[:, :, 0])
        
        if self.normalization:
            summation = torch.sum(torch.abs(field)**2)
            field = field / torch.sqrt(summation)
        return field.to(self.device)

    def mode_mixing(self, num_mode, coefficients=None,):
        grid = grids.Grid(pixel_size=self.domain.Lx/self.domain.Nx, pixel_numbers=(self.domain.Nx, self.domain.Ny))
        grin_fiber = GrinFiber(radius=self.radius, wavelength=self.wvl0, n1=self.n_core, n2=self.n_clad)
        
        total_field = torch.zeros_like(self.domain.X, dtype=torch.complex128).to('cpu')
        for n in range(num_mode):
            l, m = n_to_lm(n+1)
            mode = GrinLPMode(l, m)
            mode.compute(grin_fiber, grid)
            
            if l== 0:
                # summation = torch.sum(torch.abs(torch.tensor(mode._fields[:, :, 0]))**2)
                field = torch.tensor(mode._fields[:, :, 0]) # / torch.sqrt(summation) 
                total_field += field * coefficients[n, 0]
            else:
                # summation = torch.sum(torch.abs(torch.tensor(mode._fields))**2, dim=(0,1))
                field = torch.tensor(mode._fields)
                field = field[:, :, 0] * coefficients[n, 0] + field[:, :, 1] * coefficients[n,1]
                total_field += field
            
        total_field = total_field * self.amp
                
        
        return total_field.to(self.device)

            
def run(domain, input, fiber, wvl0, n_sample=100, nonlinear=False, dz=1e-06, mode_decompose=False, num_modes=10):
    
    # device check
    input_device = input.device
    fiber_device = fiber.device
    domain_device = domain.device
    if input_device != fiber_device or input_device != domain_device:
        raise ValueError('Input, fiber and domain should be on the same device')
    device = input_device

    
    k0 = 2 * torch.pi / wvl0
    kz = (k0 * fiber.n_clad)**2 - domain.KX**2 - domain.KY**2
    kz = kz.type(torch.complex128)
    KZ = torch.sqrt(kz)
    Kin = k0 * (fiber.n - fiber.n_clad)
    
    n_step = int(fiber.total_z / dz)
    
    energy_arr = torch.zeros(n_sample)
    field_arr = torch.zeros((n_sample, domain.Nx, domain.Ny), dtype=torch.complex128)

    if mode_decompose:
        modes = calculate_modes(domain, fiber, input, num_modes=num_modes, device=device)
        modes_arr = np.zeros((n_step, num_modes), dtype=float)

    sample_interval = n_step // n_sample
    cnt = 0
    E_real = input.field
    for i in tqdm(range(n_step), desc="Simulating propagation"):
        # Index modulation
        if nonlinear:
            Knl = fiber.n2 * k0 * torch.abs(E_real)**2
        else:  
            Knl = 0

        E_fft = torch.fft.fft2(E_real)
        E_fft = E_fft * torch.exp(1j  * KZ * dz/2)
        E_real = torch.fft.ifft2(E_fft)

        E_real = E_real * torch.exp(1j * (Kin + Knl) * dz)

        E_fft = torch.fft.fft2(E_real)
        E_fft = E_fft * torch.exp(1j  * KZ * dz/2)
        E_real = torch.fft.ifft2(E_fft)

        if mode_decompose:
            coefficients = decompose_modes(E_real, modes,)
            modes_arr[i] = np.abs(coefficients)

        if (i % sample_interval == 0) and cnt < n_sample:
            energy = torch.sum(torch.abs(E_real)**2)
            energy_arr[cnt] = energy.item()
            field_arr[cnt] = E_real
            cnt += 1

    if mode_decompose:
        return E_real, modes_arr, field_arr, energy_arr
    else:
        return E_real, field_arr, energy_arr
