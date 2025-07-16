import numpy as np
import torch

from .mmfsim import grid as grids
from .mmfsim.fiber import GrinFiber
from .mmfsim.modes import GrinLPMode

from .modes import calculate_modes, decompose_modes, n_to_lm, calculate_mode_field

from tqdm import tqdm

class Domain:
    def __init__(self, Lx, Ly, Lt, Nx, Ny, Nt, total_z, precision='single', device='cpu'):
        self.Lx = Lx
        self.Ly = Ly
        self.Lt = Lt
        self.Nx = Nx
        self.Ny = Ny
        self.Nt = Nt
        self.total_z = total_z
        self.device = device

        if precision == 'double':
            self.real_dtype = torch.float64
            self.complex_dtype = torch.complex128
        elif precision == 'single':
            self.real_dtype = torch.float32
            self.complex_dtype = torch.complex64

        self.T, self.X, self.Y = self.generate_grids(Lx, Ly, Lt, Nx, Ny, Nt)
        self.WT, self.KX, self.KY = self.generate_freqs(Lx, Ly, Lt, Nx, Ny, Nt)
        
        
    def generate_grids(self, Lx, Ly, Lt, Nx, Ny, Nt):
        x = torch.linspace(-Lx/2, Lx/2, Nx, device=self.device, dtype=self.real_dtype)
        y = torch.linspace(-Ly/2, Ly/2, Ny, device=self.device, dtype=self.real_dtype)
        t = torch.linspace(-Lt/2, Lt/2, Nt, device=self.device, dtype=self.real_dtype)
        T, X, Y = torch.meshgrid(t, x, y, indexing='ij')
        return T, X, Y

    def generate_freqs(self, Lx, Ly, Lt, Nx, Ny, Nt):
        kx = torch.fft.fftfreq(Nx, d=Lx/Nx).to(self.device) * 2 * torch.pi
        ky = torch.fft.fftfreq(Ny, d=Ly/Ny).to(self.device) * 2 * torch.pi
        wt = torch.fft.fftfreq(Nt, d=Lt/Nt).to(self.device) * 2 * torch.pi
        WT, KX, KY = torch.meshgrid(wt, kx, ky, indexing='ij')
        return WT, KX, KY


class KerrMedium:
    def __init__(self, domain, n0, n2, precision='single', device='cpu'):
        self.domain = domain
        self.n0 = n0
        self.n2 = n2

        if precision == 'double':
            self.real_dtype = torch.float64
            self.complex_dtype = torch.complex128
        elif precision == 'single':
            self.real_dtype = torch.float32
            self.complex_dtype = torch.complex64

        self.device = device
        self.n = torch.fill(self.domain.X, self.n0).to(self.device)

class Waveguide:
    def __init__(self, domain, nc, n0, n2=2e-20, radius=5e-6, beta2=16.55e-27,
                 structure_type="GRIN", disorder=False, precision='single', device='cpu'):
        self.device = device

        self.domain = domain
        self.nc = nc
        self.n0 = n0
        self.n2 = n2
        self.radius = radius
        self.beta2 = beta2

        if precision == 'single':
            self.real_dtype = torch.float32
            self.complex_dtype = torch.complex64
        elif precision == 'double':
            self.real_dtype = torch.float64
            self.complex_dtype = torch.complex128
        
        if structure_type == "GRIN":
            self.n = self.GRIN(disorder)
        elif structure_type in ("step", "step_index"):
            raise NotImplementedError("Step index fiber is not implemented yet.")
        else:
            raise ValueError("Invalid structure type. Choose 'GRIN' or 'step_index'.")

    def GRIN(self, disorder=False):
        n = torch.zeros_like(self.domain.X)
        R = torch.sqrt(self.domain.X**2 + self.domain.Y**2)
        delta = (self.nc**2 - self.n0**2) / (2 * self.nc**2)
        
        if disorder:
            perturbation = (torch.randn_like(n) - 0.5) * delta * 0.01
            perturbation[torch.where(R > self.radius)] = 0
        else:
            perturbation = 0

        n[torch.where(R > self.radius)] = self.n0
        n[torch.where(R <= self.radius)] = self.nc * torch.sqrt(1 - 2 * delta * (R[torch.where(R <= self.radius)]/self.radius)**2)         

        n = n + perturbation

        return n.to(self.device)

class Input:
    def __init__(self, domain, wvl0, n_core, n_clad, pulse=None, input_type="gaussian", beam_radius=50e-6, waveguide_radius=0., num_modes=0,
                 energy=1.0, phase_modulation=False, in_phase=True, coefficients=None, scale=1.0, cx=0, cy=0, precision='single', device='cpu'):
        
        self.domain = domain
        self.wvl0 = wvl0
        self.n_core = n_core
        self.n_clad = n_clad
        self.energy = energy
        self.waveguide_radius = waveguide_radius
        self.device = device

        if precision == 'double':
            self.real_dtype = torch.float64
            self.complex_dtype = torch.complex128
        elif precision == 'single':
            self.real_dtype = torch.float32
            self.complex_dtype = torch.complex64

        if input_type=="gaussian":
            self.field = self.gaussian_beam(beam_radius, cx=cx, cy=cy, phase_modulation=phase_modulation,)
        elif input_type=="mode":
            self.field = self.mode_mixing(num_modes, pulse=pulse, cx=cx, cy=cy, scale=scale, in_phase=in_phase, coefficients=coefficients)
        else:
            raise ValueError('Invalid Input Type')

    def gaussian_beam(self, w, pulse=None, cx=0, cy=0, phase_modulation=False, pixels=None,):
        R = torch.sqrt((self.domain.X-cx)**2 + (self.domain.Y-cy)**2)
        field = torch.exp(-R**2 / (w**2))

        if phase_modulation:
            phase_map = self.create_phase_map(pixels)
            phase_map = torch.exp(1j * phase_map)
            phase_map_fft = torch.fft.fftshift(torch.fft.fft2(phase_map))
            field = field * phase_map_fft

        dx = self.domain.Lx / self.domain.Nx
        dy = self.domain.Ly / self.domain.Ny
        dt = self.domain.Lt / self.domain.Nt

        field = field.to(self.device)
        field = normalize_field_to_power(field, self.energy, dx, dy, dt)
        field = torch.reshape(field, (1, self.domain.Nx, self.domain.Ny)) * torch.reshape(pulse, (self.domain.Nt, 1, 1)) 
        return field

    def mode_mixing(self, num_mode, pulse=None, coefficients=None, cx=0, cy=0, scale=1.0, in_phase=False):
        grid = grids.Grid(pixel_size=self.domain.Lx/self.domain.Nx, pixel_numbers=(self.domain.Nx, self.domain.Ny))
        grin_fiber = GrinFiber(radius=self.waveguide_radius, wavelength=self.wvl0, n1=self.n_core, n2=self.n_clad)
        
        total_field = torch.zeros_like(self.domain.X[0], dtype=self.complex_dtype).to(self.device)

        if coefficients is None:
            if in_phase:
                theta = np.zeros(num_mode)
                amp = np.random.uniform(0, 1, num_mode)
            else:
                theta = np.random.uniform(0, 2*np.pi, num_mode)
                amp = np.ones(num_mode)
            coefficients = amp * np.exp(1j * theta)
        else:
            if not in_phase:
                coefficients = coefficients * torch.exp(1j * torch.rand(num_mode) * 2 * torch.pi).to(self.device)
        for n in range(num_mode):            
            mode_field = calculate_mode_field(grid, grin_fiber, n).to(self.device)
            total_field += (coefficients[n] * mode_field)
        
        dx = self.domain.Lx / self.domain.Nx
        dy = self.domain.Ly / self.domain.Ny
        dt = self.domain.Lt / self.domain.Nt
        
        total_field = total_field.to(self.device)
        total_field = torch.reshape(total_field, (1, self.domain.Nx, self.domain.Ny)) * torch.reshape(pulse, (self.domain.Nt, 1, 1))
        total_field = normalize_field_to_power(total_field, self.energy, dx, dy, dt)
        total_field = torch.roll(total_field, shifts=(int(cx/dx), int(cy/dy)), dims=(0, 1))
        

        return total_field

def calculate_energy(field, dx, dy, dt):
    I_total = torch.abs(field)**2
    energy = torch.sum(I_total) * dx * dy * dt
    return energy


def normalize_field_to_power(field, E_target, dx, dy, dt, ):
    I_total = torch.abs(field)**2
    E_current = torch.sum(I_total) * dx * dy * dt
    scale = torch.sqrt(E_target / E_current)
    field *= scale

    return field

def run(domain, waveguide, input, boundary="absorbing",
                             dz=1e-06, num_field_sample=100, precision='single', ds_factor=1,):
    # device check
    input_device = input.device
    waveguide_device = waveguide.device
    domain_device = domain.device
    if input_device != waveguide_device or input_device != domain_device:
        raise ValueError('Input, fiber and domain should be on the same device')
    device = input_device

    # precision check    
    if precision == 'single':
        complex_dtype = torch.complex64
    elif precision == 'double':
        complex_dtype = torch.complex128
    else:
        raise ValueError("Precision must be 'single' or 'double'.")
    
    k0 = 2 * torch.pi / input.wvl0
    kz = (k0 * waveguide.n0)**2 - domain.KX**2 - domain.KY**2
    kz = kz.type(complex_dtype)
    Ds = torch.sqrt(kz)
    Kin = k0 * (waveguide.n - waveguide.n0)
    
  
    n_step = round(domain.total_z / dz)

    if num_field_sample > n_step:
        num_field_sample = n_step
    field_sample_interval = n_step // num_field_sample

    field_arr = torch.zeros((num_field_sample+1, domain.Nx // ds_factor, domain.Ny // ds_factor), dtype=complex_dtype)
    time_arr = torch.zeros((num_field_sample+1, domain.Nt), dtype=complex_dtype)
    energy_arr = torch.zeros((num_field_sample+1,), dtype=domain.real_dtype)

    X = domain.X = domain.X.to(device)
    Y = domain.Y = domain.Y.to(device)
    # T = domain.T = domain.T.to(device) we currently don't use T in the simulation

    if boundary == "periodic":
        boundary = 1.0
    elif boundary == "absorbing":
        boundary = torch.exp(-2*((torch.sqrt(X**2+Y**2)/(waveguide.radius*1.2))**5))
        boundary = boundary.to(device)

    Dt = ((-0.5 * waveguide.beta2) * (-1 * (domain.WT))**2)
    D = Ds

    E_real = input.field
    cnt = 0
    for i in tqdm(range(n_step), desc="Simulating propagation"):        
        if (i % field_sample_interval == 0) and cnt < num_field_sample:
            field_arr[cnt] = torch.sum(E_real[:, ::ds_factor, ::ds_factor], dim=0)
            time_arr[cnt] = torch.sum(torch.abs(E_real)**2, dim=(1, 2))
            energy_arr[cnt] = calculate_energy(E_real, domain.Lx/domain.Nx, domain.Ly/domain.Ny, domain.Lt/domain.Nt)
            cnt += 1
        
        E_fft = torch.fft.fft2(E_real)
        E_fft = E_fft * torch.exp(1j * D * dz/2)
        E_real = torch.fft.ifft2(E_fft)

        # Knl = waveguide.n2 * k0 * torch.abs(E_real)**2
        # N = Knl + Kin
        # E_real = E_real * torch.exp(1j * N * dz)

        E_fft = torch.fft.fft2(E_real)
        E_fft = E_fft * torch.exp(1j * D * dz/2)
        E_real = torch.fft.ifft2(E_fft)
        E_real = E_real * boundary

    return E_real, time_arr, energy_arr