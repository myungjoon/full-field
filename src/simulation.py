import numpy as np
import torch

from .mmfsim import grid as grids
from .mmfsim.fiber import GrinFiber
from .mmfsim.modes import GrinLPMode

from .modes import calculate_modes, decompose_modes, n_to_lm, calculate_mode_field, laguerre_gaussian_mode

from tqdm import tqdm

class Domain:
    def __init__(self, Lx, Ly, Nx, Ny, total_z, precision='single', device='cpu'):
        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny
        self.total_z = total_z
        self.device = device

        if precision == 'double':
            self.real_dtype = torch.float64
            self.complex_dtype = torch.complex128
        elif precision == 'single':
            self.real_dtype = torch.float32
            self.complex_dtype = torch.complex64

        self.X, self.Y = self.generate_grids(Lx, Ly, Nx, Ny,)
        self.KX, self.KY = self.generate_freqs(Lx, Ly, Nx, Ny,)
        
        
    def generate_grids(self, Lx, Ly, Nx, Ny, ):
        x = torch.linspace(-Lx/2, Lx/2, Nx, device=self.device, dtype=self.real_dtype)
        y = torch.linspace(-Ly/2, Ly/2, Ny, device=self.device, dtype=self.real_dtype)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        return X, Y

    def generate_freqs(self, Lx, Ly, Nx, Ny, ):
        kx = torch.fft.fftfreq(Nx, d=Lx/Nx).to(self.device) * 2 * torch.pi
        ky = torch.fft.fftfreq(Ny, d=Ly/Ny).to(self.device) * 2 * torch.pi
        KX, KY = torch.meshgrid(kx, ky, indexing='ij')
        return KX, KY


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
    def __init__(self, domain, nc, n0, n2=2e-20, radius=5e-6,
                 structure_type="GRIN", disorder=False, precision='single', device='cpu'):
        self.device = device

        self.domain = domain
        self.nc = nc
        self.n0 = n0
        self.n2 = n2
        self.radius = radius

        if precision == 'single':
            self.real_dtype = torch.float32
            self.complex_dtype = torch.complex64
        elif precision == 'double':
            self.real_dtype = torch.float64
            self.complex_dtype = torch.complex128
        
        if structure_type == "GRIN":
            self.n = self.GRIN_fiber(disorder)
        elif structure_type in ("step", "step_index"):
            raise NotImplementedError("Step index fiber is not implemented yet.")
        else:
            raise ValueError("Invalid structure type. Choose 'GRIN' or 'step_index'.")

    def GRIN_fiber(self, disorder=False):
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
    def __init__(self, domain, wvl0, n_core, n_clad, input_type="gaussian", beam_radius=50e-6, waveguide_radius=0., num_modes=0,
                 power=1.0, custom_fields=None, phase_modulation=False, 
                 in_phase=True, coefficients=None, 
                 scale=1.0, cx=0, cy=0, l=0, m=1, 
                 basis="LP",
                 precision='single', device='cpu'):
        
        self.domain = domain
        self.wvl0 = wvl0
        self.n_core = n_core
        self.n_clad = n_clad
        self.power = power
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
            self.field = self.mode_mixing(num_modes, cx=cx, cy=cy, scale=scale, in_phase=in_phase, coefficients=coefficients, basis=basis)
        elif input_type=="speckle":
            self.field = self.speckle_beam(cx=cx, cy=cy,)
        elif input_type=="fundamental":
            self.field = self.LP_modes(0, 1)
        elif input_type=="custom":
            self.field = custom_fields
            dx = self.domain.Lx / self.domain.Nx
            dy = self.domain.Ly / self.domain.Ny
            self.field = normalize_field_to_power(self.field.to(self.device), dx, dy, self.power)
            if scale != 1.0:
                self.field = scale_field(self.field, scale)
            self.field = torch.roll(self.field, shifts=(int(cx/dx), int(cy/dy)), dims=(0, 1))
            
        else:
            raise ValueError('Invalid Input Type')

    def speckle_beam(self, cx=0, cy=0, pixels=(32, 32)):
        # Generate a random phase map
        phase_map = self.create_phase_map(pixels)
        # Create a Gaussian beam with the random phase map
        field = self.gaussian_beam(self.fiber_radius, cx=cx, cy=cy, phase_modulation=True, pixels=pixels, phases=phase_map)
        return field

    def create_phase_map(self, pixels, phases=None):

        phase_map = torch.zeros_like(self.domain.X, dtype=self.real_dtype).to(self.device)
        N_pixel = pixels[0]

        # core domain
        N_total = self.domain.Nx
        
        N_half = N_total // 2
        core_start = N_half // 2
        
        # macropixel size
        macropixel_size = N_half // N_pixel 
        
        if phases is None:
            phases = torch.rand(N_pixel, N_pixel) *  2 * np.pi
        
        for i in range(N_pixel):
            for j in range(N_pixel):
                row_start = core_start + i * macropixel_size
                row_end = row_start + macropixel_size
                col_start = core_start + j * macropixel_size
                col_end = col_start + macropixel_size
                
                phase_map[row_start:row_end, col_start:col_end] = phases[i, j]
        
        return phase_map

    def gaussian_beam(self, w, cx=0, cy=0, phase_modulation=False, pixels=None,):
        R = torch.sqrt((self.domain.X-cx)**2 + (self.domain.Y-cy)**2)
        field = torch.exp(-R**2 / (w**2))

        if phase_modulation:
            phase_map = self.create_phase_map(pixels)
            phase_map = torch.exp(1j * phase_map)
            phase_map_fft = torch.fft.fftshift(torch.fft.fft2(phase_map))
            # translate the phase map to the center of the beam with cx and cy
            # phase_map = torch.roll(phase_map, shifts=(int(cx/self.domain.Lx * self.domain.Nx), int(cy/self.domain.Ly * self.domain.Ny)), dims=(0, 1))
            # field = field * torch.exp(1j * 1.0 * np.pi * torch.rand_like(field))
            field = field * phase_map_fft

        dx = self.domain.Lx / self.domain.Nx
        dy = self.domain.Ly / self.domain.Ny
        field = normalize_field_to_power(field.to(self.device), dx, dy, self.power)
        return field.to(self.device)

    def update_beam_phase(self, field, pixels, phases):
        phase_map = self.create_phase_map(pixels, phases=phases)
        field = field * torch.exp(1j * phase_map)

    def LP_modes(self, l, m,):
        epsilon = 1e-30
        grid = grids.Grid(pixel_size=self.domain.Lx/self.domain.Nx, pixel_numbers=(self.domain.Nx, self.domain.Ny))
        grin_fiber = GrinFiber(radius=self.waveguide_radius, wavelength=self.wvl0, n1=self.n_core, n2=self.n_clad)
        mode = GrinLPMode(l, m)
        mode.compute(grin_fiber, grid)
        field = torch.tensor(mode._fields[:, :, 0])
        # field += epsilon
        field = field.to(self.device)
        dx = self.domain.Lx / self.domain.Nx
        dy = self.domain.Ly / self.domain.Ny
        field = normalize_field_to_power(field, dx, dy, self.power)

        # Calculate the propagation constant for the mode
        k0 = 2 * np.pi / self.wvl0
        Delta = (self.n_core - self.n_clad) / self.n_core
        # beta = self.n_core * k0 * np.sqrt(1 - 2 * (2*m + l - 1) * np.sqrt(2*Delta) / self.n_core / k0 / self.fiber_radius)

        
        return field.to(self.device)

    def mode_mixing(self, num_mode, coefficients=None, cx=0, cy=0, scale=1.0, in_phase=False, w0=50e-6, basis="LP"):
        grid = grids.Grid(pixel_size=self.domain.Lx/self.domain.Nx, pixel_numbers=(self.domain.Nx, self.domain.Ny))
        grin_fiber = GrinFiber(radius=self.waveguide_radius, wavelength=self.wvl0, n1=self.n_core, n2=self.n_clad)
        
        total_field = torch.zeros_like(self.domain.X, dtype=self.complex_dtype).to(self.device)

        if basis == "LP":
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

        elif basis == "LG":
            # TODO: implement Laguerre-Gaussian modes using torch
            R = torch.sqrt(self.domain.X**2 + self.domain.Y**2).cpu().numpy()
            PHI = torch.arctan2(self.domain.Y, self.domain.X).cpu().numpy()
            for n in range(num_mode):
                total_field += (coefficients[n] * torch.tensor(laguerre_gaussian_mode(R, PHI, 0, n, 0, w0, self.wvl0))).to(self.device) # add only the radial modes

        
        
        dx = self.domain.Lx / self.domain.Nx
        dy = self.domain.Ly / self.domain.Ny
        if scale != 1.0:
            total_field = scale_field(total_field, scale)
        total_field = normalize_field_to_power(total_field.to(self.device), dx, dy, self.power)
        total_field = torch.roll(total_field, shifts=(int(cx/dx), int(cy/dy)), dims=(0, 1))

        return total_field.to(self.device)


def scale_field(input_field, scale_factor):
    device = input_field.device
    dtype = input_field.dtype
    
    size = input_field.shape[0]

    center = size / 2.0
    
    y_coords = torch.arange(size, device=device, dtype=torch.float32)
    x_coords = torch.arange(size, device=device, dtype=torch.float32)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
    

    orig_y = (y_grid - center) / scale_factor + center
    orig_x = (x_grid - center) / scale_factor + center
    

    valid_mask = ((orig_y >= 0) & (orig_y < size) & 
                  (orig_x >= 0) & (orig_x < size))
    
    scaled_field = torch.zeros((size, size), device=device, dtype=dtype)
    
    # interpolation
    y_floor = torch.floor(orig_y[valid_mask]).long()
    x_floor = torch.floor(orig_x[valid_mask]).long()
    y_ceil = torch.clamp(y_floor + 1, max=size-1)
    x_ceil = torch.clamp(x_floor + 1, max=size-1)
    
    dy = orig_y[valid_mask] - y_floor.float()
    dx = orig_x[valid_mask] - x_floor.float()
    
    scaled_field[valid_mask] = (
        input_field[y_floor, x_floor] * (1 - dx) * (1 - dy) +
        input_field[y_floor, x_ceil] * dx * (1 - dy) +
        input_field[y_ceil, x_floor] * (1 - dx) * dy +
        input_field[y_ceil, x_ceil] * dx * dy
    )
    
    return scaled_field


def normalize_field_to_power(E_field, dx, dy, P_target):
    dA = dx * dy
    intensity = torch.abs(E_field)**2
    P_current = torch.sum(intensity) * dA
    scale = torch.sqrt(P_target / P_current)
    return E_field * scale


def calculate_effective_mode_area(mode_field, dx, dy):
    dA = dx * dy  # Area of a pixel
    integral_E2 = torch.sum(torch.abs(mode_field)**2) * dA
    integral_E4 = torch.sum(torch.abs(mode_field)**4) * dA

    A_eff = (integral_E2**2) / integral_E4 
    return A_eff

def run_spatiotemporal_simulation(domain, medium, input, boundary="periodic",
                                  dz=1e-06, num_field_sample=100, precision='single', ds_factor=1,):
    # device check
    input_device = input.device
    medium_device = medium.device
    domain_device = domain.device
    if input_device != medium_device or input_device != domain_device:
        raise ValueError('Input, fiber and domain should be on the same device')
    device = input_device

    # precision check    
    if precision == 'single':
        real_dtype = torch.float32
        complex_dtype = torch.complex64
    elif precision == 'double':
        real_dtype = torch.float64
        complex_dtype = torch.complex128
    else:
        raise ValueError("Precision must be 'single' or 'double'.")
    

    k0 = 2 * torch.pi / input.wvl0
    kz = (k0 * medium.n0)**2 - domain.KX**2 - domain.KY**2
    kz = kz.type(complex_dtype)
    KZ = torch.sqrt(kz)
    Kin = k0 * (medium.n - medium.n0)
    
    n_step = int(domain.total_z / dz)

    if num_field_sample > n_step:
        num_field_sample = n_step
    field_sample_interval = n_step // num_field_sample

    energy_arr = torch.zeros(num_field_sample+1)
    field_arr = torch.zeros((num_field_sample+1, domain.Nx // ds_factor, domain.Ny // ds_factor), dtype=complex_dtype)
    
    X = domain.X = domain.X.to(device)
    Y = domain.Y = domain.Y.to(device)

    dx = domain.Lx / domain.Nx
    dy = domain.Ly / domain.Ny

    if boundary == "periodic":
        # For periodic boundary conditions, we assume the field is periodic in both x and y directions
        boundary = 1.0
    elif boundary == "absorbing":
        boundary = torch.exp(-2*((torch.sqrt(X**2+Y**2)/(medium.radius*1.2))**10))
        boundary = boundary.to(device)

    D = KZ

    E_real = input.field
    E_shape = E_real.shape

    for i in tqdm(range(n_step), desc="Simulating propagation"):        
        if (i % field_sample_interval == 0) and cnt < num_field_sample:
            energy = torch.sum(torch.abs(E_real)**2)
            energy_arr[cnt] = energy.item()
            field_arr[cnt] = E_real[::ds_factor, ::ds_factor]
            cnt += 1
        
        E_fft = torch.fft.fft2(E_real)
        E_fft = E_fft * torch.exp(1j * D * dz/2)
        E_real = torch.fft.ifft2(E_fft)

        Knl = medium.n2 * k0 * torch.abs(E_real)**2
        N = Knl
        E_real = E_real * torch.exp(1j * (Kin + Knl) * dz)

        E_fft = torch.fft.fft2(E_real)
        E_fft = E_fft * torch.exp(1j * D * dz/2)
        E_real = torch.fft.ifft2(E_fft)
        E_real = E_real * boundary

def run(domain, medium, input, boundary="absorbing", dz=1e-06, num_field_sample=100, 
        num_mode_sample=500, precision='single', ds_factor=1, disorder=False, 
        trace_modes=False, calculate_nl_phase=False, max_mode_num = 20):
    
    # device check
    input_device = input.device
    medium_device = medium.device
    domain_device = domain.device
    if input_device != medium_device or input_device != domain_device:
        raise ValueError('Input, fiber and domain should be on the same device')
    device = input_device

    # precision check    
    if precision == 'single':
        real_dtype = torch.float32
        complex_dtype = torch.complex64
    elif precision == 'double':
        real_dtype = torch.float64
        complex_dtype = torch.complex128
    else:
        raise ValueError("Precision must be 'single' or 'double'.")
     
    k0 = 2 * torch.pi / input.wvl0
    kz = (k0 * medium.n0)**2 - domain.KX**2 - domain.KY**2
    kz = kz.type(complex_dtype)
    KZ = torch.sqrt(kz)
    Kin = k0 * (medium.n - medium.n0)
    
    n_step = round(domain.total_z / dz)

    L_perturbation = domain.total_z / 100
    
    if num_field_sample > n_step:
        num_field_sample = n_step
    field_sample_interval = n_step // num_field_sample

    if num_mode_sample > n_step:
        num_mode_sample = n_step
    mode_sample_interval = n_step // num_mode_sample

    energy_arr = torch.zeros(num_field_sample+1)
    field_arr = torch.zeros((num_field_sample+1, domain.Nx // ds_factor, domain.Ny // ds_factor), dtype=complex_dtype)
    Knl_arr = torch.zeros(num_field_sample+1)

    
    X = domain.X = domain.X.to(device)
    Y = domain.Y = domain.Y.to(device)

    dx = domain.Lx / domain.Nx
    dy = domain.Ly / domain.Ny

    if boundary == "periodic":
        # For periodic boundary conditions, we assume the field is periodic in both x and y directions
        boundary = 1.0
    elif boundary == "absorbing":
        boundary = torch.exp(-2*((torch.sqrt(X**2+Y**2)/(medium.radius*1.2))**10))
        boundary = boundary.to(device)

    if trace_modes:
        modes_arr = torch.zeros((num_mode_sample, max_mode_num), dtype=real_dtype)
        modes = calculate_modes(domain, medium, input, num_modes=max_mode_num, device=device)

    if calculate_nl_phase:
        nl_phase = torch.zeros(domain.Nx, domain.Ny, dtype=real_dtype, device=device)

    cnt = 0
    cnt_mode = 0
    cnt_perturbation = 0
    E_real = input.field

    max_trajectory = torch.zeros((n_step, 2), dtype=int)

    E_shape = E_real.shape

    for i in tqdm(range(n_step), desc="Simulating propagation"):
        # if disorder and ((i+1)*dz > L_perturbation * cnt_perturbation):
        #     medium.n = medium.GRIN_fiber(disorder)
        #     Kin = k0 * (medium.n - medium.n_clad)
        #     cnt_perturbation += 1

        # Trajectory tracking for off-axis input beams
        max_trajectory[i,0], max_trajectory[i, 1] = torch.unravel_index(torch.argmax(torch.abs(E_real)), E_shape)
        # Knl = medium.n2 * k0 * torch.abs(E_real)**2

        if trace_modes and (i % mode_sample_interval == 0) and cnt_mode < num_mode_sample:
            coefficients = decompose_modes(E_real, modes, max_mode_num)
            modes_arr[cnt_mode] = torch.abs(coefficients)
            cnt_mode += 1

        
        if (i % field_sample_interval == 0) and cnt < num_field_sample:
            energy = torch.sum(torch.abs(E_real)**2)
            energy_arr[cnt] = energy.item()
            Knl_arr[cnt] = torch.max(medium.n2 * k0 * torch.abs(E_real)**2).item()
            field_arr[cnt] = E_real[::ds_factor, ::ds_factor]
            cnt += 1
        
        E_fft = torch.fft.fft2(E_real)
        E_fft = E_fft * torch.exp(1j  * KZ * dz/2)
        E_real = torch.fft.ifft2(E_fft)

        # Knl = medium.n2 * k0 * torch.abs(E_real)**2
        Knl = 0
        E_real = E_real * torch.exp(1j * (Kin + Knl) * dz)

        E_fft = torch.fft.fft2(E_real)
        E_fft = E_fft * torch.exp(1j  * KZ * dz/2)
        E_real = torch.fft.ifft2(E_fft)
        E_real = E_real * boundary

        if calculate_nl_phase:
            intensity = torch.abs(E_real)**2
            nl_phase = nl_phase + k0 * medium.n2 * intensity * dz
    
    energy = torch.sum(torch.abs(E_real)**2)
    energy_arr[cnt] = energy.item()
    field_arr[cnt] = E_real[::ds_factor, ::ds_factor]

    if calculate_nl_phase:
        if trace_modes:
            return field_arr, energy_arr, modes_arr, nl_phase
        else:
            return field_arr, energy_arr, nl_phase, max_trajectory
            # return field_arr, energy_arr, nl_phase,
    else:
        if trace_modes:
            return field_arr, energy_arr, modes_arr
        else:
            return field_arr, energy_arr, 
