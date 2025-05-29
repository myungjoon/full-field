import numpy as np
import torch
from .mmfsim import grid as grids
from .mmfsim.fiber import GrinFiber
from .mmfsim.modes import GrinLPMode
from .modes import calculate_modes, decompose_modes, n_to_lm, calculate_mode_field, calculate_mode_area

from tqdm import tqdm

class Domain:
    def __init__(self, Lx, Ly, Nx, Ny, precision='double', device='cpu'):
        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny
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


class Fiber:
    def __init__(self, domain, n_core, n_clad, total_z, dz, radius=5e-6, n2=0, 
                 structure_type="GRIN", disorder=False, precision='double', device='cpu'):
        self.domain = domain
        self.n_core = n_core
        self.n_clad = n_clad
        self.radius = radius
        
        self.total_z = total_z
        self.dz = dz
        self.n_step = int(total_z / dz)
        self.n2 = n2

        if precision == 'double':
            self.real_dtype = torch.float64
            self.complex_dtype = torch.complex128
        elif precision == 'single':
            self.real_dtype = torch.float32
            self.complex_dtype = torch.complex64

        self.device = device

        if structure_type == "GRIN":
            self.n = self.GRIN_fiber(disorder)
        elif structure_type == "homogeneous":
            self.n = self.homogeneous_fiber()
    
    def homogeneous_fiber(self,):
        n = torch.fill(self.domain.X, self.n_clad)
        return n.to(self.device)


    # def GRIN_fiber(self, disorder=False):
    #     n = torch.zeros_like(self.domain.X)
        
    #     if disorder:
    #         # Elliptical deformation parameters
    #         max_excursion = 5e-6  # 2 µm maximum excursion
        
    #         # 2D case - single random angle
    #         theta = np.random.random() * 2 * np.pi
        
    #         # Random ellipticity (semi-major and semi-minor axes)
    #         # Ensure the deformation doesn't exceed max_excursion
    #         a = self.radius + (np.random.random() - 0.5) * 2 * max_excursion
    #         b = self.radius + (np.random.random() - 0.5) * 2 * max_excursion
            
    #         # Apply rotation to coordinates
    #         X_rot = self.domain.X * np.cos(theta) - self.domain.Y * np.sin(theta)
    #         Y_rot = self.domain.X * np.sin(theta) + self.domain.Y * np.cos(theta)
            
    #         # Elliptical radius calculation
    #         R_elliptical = torch.sqrt((X_rot/a)**2 + (Y_rot/b)**2)
            
    #         # For coarse pitch variation along fiber length (if you have Z dimension)
    #         # This part would need adjustment based on your domain structure
    #         # For 2D case, we'll use a simplified approach
            
    #         core_mask = R_elliptical <= 1.0
    #         clad_mask = R_elliptical > 1.0
    #     else:
    #         # Original circular geometry
    #         R = torch.sqrt(self.domain.X**2 + self.domain.Y**2)
    #         core_mask = R <= self.radius
    #         clad_mask = R > self.radius
    #         R_elliptical = R / self.radius  # Normalized radius for GRIN calculation
        
    #     # Calculate delta
    #     delta = (self.n_core**2 - self.n_clad**2) / (2 * self.n_core**2)
        
    #     # Assign refractive indices
    #     n[clad_mask] = self.n_clad
    #     n[core_mask] = self.n_core * torch.sqrt(1 - 2 * delta * R_elliptical[core_mask]**2)
        
    #     return n.to(self.device)

    def GRIN_fiber(self, disorder=False):
        n = torch.zeros_like(self.domain.X)
        R = torch.sqrt(self.domain.X**2 + self.domain.Y**2)
        delta = (self.n_core**2 - self.n_clad**2) / (2 * self.n_core**2)
        if disorder:
            perturbation = (torch.randn_like(n) - 0.5) * delta * 0.01
            perturbation[torch.where(R > self.radius)] = 0
        else:
            perturbation = 0
        n[torch.where(R > self.radius)] = self.n_clad
        n[torch.where(R <= self.radius)] = self.n_core * torch.sqrt(1 - 2 * delta * (R[torch.where(R <= self.radius)]/self.radius)**2)         

        n = n + perturbation

        return n.to(self.device)

class Input:
    def __init__(self, domain, wvl0, n_core, n_clad, type="gaussian", beam_radius=50e-6, fiber_radius=0., num_modes=0,
                 power=1.0, phase_modulation=False, pixels=(32, 32), in_phase=False, cx=0, cy=0, l=0, m=1, precision='single', device='cpu'):
        
        self.domain = domain
        self.wvl0 = wvl0
        self.n_core = n_core
        self.n_clad = n_clad
        self.power = power
        self.fiber_radius = fiber_radius
        self.device = device

        if precision == 'double':
            self.real_dtype = torch.float64
            self.complex_dtype = torch.complex128
        elif precision == 'single':
            self.real_dtype = torch.float32
            self.complex_dtype = torch.complex64

        if type=="gaussian":
            self.field = self.gaussian_beam(beam_radius, cx=cx, cy=cy, phase_modulation=phase_modulation, pixels=pixels)
        elif type=="mode":
            self.field = self.mode_mixing(num_modes, cx=cx, cy=cy, in_phase=in_phase)
        elif type=="speckle":
            self.field = self.speckle_beam(cx=cx, cy=cy, pixels=pixels)
        elif type=="single mode":
            self.field = self.LP_modes(l, m)
        else:
            raise ValueError('Invalid Type')

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
        macropixel_size = N_half // N_pixel  # 각 매크로픽셀은 8x8 그리드 포인트로 구성
        
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

    def gaussian_beam(self, w, cx=0, cy=0, phase_modulation=False, pixels=None, phases=None):
        R = torch.sqrt((self.domain.X-cx)**2 + (self.domain.Y-cy)**2)
        field = torch.exp(-R**2 / w**2)

        if phase_modulation:
            phase_map = self.create_phase_map(pixels)
            # translate the phase map to the center of the beam with cx and cy
            # phase_map = torch.roll(phase_map, shifts=(int(cx/self.domain.Lx * self.domain.Nx), int(cy/self.domain.Ly * self.domain.Ny)), dims=(0, 1))
            # field = field * torch.exp(1j * 1.0 * np.pi * torch.rand_like(field))
            field = field * torch.exp(1j * phase_map)

        dx = self.domain.Lx / self.domain.Nx
        dy = self.domain.Ly / self.domain.Ny
        field = normalize_field_to_power(field.to(self.device), dx, dy, self.power)
        return field.to(self.device)

    def update_beam_phase(self, field, pixels, phases):
        phase_map = self.create_phase_map(pixels, phases=phases)
        field = field * torch.exp(1j * phase_map)

    def LP_modes(self, l, m,):
        grid = grids.Grid(pixel_size=self.domain.Lx/self.domain.Nx, pixel_numbers=(self.domain.Nx, self.domain.Ny))
        grin_fiber = GrinFiber(radius=self.fiber_radius, wavelength=self.wvl0, n1=self.n_core, n2=self.n_clad)
        mode = GrinLPMode(l, m)
        mode.compute(grin_fiber, grid)
        field = torch.tensor(mode._fields[:, :, 0])
        field = field.to(self.device)
        # Calculate the propagation constant for the mode
        k0 = 2 * np.pi / self.wvl0
        Delta = (self.n_core - self.n_clad) / self.n_core
        # beta = self.n_core * k0 * np.sqrt(1 - 2 * (2*m + l - 1) * np.sqrt(2*Delta) / self.n_core / k0 / self.fiber_radius)

        # field = normalize_field_to_power(field, self.domain.X, self.domain.Y, self.power)
        return field.to(self.device)

    def mode_mixing(self, num_mode, coefficients=None, cx=0, cy=0, in_phase=False):
        grid = grids.Grid(pixel_size=self.domain.Lx/self.domain.Nx, pixel_numbers=(self.domain.Nx, self.domain.Ny))
        grin_fiber = GrinFiber(radius=self.fiber_radius, wavelength=self.wvl0, n1=self.n_core, n2=self.n_clad)
        
        total_field = torch.zeros_like(self.domain.X, dtype=self.complex_dtype).to('cpu')

        if coefficients is None:
            if in_phase:
                theta = np.zeros(num_mode)
            else:
                theta = np.random.uniform(0, 2*np.pi, num_mode)
            amp = np.random.uniform(0, 1, num_mode)
            coefficients = amp * np.exp(1j * theta)
        for n in range(num_mode):            
            mode_field = calculate_mode_field(grid, grin_fiber, n)
            # mode = GrinLPMode(l, m)
            # mode.compute(grin_fiber, grid)
            
                # field = torch.tensor(mode._fields[:, :, 0]) # / torch.sqrt(summation) 
                # total_field += field * coefficients[n, 0]
            total_field += (coefficients[n] * mode_field)
        
        dx = self.domain.Lx / self.domain.Nx
        dy = self.domain.Ly / self.domain.Ny
        total_field = normalize_field_to_power(total_field.to(self.device), dx, dy, self.power)
        total_field = torch.roll(total_field, shifts=(int(cx/dx), int(cy/dy)), dims=(0, 1))
        return total_field.to(self.device)

def normalize_field_to_power(E_field, dx, dy, P_target):
    """
    Normalize the input field to have total power = P_target.

    Parameters:
        E_field : np.ndarray (complex) - 2D complex field E(x, y)
        dX, dY  : float - scalar grid spacing in x and y
        P_target: float - desired total power

    Returns:
        E_normalized : np.ndarray (complex) - normalized field
    """
    dA = dx * dy
    intensity = torch.abs(E_field)**2
    P_current = torch.sum(intensity) * dA
    scale = torch.sqrt(P_target / P_current)
    return E_field * scale


    
def run(domain, input, fiber, wvl0,  dz=1e-06, num_field_sample=100, num_mode_sample=500, precision='double', ds_factor=1, disorder=False):
    # device check
    input_device = input.device
    fiber_device = fiber.device
    domain_device = domain.device
    if input_device != fiber_device or input_device != domain_device:
        raise ValueError('Input, fiber and domain should be on the same device')
    device = input_device

    max_mode_num = 30

    if precision == 'double':
        real_dtype = torch.float64
        complex_dtype = torch.complex128
    elif precision == 'single':
        real_dtype = torch.float32
        complex_dtype = torch.complex64
    
    k0 = 2 * torch.pi / wvl0
    kz = (k0 * fiber.n_clad)**2 - domain.KX**2 - domain.KY**2
    kz = kz.type(complex_dtype)
    KZ = torch.sqrt(kz)
    Kin = k0 * (fiber.n - fiber.n_clad)
    
    n_step = int(fiber.total_z / dz)

    L_perturbation = fiber.total_z / 100

    
    if num_field_sample > n_step:
        num_field_sample = n_step
    field_sample_interval = n_step // num_field_sample

    if num_mode_sample > n_step:
        num_mode_sample = n_step
    mode_sample_interval = n_step // num_mode_sample

    energy_arr = torch.zeros(num_field_sample+1)
    field_arr = torch.zeros((num_field_sample+1, domain.Nx // ds_factor, domain.Ny // ds_factor), dtype=complex_dtype)
    modes_arr = torch.zeros((num_mode_sample, max_mode_num), dtype=real_dtype)
    modes = calculate_modes(domain, fiber, input, num_modes=max_mode_num, device=device)

    X = domain.X = domain.X.to(device)
    Y = domain.Y = domain.Y.to(device)

    absorption = torch.exp(-2*((torch.sqrt(X**2+Y**2)/(fiber.radius*1.5))**10))
    absorption = absorption.to(device)

    # if mode_decompose:
    #     modes = calculate_modes(domain, fiber, input, num_modes=num_modes, device=device)
    #     modes_arr = np.zeros((n_step, num_modes, 2), dtype=float)

    nl_phase = 0.0
    mode_area = calculate_mode_area(domain, fiber, mode=0, device=device)

    cnt = 0
    cnt_mode = 0
    cnt_perturbation = 0
    E_real = input.field
    for i in tqdm(range(n_step), desc="Simulating propagation"):
        if disorder and ((i+1)*dz > L_perturbation * cnt_perturbation):
            fiber.n = fiber.GRIN_fiber(disorder)
            Kin = k0 * (fiber.n - fiber.n_clad)
            cnt_perturbation += 1

        Knl = fiber.n2 * k0 * torch.abs(E_real)**2
        
        if (i % field_sample_interval == 0) and cnt < num_field_sample:
            energy = torch.sum(torch.abs(E_real)**2)
            energy_arr[cnt] = energy.item()
            field_arr[cnt] = E_real[::ds_factor, ::ds_factor]
            cnt += 1

        if (i % mode_sample_interval == 0) and cnt_mode < num_mode_sample:
            coefficients = decompose_modes(E_real, modes, max_mode_num)
            modes_arr[cnt_mode] = torch.abs(coefficients)
            cnt_mode += 1
        
        E_fft = torch.fft.fft2(E_real)
        E_fft = E_fft * torch.exp(1j  * KZ * dz/2)
        E_real = torch.fft.ifft2(E_fft)

        E_real = E_real * torch.exp(1j * (Kin + Knl) * dz)

        E_fft = torch.fft.fft2(E_real)
        E_fft = E_fft * torch.exp(1j  * KZ * dz/2)
        E_real = torch.fft.ifft2(E_fft)
        E_real = E_real * absorption

        # nl_phase = nl_phase + torch.sum(Knl) * dz

    energy = torch.sum(torch.abs(E_real)**2)
    energy_arr[cnt] = energy.item()
    field_arr[cnt] = E_real[::ds_factor, ::ds_factor]
    # if mode_decompose:
    #     return modes_arr, field_arr, energy_arr
    # else:
    return field_arr, energy_arr, modes_arr
