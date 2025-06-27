import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from mmfsim import grid as grids
from mmfsim.fiber import GrinFiber
from mmfsim.modes import GrinLPMode
from src.modes import calculate_modes, decompose_modes, n_to_lm

plt.rcParams['font.size'] = 15
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

np.random.seed(40)
torch.manual_seed(40)
    
    def initialize_grid(self):
        """그리드 초기화"""
        dx = self.domain_size / self.N
        x = torch.linspace(-self.domain_size/2, self.domain_size/2, self.N, dtype=self.float_dtype, device=self.device)
        self.X, self.Y = torch.meshgrid(x, x, indexing='ij')
        self.R = torch.sqrt(self.X**2 + self.Y**2)
        
        # 주파수 영역 설정
        kx = 2 * np.pi * torch.fft.fftfreq(self.N, dx).to(dtype=self.float_dtype, device=self.device)
        self.KX, self.KY = torch.meshgrid(kx, kx, indexing='ij')
        self.KR2 = self.KX**2 + self.KY**2
    
    def initialize_operators(self):
        """회절 및 굴절 연산자 초기화"""
        
        # GRIN 광섬유 굴절률 프로파일
        delta = (self.n_core**2 - self.n_clad**2) / (2 * self.n_core**2)
        delta_tensor = torch.tensor(delta, dtype=self.float_dtype, device=self.device)
        
        n_profile = torch.zeros_like(self.R)
        n_profile[torch.where(self.R > self.fiber_radius)] = self.n_clad
        n_profile[torch.where(self.R <= self.fiber_radius)] = self.n_core * torch.sqrt(1 - 2 * delta * (self.R[torch.where(self.R <= self.fiber_radius)]/self.fiber_radius)**2) 
        self.delta_n = n_profile - self.n_clad
        
        kz = ((self.k0 * self.n_clad)**2 - self.KX**2 - self.KY**2).to(self.dtype)
        kz = torch.sqrt(kz)
        self.diffraction_op = torch.exp(1j * kz  * self.dz / 2)

        kin = self.k0 * (n_profile - self.n_clad)
        self.refraction_op = torch.exp(1j * kin * self.dz).to(self.dtype)
    
    
    def apply_nonlinearity(self, field):
        """비선형 효과와 굴절을 한 번에 적용"""
        intensity = torch.abs(field)**2
        nonlinear_phase = self.gamma * intensity * self.dz
        return field * torch.exp(1j * nonlinear_phase) * self.refraction_op

    def propagate_one_step(self, E_real,):
        
        # 회절 단계
        E_fft = torch.fft.fft2(E_real)
        E_fft = E_fft * self.diffraction_op
        E_real = torch.fft.ifft2(E_fft)
        
        E_real = self.apply_nonlinearity(E_real)

        E_fft = torch.fft.fft2(E_real)
        E_fft = E_fft * self.diffraction_op
        E_real = torch.fft.ifft2(E_fft)
        E_real = E_real * self.absorption

        return E_real
    
    def propagate(self, input_field, use_checkpoint=False, checkpoint_every=2000, recording=False, n_sample=100):
        """전체 광섬유 길이에 대한 빔 전파"""
        field = input_field
        if recording:
            sample_interval = self.num_steps // n_sample
            cnt = 0
            field_arr = torch.zeros((n_sample, self.N, self.N), dtype=self.dtype, device=self.device)
        if not use_checkpoint:
            # Checkpoint off
            for i in tqdm(range(self.num_steps), desc="Simulating propagation"):
                if recording and (i % sample_interval == 0) and (cnt < n_sample):
                    field_arr[cnt] = field
                    cnt += 1
                field = self.propagate_one_step(field,)
                
        else:
            # Checkpoint on
            def propagate_segment(field, steps):
                for _ in tqdm(range(steps), desc="Simulating propagation"):
                    field = self.propagate_one_step(field)
                return field
            
            for i in range(0, self.num_steps, checkpoint_every):
                steps = min(checkpoint_every, self.num_steps - i)
                field = torch.utils.checkpoint.checkpoint(
                    propagate_segment, field, steps
                )
        
        if recording:
            return field, field_arr
        else:
            return field
    

def compute_mode_overlap(output_field, target_mode_profile):
    target_mode_norm = target_mode_profile / torch.sqrt(torch.sum(torch.abs(target_mode_profile)**2))
    output_field_norm = output_field / torch.sqrt(torch.sum(torch.abs(output_field)**2))
    overlap = torch.abs(torch.sum(output_field_norm.conj() * target_mode_norm))**2
    return -1*overlap # for maximization
    

def create_phase_map(self, phases,):

    # 위상 맵 초기화
    if phases.dim() == 1:
        N_total_pixels = phases.shape[0]
        N_pixel = int(np.sqrt(N_total_pixels))
    else:
        N_pixel = phases.shape[0]

    phase_map = torch.zeros_like(self.R)
    
    # 코어 영역 정의 (중앙 128x128)
    N_phase = phase_map.shape[0]
    N_phase_half = N_phase // 2
    core_start = (self.N - N_phase_half) // 2
    core_end = core_start + N_phase_half
    
    # macropixel
    macropixel_size = N_phase_half // N_pixel
    
    if phases.dim() == 1:
        phases = phases.reshape(N_pixel, N_pixel)
    
    # 각 매크로픽셀에 해당하는 위상 적용
    for i in range(N_pixel):
        for j in range(N_pixel):
            # 매크로픽셀의 그리드 범위 계산
            row_start = core_start + i * macropixel_size
            row_end = row_start + macropixel_size
            col_start = core_start + j * macropixel_size
            col_end = col_start + macropixel_size
            
            # 해당 영역에 위상 적용
            phase_map[row_start:row_end, col_start:col_end] = phases[i, j]
    
    return phase_map
    
    

wvl = 1064e-9  # wavelength (m)
k0 = 2 * np.pi / wvl 


# Fiber parameters
n_core = 1.47
NA = 0.1950
fiber_radius = 26e-6
input_radius = 20e-6
domain_size = 100e-6
grid_number = 512
dz = 1e-5
fiber_length = 0.1
total_power = 500000.0

model = FiberPropagationModel(
    n_core=1.47,
    NA=0.1950,
    fiber_radius=fiber_radius,
    input_radius=input_radius,
    gamma=gamma, 
    domain_size=domain_size,
    N=grid_number,
    dz=dz,
    k0=k0,
    fiber_length=fiber_length,
    power=total_power,
    use_complex128=False,
)


def gaussian_beam(w, total_power=1.0):
    R = torch.sqrt(X**2 + Y**2)
    field = torch.exp(-R**2 / (w**2))

    dx = self.domain.Lx / self.domain.Nx
    dy = self.domain.Ly / self.domain.Ny
    field = normalize_field_to_power(field, dx, dy, total_power)
    return field

def initialize_amplitude(self, total_power=1.0):
    """
    가우시안 진폭 초기화 (전체 파워에 맞게 정규화)
    
    Parameters:
    -----------
    total_power : float
        입력 빔의 총 파워 (W)
    """
    # 가우시안 빔 진폭 계산
    self.amplitude = torch.exp(-(self.R**2) / (2 * self.input_radius**2))
    # self.amplitude = torch.exp(-(self.R**2) / (1 * self.input_radius**2))
    
    # 현재 총 파워 계산 (강도의 적분)
    dx = self.domain_size / self.N
    current_power = torch.sum(torch.abs(self.amplitude)**2) * dx**2
    
    # 원하는 총 파워로 정규화
    self.amplitude = self.amplitude * torch.sqrt(total_power / current_power)
    
    # 확인: 정규화 후 총 파워 계산
    normalized_power = torch.sum(torch.abs(self.amplitude)**2)
    print(f"Total input power: {normalized_power.item() * dx**2:.6f} W")

def LP_modes(l, m,):
    grid = grids.Grid(pixel_size=self.domain_size/self.N, pixel_numbers=(self.N, self.N))
    grin_fiber = GrinFiber(radius=self.fiber_radius, wavelength=self.wvl0, n1=self.n_core, n2=self.n_clad)
    mode = GrinLPMode(l, m)
    mode.compute(grin_fiber, grid)
    field = torch.tensor(mode._fields[:, :, 0])

    return field

def optimize(target_mode_profile, input_radius=25e-6, num_iterations=100, learning_rate=0.01, N_pixel=16, use_checkpoint=True, device='cpu'):

    initial_phases = 2 * torch.pi * torch.rand(N_pixel**2, device=device)
    phases = initial_phases.clone().requires_grad_(True)  # 미분 가능한 복사본 생성
    
    optimizer = torch.optim.Adam([phases], lr=learning_rate)
    input_field = gaussian_beam(input_radius, total_power=500e3).to(device)
    objective_values = np.zeros(num_iterations)
    
    for i in range(num_iterations):
        optimizer.zero_grad()
        
        phase_map = create_phase_map(phases,).to(device)
        input_field = input_field * torch.exp(1j * phase_map)
        output_field = run(input_field, use_checkpoint)
        objective = compute_mode_overlap(output_field, target_mode_profile)
        
        objective_values[i] = -1 * objective.item()

        objective.backward()
        optimizer.step()
        
        with torch.no_grad():
            phases.data = phases.data % (2 * torch.pi)
        
        print(f"Iteration {i+1}, Overlap: {objective_values[-1]:.6f}")
    
    optimized_phases = phases.reshape(N_pixel, N_pixel)
    initial_phases = initial_phases.reshape(N_pixel, N_pixel)


    return optimized_phases, initial_phases, objective_values


def visualize_results(self, optimized_phases, initial_phases, target_mode, overlap_values=None):
    initial_field_arr = initial_field_arr.cpu().numpy()
    optimized_field_arr = optimized_field_arr.cpu().numpy()
    np.save('initial_field_arr_22.npy', initial_field_arr)
    np.save('optimized_field_arr_22.npy', optimized_field_arr)

    # 코어-클래딩 경계 계산
    theta = np.linspace(0, 2*np.pi, 100)
    core_radius_px = self.fiber_radius / (self.domain_size/self.N) # 픽셀 단위 코어 반지름
    center_px = self.N // 2
    core_x = center_px + core_radius_px * np.cos(theta)
    core_y = center_px + core_radius_px * np.sin(theta)
    
    # 1. 강도 비교 (타겟, 초기, 최적화)
    plt.figure(figsize=(15, 5))
    
    # 타겟 모드 강도
    plt.subplot(131)
    target_intensity = np.abs(target_mode)**2
    plt.imshow(target_intensity, cmap='turbo')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(label='Intensity')
    plt.title('Target Mode Intensity')
    plt.plot(core_y, core_x, 'w-', linewidth=1.5)  # 코어 경계 표시
    
    # 초기 출력 강도
    plt.subplot(132)
    initial_intensity = torch.abs(initial_output.cpu())**2
    plt.imshow(initial_intensity, cmap='turbo')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(label='Intensity')
    plt.title('Initial Output Intensity')
    plt.plot(core_y, core_x, 'w-', linewidth=1.5)  # 코어 경계 표시
    
    # 최적화된 출력 강도
    plt.subplot(133)
    optimized_intensity = torch.abs(optimized_output.cpu())**2
    plt.imshow(optimized_intensity, cmap='turbo')
    plt.colorbar(label='Intensity')
    plt.title('Optimized Output Intensity')
    plt.plot(core_y, core_x, 'w-', linewidth=1.5)  # 코어 경계 표시
    
    plt.tight_layout()
    plt.show()
    
    # 2. 위상 맵 비교 (16x16 매크로픽셀)
    plt.figure(figsize=(10, 5))
    
    # 초기 위상 패턴
    plt.subplot(121)
    plt.imshow(initial_phases, cmap='gray', vmin=0, vmax=2*np.pi)
    plt.colorbar(label='Phase (rad)')
    plt.xticks([])
    plt.yticks([])
    plt.title(f'Initial Phase Pattern ({N_pixel}x{N_pixel})')
    
    # 최적화된 위상 패턴
    plt.subplot(122)
    plt.imshow(optimized_phases, cmap='gray', vmin=0, vmax=2*np.pi)
    plt.colorbar(label='Phase (rad)')
    plt.xticks([])
    plt.yticks([])
    plt.title(f'Optimized Phase Pattern ({N_pixel}x{N_pixel})')
    
    plt.tight_layout()
    plt.show()
    
    # 3. 전체 위상 맵
    plt.figure(figsize=(10, 5))
    
    # 초기 전체 위상 맵
    plt.subplot(121)
    plt.imshow(initial_phase_map.cpu(), cmap='gray', vmin=0, vmax=2*np.pi)
    plt.colorbar(label='Phase (rad)')
    plt.title('Initial Full Phase Map')
    plt.xticks([])
    plt.yticks([])
    plt.plot(core_y, core_x, 'w-', linewidth=1.5)  # 코어 경계 표시
    
    # 최적화된 전체 위상 맵
    plt.subplot(122)
    plt.imshow(optimized_phase_map.cpu(), cmap='gray', vmin=0, vmax=2*np.pi)
    plt.colorbar(label='Phase (rad)')
    plt.title('Optimized Full Phase Map')
    plt.xticks([])
    plt.yticks([])
    plt.plot(core_y, core_x, 'w-', linewidth=1.5)  # 코어 경계 표시
    
    plt.tight_layout()
    plt.show()

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

target_mode = (1, 1)  # LP11 모드
l, m = target_mode
target_mode_profile = LP_modes(l, m).to(device)

N_pixel = 16

optimized_phases, initial_phases, objective_values, = optimize(target_profile=target_mode_profile,
    N_pixel=N_pixel, 
    num_iterations=30,
    learning_rate=0.10,
    use_checkpoint=True
)


initial_output_field = run(initial_phases, initial_phases)
optimized_output_field = run(optimized_phases, optimized_phases)

initial_output_field = initial_output_field.cpu().numpy()
optimized_output_field = optimized_output_field.cpu().numpy()

optimized_phases = optimized_phases.detach().cpu().numpy()
target_mode_profile = target_mode_profile.detach().cpu().numpy()


compare_result(target_mode_profile, initial_output_field, optimized_output_field)

objective_values = np.array(objective_values)

# 목적 함수 값 변화 그래프
plt.figure(figsize=(10, 5))
plt.plot(objective_values)
plt.xlabel('Iteration')
plt.ylabel('Objective Value')
plt.grid(True)
plt.show()