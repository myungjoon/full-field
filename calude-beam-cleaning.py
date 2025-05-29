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

seed = 43
np.random.seed(seed)
torch.manual_seed(seed)

class FiberPropagationModel:
    def __init__(self, n_core, NA, fiber_radius, input_radius, gamma=0.0, 
                 domain_size=50e-6, N=256, dz=5e-6, L=0.01, k0=0, power=1.0, wvl0=1064e-9,
                 use_complex128=True, device=None):
        """
        광섬유 빔 전파 모델 초기화
        """
        self.wvl0 = wvl0    
        self.n_core = n_core
        self.NA = NA
        self.n_clad = np.sqrt(n_core**2 - NA**2)
        self.fiber_radius = fiber_radius
        self.input_radius = input_radius
        self.gamma = gamma
        self.domain_size = domain_size
        self.N = N
        self.dz = dz
        self.L = L
        self.k0 = k0
        self.num_steps = int(L / dz)
        
        # 장치 설정
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 정밀도 설정
        if use_complex128:
            self.dtype = torch.complex128
            self.float_dtype = torch.float64
        else:
            self.dtype = torch.complex64
            self.float_dtype = torch.float32
        
        # 그리드 초기화
        self.initialize_grid()
        
        # 연산자 초기화
        self.initialize_operators()
        
        # 가우시안 진폭 초기화
        self.initialize_amplitude(total_power=power)

        self.absorption = torch.exp(-2*((torch.sqrt(self.X**2+self.Y**2)/(self.fiber_radius*1.5))**10))
        self.absorption = self.absorption.to(device)
    
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
        
        kz = ((self.k0 * self.n_clad)**2 - self.KX**2 - self.KY**2).to(self.dtype)
        kz = torch.sqrt(kz)
        self.diffraction_op = torch.exp(1j * kz  * self.dz / 2)
        kin = self.k0 * (n_profile - self.n_clad)
        self.refraction_op = torch.exp(1j * kin * self.dz).to(self.dtype)
    
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
        normalized_power = torch.sum(torch.abs(self.amplitude)**2) * dx**2
        print(f"Total input power: {normalized_power.item():.6f} W")
    
    def create_phase_map(self, phases):
        """
        16x16 매크로픽셀을 사용하여 위상 맵 생성
        """
        # 위상 맵 초기화
        phase_map = torch.zeros_like(self.R)
        
        

        # 코어 영역 정의
        N_phase = phase_map.shape[0]
        N_phase_half = N_phase // 2
        core_start = (self.N - N_phase_half) // 2
        
        # 매크로픽셀 크기 계산 (코어 영역을 16x16 매크로픽셀로 나눔)
        macropixel_size = N_phase_half // N_phase  # 각 매크로픽셀은 8x8 그리드 포인트로 구성
        
        # phases가 1차원이라면 16x16으로 변환
        if phases.dim() == 1:
            N_pixel = int(np.sqrt(len(phases)))
            phases = phases.reshape(N_pixel, N_pixel)
        else:
            N_pixel = phases.shape[0]
        
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

    def apply_nonlinearity(self, field):
        """비선형 효과와 굴절을 한 번에 적용"""
        intensity = torch.abs(field)**2
        nonlinear_phase = self.gamma * intensity * self.dz
        return field * torch.exp(1j * nonlinear_phase) * self.refraction_op

    def propagate_one_step(self, E_real,):
        """한 스텝 빔 전파"""
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
            # 체크포인트 없이 전파
            for i in tqdm(range(self.num_steps), desc="Simulating propagation"):
                if recording and (i % sample_interval == 0) and (cnt < n_sample):
                    field_arr[cnt] = field
                    cnt += 1
                field = self.propagate_one_step(field,)
                
        else:
            # 체크포인트를 사용한 전파
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
    
    def compute_mode_overlap_objective(self, output_field, target_mode):
        # 타겟 모드 정규화
        target_mode_norm = target_mode / torch.sqrt(torch.sum(torch.abs(target_mode)**2))
        
        # 출력 필드 정규화
        output_field_norm = output_field / torch.sqrt(torch.sum(torch.abs(output_field)**2))
        
        # 중첩 계산 (내적의 절대값 제곱)
        overlap = torch.abs(torch.sum(output_field_norm.conj() * target_mode_norm))**2
        
        # 최대화를 위해 음수 반환
        return -overlap
    
    def LP_modes(self, l, m,):
        grid = grids.Grid(pixel_size=self.domain_size/self.N, pixel_numbers=(self.N, self.N))
        grin_fiber = GrinFiber(radius=self.fiber_radius, wavelength=self.wvl0, n1=self.n_core, n2=self.n_clad)
        mode = GrinLPMode(l, m)
        mode.compute(grin_fiber, grid)
        field = torch.tensor(mode._fields[:, :, 0])
    
        return field.to(self.device)
    
    def mode_mixing(self, num_mode, coefficients=None, cx=0, cy=0,):
        grid = grids.Grid(pixel_size=self.domain_size/self.N, pixel_numbers=(self.N, self.N))
        grin_fiber = GrinFiber(radius=self.fiber_radius, wavelength=self.wvl0, n1=self.n_core, n2=self.n_clad)
        
        total_field = torch.zeros_like(self.X, dtype=self.dtype, device=self.device)
        for n in range(num_mode):
            l, m = n_to_lm(n+1)
            mode = GrinLPMode(l, m)
            mode.compute(grin_fiber, grid)
            
            if l== 0:
                field = torch.tensor(mode._fields[:, :, 0]) 
                total_field += field * coefficients[n, 0]
            else:
                field = torch.tensor(mode._fields)
                field = field[:, :, 0] * coefficients[n, 0] + field[:, :, 1] * coefficients[n,1]
                total_field += field
        
        dx = self.domain_size / self.N
        dy = self.domain_size / self.N
        total_field = normalize_field_to_power(total_field.to(self.device), dx, dy, self.power)
        # move total field with cx and cy
        total_field = torch.roll(total_field, shifts=(int(cx/dx), int(cy/dy)), dims=(0, 1))
        return total_field.to(self.device)

    def run(self,):
        N_pixel = 256
        random_phases = 2 * torch.pi * torch.rand(N_pixel**2, dtype=self.float_dtype, device=self.device)
        # phases = initial_phases.clone().requires_grad_(True)  # 미분 가능한 복사본 생성
        
        # # 최적화기 설정
        # optimizer = torch.optim.Adam([phases], lr=learning_rate)
        
        # # 목적 함수 값 기록
        # objective_values = []
        # overlap_values = []
        
    
        # 위상 맵 생성
        phase_map = self.create_phase_map(random_phases)
        # 입력 필드 생성
        input_field = self.amplitude * torch.exp(1j * phase_map)
        output, field_arr = self.propagate(input_field, recording=True)
            
        
        # 최적화된 위상값과 초기 위상값 반환
        return output.detach().cpu().numpy(), field_arr.detach().cpu().numpy()

    def optimize_phases(self, target_mode=None, num_iterations=100, learning_rate=0.01, use_checkpoint=True):
        if target_mode is None:
            target_mode = self.LP_modes(1, 1) 
        
        # 초기 위상 값 (0-2π 랜덤 값으로 초기화)
        initial_phases = 2 * torch.pi * torch.rand(256, dtype=self.float_dtype, device=self.device)
        # initial_phases = torch.zeros(256, dtype=self.float_dtype, device=self.device)
        phases = initial_phases.clone().requires_grad_(True)  # 미분 가능한 복사본 생성
        
        # 최적화기 설정
        optimizer = torch.optim.Adam([phases], lr=learning_rate)
        
        # 목적 함수 값 기록
        objective_values = []
        overlap_values = []
        
        # 최적화 루프
        for i in range(num_iterations):
            # 그래디언트 초기화
            optimizer.zero_grad()
            
            # 위상 맵 생성
            phase_map = self.create_phase_map(phases)
            
            # 입력 필드 생성
            input_field = self.amplitude * torch.exp(1j * phase_map)

            output_field = self.propagate(input_field, use_checkpoint)
            
        
        # 최적화된 위상값과 초기 위상값 반환
        return (phases.detach().cpu().reshape(16, 16), 
                initial_phases.detach().cpu().reshape(16, 16), 
                objective_values,
                overlap_values, target_mode.detach().cpu().numpy())

    def visualize_results(self, optimized_phases, initial_phases, target_mode, overlap_values=None):
        with torch.no_grad():
            # 초기 위상 맵 및 필드

            initial_phase_map = self.create_phase_map(initial_phases)
            initial_field = self.amplitude * torch.exp(1j * initial_phase_map)
            initial_output, initial_field_arr = self.propagate(initial_field, recording=True, n_sample=100)
            
            # 최적화된 위상 맵 및 필드
            optimized_phase_map = self.create_phase_map(
                torch.tensor(optimized_phases, device=self.device)
            )
            optimized_phase_map = self.create_phase_map(optimized_phases)
            optimized_field = self.amplitude * torch.exp(1j * optimized_phase_map)
            optimized_output, optimized_field_arr = self.propagate(optimized_field, recording=True, n_sample=100)
        
        initial_field_arr = initial_field_arr.cpu().numpy()
        optimized_field_arr = optimized_field_arr.cpu().numpy()
        np.save('initial_field_arr.npy', initial_field_arr)
        np.save('optimized_field_arr.npy', optimized_field_arr)

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
        plt.title('Initial Phase Pattern (16x16)')
        
        # 최적화된 위상 패턴
        plt.subplot(122)
        plt.imshow(optimized_phases, cmap='gray', vmin=0, vmax=2*np.pi)
        plt.colorbar(label='Phase (rad)')
        plt.xticks([])
        plt.yticks([])
        plt.title('Optimized Phase Pattern (16x16)')
        
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

wvl = 775e-9  # 파장 (m)
k0 = 2 * np.pi / wvl  # 파수 (m^-1)
gamma = 2.3e-20*k0

# for power in powers:
# 모델 초기화
fiber_radius_list = [25e-6, 50e-6, 100e-6, 200e-6]
input_radius_list = [20e-6, 40e-6, 50e-6, 50e-6]
N_list = [512, 1024, 2048, 4096]
ds_list = [1, 1, 2, 4]
power_list = [1000000, 1500000, 500000, 100000, 250000]

input_radius_list = [100e-6, 200e-6]

for i in range(5):
    fiber_radius = 450e-6
    N = 4096
    input_radius = 100e-6
    ds = 4
    power = power_list[i]
    # power = power_list[i]
    # input_radius = input_radius_list[i]
    
    # N = N_list[i]
    # ds = ds_list[i]
    
    
    model = FiberPropagationModel(
        n_core=1.47,
        NA=0.25,
        fiber_radius=fiber_radius,
        input_radius=input_radius,
        gamma=gamma, 
        domain_size=fiber_radius*4,
        N=N,
        dz=3e-6,
        k0=k0,
        L=0.3,
        power=power,
        use_complex128=False,
        device=device,
    )

    output, fields = model.run()
    fields = fields[:, ::ds, ::ds]
    np.save(f'./claude_gaussian_256pixel_{fiber_radius}_{input_radius}_{int(power)}_fields_on.npy', fields)
    np.save(f'./claude_gaussian_256pixel_{fiber_radius}_{input_radius}_{int(power)}_output_on.npy', output)