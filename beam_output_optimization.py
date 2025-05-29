import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os, argparse, time

from mmfsim import grid as grids
from mmfsim.fiber import GrinFiber
from mmfsim.modes import GrinLPMode
from src.modes import calculate_modes, decompose_modes, n_to_lm


plt.rcParams['font.size'] = 15
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

np.random.seed(42)
torch.manual_seed(42)

# arguments for total_power, beam_radius, position
parser = argparse.ArgumentParser(description='Simulation parameters')
parser.add_argument('--total_power', type=float, default=500e3, help='Total power (W)')
parser.add_argument('--beam_radius', type=float, default=50e-6, help='Beam radius (m)')
parser.add_argument('--precision', type=str, default='single', choices=['single', 'double'], help='Precision of the simulation')
parser.add_argument('--num_pixels', type=int, default=32, help='Number of pixels for the phase map')
parser.add_argument('--device_id', type=int, default=0, help='Device ID for CUDA')

args = parser.parse_args()


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
        
        # E_real = E_real * self.refraction_op
        E_real = self.apply_nonlinearity(E_real)
        # 굴절 및 비선형 단계
        # if self.gamma > 0:
            # intensity = 
        # nonlinear_phase = 
        # field = field * torch.exp(1j * self.gamma * torch.abs(field)**2 * self.dz) * self.refraction_op
        # else:
        
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
            # plot input_field
            # input_field_val = input_field.detach().cpu().numpy()
            # # 코어-클래딩 경계 계산
            # theta = np.linspace(0, 2*np.pi, 100)
            # core_radius_px = self.fiber_radius / (self.domain_size/self.N) # 픽셀 단위 코어 반지름
            # center_px = self.N // 2
            # core_x = center_px + core_radius_px * np.cos(theta)
            # core_y = center_px + core_radius_px * np.sin(theta)
            # plt.plot(core_y, core_x, 'w-', linewidth=1.5)  # 코어 경계 표시
        
            # plt.imshow(np.abs(input_field_val)**2, cmap='turbo')
            # plt.colorbar(label='Intensity')
            # plt.title('Input Field Intensity')
            # plt.show()

            # 전파
            output_field = self.propagate(input_field, use_checkpoint)
            
            # theta = np.linspace(0, 2*np.pi, 100)
            # core_radius_px = self.fiber_radius / (self.domain_size/self.N) # 픽셀 단위 코어 반지름
            # center_px = self.N // 2
            # core_x = center_px + core_radius_px * np.cos(theta)
            # core_y = center_px + core_radius_px * np.sin(theta)
            # plt.plot(core_y, core_x, 'w-', linewidth=1.5)  # 코어 경계 표시
        
            # plt.imshow(np.abs(output_field.detach().cpu().numpy())**2, cmap='turbo', interpolation="bilinear")
            # plt.colorbar(label='Intensity')
            # plt.title('Input Field Intensity')
            # plt.show()


            # 목적 함수 계산 (출력 필드와 타겟 모드 간의 중첩)
            objective = self.compute_mode_overlap_objective(output_field, target_mode)
            
            # 목적 함수 값 기록
            objective_values.append(objective.item())
            overlap_values.append(-objective.item())  # 실제 중첩 값 (양수)
            
            # 역전파
            objective.backward()
            
            # 파라미터 업데이트
            optimizer.step()
            
            # 위상값을 0-2π 범위로 제한 (옵션)
            with torch.no_grad():
                phases.data = phases.data % (2 * torch.pi)
            
            # 진행 상황 출력
            print(f"Iteration {i+1}, Overlap: {overlap_values[-1]:.6f}")
        
        # 최적화된 위상값과 초기 위상값 반환
        return (phases.detach().cpu().reshape(16, 16), 
                initial_phases.detach().cpu().reshape(16, 16), 
                objective_values,
                overlap_values, target_mode.detach().cpu().numpy())
    
    def LP_modes(self, l, m,):
        grid = grids.Grid(pixel_size=self.domain_size/self.N, pixel_numbers=(self.N, self.N))
        grin_fiber = GrinFiber(radius=self.fiber_radius, wavelength=self.wvl0, n1=self.n_core, n2=self.n_clad)
        mode = GrinLPMode(l, m)
        mode.compute(grin_fiber, grid)
        field = torch.tensor(mode._fields[:, :, 0])
    
        return field.to(self.device)

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
    
wvl = 1064e-9  # 파장 (m)
k0 = 2 * np.pi / wvl  # 파수 (m^-1)
gamma = 2.3e-20 * k0
num_iters=2


optimize(num_iters=num_iters, learning_rate=0.1
         use_checkpoint=True,
         target_field=target_field)

# 모델 초기화
model = FiberPropagationModel(
    n_core=1.47,
    NA=0.1950,
    fiber_radius=26e-6,
    input_radius=20e-6,
    gamma=gamma,  # 비선형성 정도 (0이면 선형)
    domain_size=100e-6,
    N=512,
    dz=5e-6,
    k0=k0,
    L=1.0,
    power=100000.0,
    use_complex128=False,
)


# 위상 최적화
optimized_phases, initial_phases, objective_values, overlap_values, target_mode = model.optimize_phases(
    num_iterations=12,
    learning_rate=0.10,
    use_checkpoint=True
)

# 결과 시각화
model.visualize_results(
    optimized_phases, 
    initial_phases, 
    target_mode,
    overlap_values
)

objective_values = np.array(objective_values)

# 목적 함수 값 변화 그래프
plt.figure(figsize=(10, 5))
plt.plot(-1 * objective_values)
plt.xlabel('Iteration')
plt.ylabel('Objective Value')
plt.grid(True)
plt.show()