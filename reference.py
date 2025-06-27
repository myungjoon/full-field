import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import scipy.io as sio
import time
from scipy import special
from tqdm import tqdm

def gaus2d(x=0, y=0, mx=0, my=0, sx=1, sy=1,p=40,w=26.5e-6):
  """define normalized 2D gaussian"""

  return np.exp(-2*((np.sqrt(x**2+y**2)/w)**p))


def modes(m,l,fwhm):
    # ref: Arash Mafi, "Bandwidth Improvement in Multimode Optical Fibers Via Scattering From Core Inclusions," J. Lightwave Technol. 28, 1547-1555 (2010)
    # mode numbers:
    p=l-1   # l-1 of LP_ml
    m=m   # m of LP_ml 

    Apm=np.sqrt(np.math.factorial(p)/np.pi/np.math.factorial(p+np.abs(m)))

    c = 299792458               # [m/s]
    n0 = 1.45                   # Refractive index of medium (1.44 for 1550 nm, 1.45 for 1030 nm)
    lambda_c = 1030e-9          # Central wavelength of the input pulse in [m]
    R = 25e-6                   # fiber radius
    w=2*cp.pi*c/lambda_c        # [Hz]
    k0 = w*n0/c
    delta = 0.01                #

    N_2 = 0.5*(R**2)*(k0**2)*(n0**2)*delta
    ro_0= R/(4*N_2)**0.25

    Epm=Apm*(np.sqrt(x1**2+y1**2)**np.abs(m))/(ro_0**(1+np.abs(m)))*np.exp(-(x1**2+y1**2)/2/ro_0**2)*special.eval_genlaguerre(p,np.abs(m),(x1**2+y1**2)/ro_0**2,out=None)

    Epm_=np.multiply(Epm,(np.cos(m*np.arctan2(y1,x1))+np.sin(m*np.arctan2(y1,x1))))
    phase_mod=np.abs(Epm_/np.max(np.abs(Epm_)))*np.pi
    gaussian_mag=np.exp( - (((x1**2)/(2*(fwhm/2.35482)**2)+ (y1**2)/(2*(fwhm/2.35482)**2))))
    return cp.asarray(gaussian_mag*np.exp(1j*phase_mod))

Nx, Ny = 256, 256
fiber_radius = 25e-6
# spacewidth = 54.1442e-6
spacewidth = fiber_radius * 4  # 4 times the fiber radius
xres=spacewidth/Nx
x = np.linspace(-spacewidth*0.5,spacewidth*0.5,int(spacewidth/xres))

xsteps=len(x)
y = x


x1, y1 = np.meshgrid(x, x) # get 2D variables instead of 1D
z = gaus2d(x1, y1)

cp_super_gauss2d =cp.asarray(z)
cp_super_gauss2d = cp.repeat(cp_super_gauss2d[:,:,cp.newaxis], 1, axis=2)  # Changed from 1024 to 1

FF=modes(1,1,20e-6)
print(cp.max(cp.abs(FF)))

FFabs=cp.abs(FF)
FFangle=cp.angle(FF)

ttt = time.time()
c = 299792458 # [m/s]
n0 = 1.45                   # Refractive index of medium (1.44 for 1550 nm, 1.45 for 1030 nm)
lambda_c = 775e-9          # Central wavelength of the input pulse in [m]

## TIME SPACE DOMAIN - CW VERSION
timewidth = 1.8e-12          # Width of the time window in [s]
tres = timewidth  # Single time point
t = cp.array([0.0])  # Single time point at t=0
timesteps = 1  # Single time step

x = cp.arange(-spacewidth*0.5,(spacewidth*0.5),xres)
y = x
[X,Y,T] = cp.meshgrid(x,y,t)


## FOURIER DOMAIN - CW VERSION
fs=1/timewidth
freq = cp.array([c/lambda_c])  # Single frequency at center wavelength
wave=c/freq # [m]
w=2*cp.pi*c/lambda_c # [Hz]
omegas=2*cp.pi*freq
wt = omegas-w

a = cp.pi/xres  # grid points in "frequency" domain--> {2*pi*(points/mm)}
N = len(x)
zbam = cp.arange(-a,(a-2*a/N)+(2*a/N),2*a/N)
kx = cp.transpose(zbam) # "frequency" domain indexing ky = kx; 
ky = kx
[KX,KY,WT] = cp.meshgrid(kx,ky,wt)

## OPERATORS
k0 = w*n0/c
n2 = 3.2e-20       #Kerr coefficient (m^2/W)
R = 25e-6
beta2 = 24.8e-27
beta3 = 23.3e-42
gamma = (2*cp.pi*n2/(lambda_c))
delta = 0.01
NL1 = -1j*((k0*delta)/(R*R))*((X**2)+(Y**2))

D1 = (0.5*1j/k0)*((-1j*(KX))**2+(-1j*(KY))**2)
D2 = ((-0.5*1j*beta2)*(-1j*(WT))**2)+((beta3/6)*(-1j*(WT))**3)  # Will be zero for CW
D = D1 + D2
s_imgper = (cp.pi*R)/cp.sqrt(2*delta)
dz = s_imgper/48
DFR = cp.exp(D*dz/2)

## INPUT 
flength = s_imgper*10
fstep = flength / dz
x_fwhm = 1
p_don=20
t_fwhm = 100e-15  # Not used in CW
Ppeak = 1e9 #270*50e3 # W 180
data_s=np.zeros((480,Nx,Ny))
data_t=np.zeros((480,1))  # Changed from 1024 to 1
fwhm=20e-6

for ulas2 in tqdm(range(1), desc='Processing'):
    coefs=np.random.rand(6)
    coefs=coefs/np.sum(coefs)
    A_transverse=cp.abs(modes(0,1,fwhm))*cp.exp(1j*(cp.angle(modes(0,2,fwhm))*coefs[1]+cp.angle(modes(0,3,fwhm))*coefs[2]+cp.angle(modes(1,1,fwhm))*coefs[3]+cp.angle(modes(1,2,fwhm))*coefs[4]+cp.angle(modes(2,1,fwhm))*coefs[5]+cp.angle(modes(0,1,fwhm))*coefs[0] ))
    
    # CW version - no time pulse, just constant amplitude
    pulse_time = cp.ones_like(T)  # Constant in time for CW
    # A=( pulse_time.transpose() * A_transverse.transpose() ).transpose()
    # A_tr_max =cp.max(cp.squeeze(cp.sum(cp.square(cp.abs(A)),axis=2)))
    # A=A/cp.sqrt(A_tr_max)*cp.sqrt(Ppeak/(cp.pi*(fwhm**2)))
    
    ### MAIN FUNCTION
    Ain = A_transverse.reshape([Nx, Ny, 1]) # Combine transverse and time components
    A_initial = Ain.get().squeeze()
    plt.imshow(np.abs(A_initial)**2, cmap='turbo', interpolation='nearest')
    plt.figure(2)
    plt.imshow(np.angle(A_initial), cmap='turbo', interpolation='nearest')
    plt.colorbar()
    plt.show()
    for ugur in range(int(fstep)):
        Einf=cp.fft.fftshift(cp.fft.fftn(Ain))
        Ein2=cp.fft.ifftn(cp.fft.ifftshift(Einf*DFR))
        Eout = Ein2
        
        NL2 = 1j*gamma*cp.abs(Eout)**2
        NL = NL1+NL2
        Eout = Eout*cp.exp(NL*dz)
        
        Einf=cp.fft.fftshift(cp.fft.fftn(Eout))
        Ein2=cp.fft.ifftn(cp.fft.ifftshift(Einf*DFR))
        Ain =cp.multiply(cp_super_gauss2d,Ein2)
        Ain_cpu=Ain

        Ain_cpu=cp.square(cp.abs(Ain_cpu))

        ss =cp.squeeze(cp.sum(Ain_cpu,axis=2))
        tt =cp.sum(cp.squeeze(cp.sum(Ain_cpu,axis=0)),axis=0)

        data_s[ugur,:,:]=ss.get()
        data_t[ugur,:]=tt.get()

A_dis=Ain.get().squeeze()

plt.imshow(np.abs(A_dis)**2, cmap='turbo', interpolation='nearest')
plt.show()
# A_disp=np.squeeze(np.sum(A_dis,axis=2))