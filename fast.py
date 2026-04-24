import numpy as np
import matplotlib.pyplot as plt

# grid setup
nx, ny, nz = 100, 100, 100
Lx = Ly = Lz = 1.0

x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
z = np.linspace(0, Lz, nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# parameters
L = 1.0 
l = 1.0

B0 = 1.0
rho0 = 1.0
P0 = 1.0
gamma = 5/3
C = 0.1

cs = np.sqrt(gamma * P0 / rho0)
vA = B0 / np.sqrt(rho0)

B0_vec = np.array([0.0, 0.0, B0])

N = 200   # no of loops or k vectors
nmax = 10

k_fast,u_fast, dB2_fast = [], [], []

def ratio(omega, kxp, kzp):      # ratio term is the relation btw u_z and u_x
    den = omega**2 - kzp**2 * cs**2
    if np.abs(den) < 1e-12:
        return 0.0
    return (kxp * kzp * cs**2) / den


s_ux = np.zeros((nx, ny, nz))
s_uy = np.zeros((nx, ny, nz))
s_uz = np.zeros((nx, ny, nz))

s_Bx = np.zeros((nx, ny, nz))
s_By = np.zeros((nx, ny, nz))
s_Bz = np.zeros((nx, ny, nz))


for _ in range(N): 

    while True:
        nx_m = np.random.randint(-nmax, nmax+1)
        ny_m = np.random.randint(-nmax, nmax+1)
        nz_m = np.random.randint(-nmax, nmax+1)
        if nx_m != 0 or ny_m != 0 or nz_m != 0:
            break

    kx = 2*np.pi*nx_m / Lx
    ky = 2*np.pi*ny_m / Ly
    kz = 2*np.pi*nz_m / Lz

    k_mag = np.sqrt(kx**2 + ky**2 + kz**2)

    amp = C * k_mag**(-4)

    phi = np.arctan2(ky, kx)
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)

    kxp = np.sqrt(kx**2 + ky**2)
    kzp = kz
    cos_theta = kzp / k_mag

    
    root = np.sqrt((vA**2 + cs**2)**2 - 4*vA**2*cs**2*cos_theta**2)
    omega_f = k_mag * np.sqrt(0.5*((vA**2 + cs**2) + root))

    phi0 = 2*np.pi*np.random.rand()   
    phase = kx*X + ky*Y + kz*Z + phi0

    # ratio
    rf = ratio(omega_f, kxp, kzp)

  # the equations in rotated frame
    uxp_f = amp
    uzp_f = rf * uxp_f

    Bxp_f = -(kzp * B0 / omega_f) * uxp_f
    Bzp_f = -(kxp * B0 / omega_f) * uzp_f

    # rotate back to get relations in initial frame
    
    ux_f = np.cos(-phi) * uxp_f
    uy_f = np.sin(-phi) * uxp_f 
    uz_f = uzp_f

    Bx_f = np.cos(-phi) * Bxp_f
    By_f = np.sin(-phi) * Bxp_f
    Bz_f = Bzp_f
    

    s_ux += ux_f * np.cos(phase)
    s_uy += uy_f * np.cos(phase)
    s_uz += uz_f * np.cos(phase)

    s_Bx += Bx_f * np.cos(phase)
    s_By += By_f * np.cos(phase)
    s_Bz += Bz_f * np.cos(phase)
        


sum_u = np.sqrt(np.mean(s_ux**2 + s_uy**2 + s_uz**2))
sum_B = np.sqrt(np.mean(s_Bx**2 + s_By**2 + s_Bz**2))

print(sum_u, sum_B)