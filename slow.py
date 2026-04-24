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
P0 = 0.1
gamma = 5/3

cs = np.sqrt(gamma * P0 / rho0)
vA = B0 / np.sqrt(rho0)

B0_vec = np.array([0.0, 0.0, B0])


k_slow, dB2_slow = [], []
v_slo = []

s_ux = np.zeros((nx, ny, nz))
s_uy = np.zeros((nx, ny, nz))
s_uz = np.zeros((nx, ny, nz))

s_Bx = np.zeros((nx, ny, nz))
s_By = np.zeros((nx, ny, nz))
s_Bz = np.zeros((nx, ny, nz))

def ratio(omega, kxp, kzp):    # ratio btw uz and ux
    den = omega**2 - kzp**2 * cs**2
    if np.abs(den) < 1e-12:
        return 0.0

    return (kxp * kzp * cs**2) / den

kmax = 20   

for nx_m in range(-kmax, kmax+1):
    for ny_m in range(-kmax, kmax+1):
        for nz_m in range(-kmax, kmax+1):

            # skip zero mode
            if nx_m == 0 and ny_m == 0 and nz_m == 0:
                continue
            kx = 2*np.pi*nx_m / Lx
            ky = 2*np.pi*ny_m / Ly
            kz = 2*np.pi*nz_m / Lz

            k_mag = np.sqrt(kx**2 + ky**2 + kz**2)

            # GS decomposition
            kperp = np.sqrt(kx**2 + ky**2)   # k perpendicular and k parallel
            kpar = abs(kz)

            if kperp == 0 or kpar == 0:
                continue

            # GS condition or condition for g(x) to be one
            if kpar >= (kperp**(2/3)) * (l**(-1/3)):
                continue

            # GS amplitude
            prefactor = np.sqrt(L**3 / (6*np.pi))
            deltaB_amp = prefactor * (kperp**(-8/3)) * (l**(-1/6)) * kpar**(-2)

            amp = deltaB_amp / np.sqrt(rho0)

            # rotate k into x'–z' plane
            phi = np.arctan2(ky, kx)
            cos_phi, sin_phi = np.cos(phi), np.sin(phi)  
            kxp = np.sqrt(kx**2 + ky**2)
            kzp = kz
            cos_theta = kzp / k_mag

            # from MHD dispersion relation
            root = np.sqrt((vA**2 + cs**2)**2 - 4*vA**2*cs**2*cos_theta**2)
            omega_s = k_mag * np.sqrt(0.5*((vA**2 + cs**2) - root))

            if omega_s < 1e-6:
                continue

            phi0 = 2*np.pi*np.random.rand()  # random phase
            phase = kx*X + ky*Y + kz*Z + phi0

            rs = ratio(omega_s, kxp, kzp)

            # for slow mode
            uxp_s = amp 
            uzp_s = rs * uxp_s

            Bxp_s = -(kzp * B0 / omega_s) * uxp_s
            Bzp_s = -(kxp * B0 / omega_s) * uzp_s

            # rotate back
            ux_s = np.cos(-phi) * uxp_s 
            uy_s = np.sin(-phi) * uxp_s 
            uz_s = uzp_s

            Bx_s = np.cos(-phi) * Bxp_s 
            By_s = np.sin(-phi) * Bxp_s 
            Bz_s = Bzp_s


            s_ux += ux_s * np.cos(phase)
            s_uy += uy_s * np.cos(phase)
            s_uz += uz_s * np.cos(phase)

            s_Bx += Bx_s * np.cos(phase)
            s_By += By_s * np.cos(phase)
            s_Bz += Bz_s * np.cos(phase)

sum_u = np.sqrt(np.mean(s_ux**2 + s_uy**2 + s_uz**2))
sum_B = np.sqrt(np.mean(s_Bx**2 + s_By**2 + s_Bz**2))

print(sum_u, sum_B)

plt.figure()
mesh = plt.pcolormesh(X[:,0,:],Z[:,0,:],s_Bx[:,0,:], shading='auto', cmap='viridis')
plt.show()

plt.figure()
mesh2 = plt.pcolormesh(X[:,:,0],Y[:,:,0],s_Bx[:,:,0], shading='auto', cmap='viridis')
plt.show()