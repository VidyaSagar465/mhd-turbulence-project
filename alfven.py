import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# grid setup
nx, ny, nz = 100, 100, 100
Lx, Ly, Lz = 1.0, 1.0, 1.0

x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
z = np.linspace(0, Lz, nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# parameters
B0 = 1.0
rho0 = 1.0

# GS parameters
l = 1.0
L = 1.0

vx = np.zeros_like(X)
vy = np.zeros_like(Y)
vz = np.zeros_like(Z)

Bx = np.zeros_like(X)
By = np.zeros_like(Y)
Bz = np.zeros_like(Z)

s_ux = np.zeros((nx, ny, nz))
s_uy = np.zeros((nx, ny, nz))
s_uz = np.zeros((nx, ny, nz))

k_list = []    # initial empty lists
deltaB_list = []
v_list = []
deltaB_sq = []

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

            k_vec = np.array([kx, ky, kz])
            k_mag = np.linalg.norm(k_vec)

            kpar = abs(kz)
            kperp = np.sqrt(kx**2 + ky**2)

            if kperp == 0 or kpar == 0 :
                continue

            # GS condition
            if kpar >= (kperp**(2/3)) * (l**(-1/3)):
                continue

            # amplitude
            prefactor = np.sqrt(L**3 / (6*np.pi))
            deltaB_amp = prefactor * (kperp**(-8/3)) * (l**(-1/6)) * kpar**(-2)
            deltaV_amp = deltaB_amp / np.sqrt(rho0)

            B0_vec = [0, 0, B0]

            # polarization
            pol = np.cross(k_vec, B0_vec)
            if np.linalg.norm(pol) == 0:
                continue
            pol /= np.linalg.norm(pol)

            vA = B0 / np.sqrt(rho0)
            omega = kz * vA

            if abs(omega) < 1e-10:
                continue

            phi0 = 2*np.pi*np.random.rand()
            phase = kx*X + ky*Y + kz*Z + phi0

            vx_m = deltaV_amp * pol[0]
            vy_m = deltaV_amp * pol[1]
            vz_m = deltaV_amp * pol[2]

            Bx_m = -(vx_m * kz * B0) / omega
            By_m = -(vy_m * kz * B0) / omega
            Bz_m = ((vx_m * kx * B0) - (vy_m * ky * B0)) / omega

            # store stats
            deltaB_m = np.sqrt(Bx_m**2 + By_m**2 + Bz_m**2)
            deltaV_m = np.sqrt(vx_m**2 + vy_m**2 + vz_m**2)

            v_list.append(deltaV_m)
            k_list.append(kperp)
            deltaB_list.append(deltaB_m)
            deltaB_sq.append(deltaB_m**2)

            # accumulate fields
            vx += vx_m * np.cos(phase)
            vy += vy_m * np.cos(phase)
            vz += vz_m * np.cos(phase)

            Bx += Bx_m * np.cos(phase)
            By += By_m * np.cos(phase)
            Bz += Bz_m * np.cos(phase)


sum_v = np.sqrt(np.mean(vx**2 + vy**2 + vz**2))
sum_B = np.sqrt(np.mean(Bx**2 + By**2 + Bz**2))

print(sum_v,sum_B)

df = pd.read_csv("alfven_mod.txt")
print(df)

plt.figure()
mesh = plt.pcolormesh(X[:,0,:],Z[:,0,:],Bx[:,0,:], shading='auto', cmap='viridis')
plt.show()

plt.figure()
mesh2 = plt.pcolormesh(X[:,:,0],Y[:,:,0],Bx[:,:,0], shading='auto', cmap='viridis')
plt.show()