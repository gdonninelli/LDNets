"""Validate the PDE residual mapping on FOM ground truth (physical units).

PDE (TestCase 1b): dz/dt - mu1*d2z/dx2 + mu3*z = A(t)*cos(pi*x - P(t))
  mu1 = 0.05, mu3 = 0.02. Grid is periodic in x (endpoint excluded),
  so d2z/dx2 uses wrapped central differences.
"""
import numpy as np

MU1, MU3 = 0.05, 0.02

d = np.load('/data/ADR/data_1b.npy',
            allow_pickle=True)[()]
t, x = d['t'], d['x']
z = d['output']          # (samples, times, points), physical units
A = d['forcing'][..., 0][:, :, None]  # (samples, times, 1)
P = d['forcing'][..., 1][:, :, None]

dt = t[1] - t[0]
dx = x[1] - x[0]
assert np.allclose(np.diff(t), dt) and np.allclose(np.diff(x), dx)

dzdt = np.gradient(z, dt, axis=1)
d2zdx2 = (np.roll(z, -1, axis=2) - 2 * z + np.roll(z, 1, axis=2)) / dx**2 #* second derivative of z with respect to x using central differences and periodic boundary conditions
f = A * np.cos(np.pi * x[None, None, :] - P)

r = dzdt - MU1 * d2zdx2 + MU3 * z - f #* calculate the difference between the left-hand side and right-hand side of the PDE to get the residual which is measured on every point in the grid, for every time step, for every sample in the batch

print(f'grid: {z.shape}, dt={dt}, dx={dx}')
for name, term in [('dz/dt', dzdt), ('mu1*d2z/dx2', MU1 * d2zdx2),
                   ('mu3*z', MU3 * z), ('f', f)]:
    print(f'RMS({name:12s}) = {np.sqrt(np.mean(term**2)):.3e}')
print(f'RMS(residual)  = {np.sqrt(np.mean(r**2)):.3e}')
print(f'max|residual|  = {np.abs(r).max():.3e}')
print(f'rel RMS res/f  = {np.sqrt(np.mean(r**2)) / np.sqrt(np.mean(f**2)):.3e}')
print(f'rel max res/f  = {np.abs(r).max() / np.abs(f).max():.3e}')

# ponytail: full-grid FD check; wrap-around d2z/dx2 exact only for periodic grid
