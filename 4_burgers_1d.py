import numpy as np
import sympy as sp
import pylab as pl
pl.ion()

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

plots_path='./'
# conservative colour palette appropriate for colour-blind (http://mkweb.bcgsc.ca/colorblind/)
Vermillion    = '#D55E00'
Blue          = '#0072B2'
BluishGreen   = '#009E73'
Orange        = '#E69F00'
SkyBlue       = '#56B4E9'
ReddishPurple = '#CC79A7'
Yellow        = '#F0E442'
Black         = '#000000'

x, nu, t = sp.symbols('x nu t')
phi = sp.exp(-(x-4*t)**2/(4*nu*(t+1))) + sp.exp(-(x-4*t-2*np.pi)**2/(4*nu*(t+1)))

phiprime = phi.diff(x)
u = -2*nu*(phiprime/phi)+4

from sympy.utilities.lambdify import lambdify
ufunc = lambdify ((t, x, nu), u)

Lx = 2*np.pi; nx = 101; dx = Lx/(nx-1)
nt = 100; nu = 0.07
dt = dx*nu; T = nt*dt

grid = np.linspace(0, Lx, nx)
u = np.empty(nx)
t = 0
u = np.asarray([ufunc(t, x, nu) for x in grid])

ui = u.copy()

# Generate a plot of the signal and its power spectrum
fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=300)
ax.plot(grid, ui, c='grey',alpha=1)
ax.set_ylim(bottom=0, top=8)
ax.set_xlim(left=0, right=Lx)
fig.tight_layout()

fnam = plots_path+'4_burgers_1d_init.png'
plt.savefig(fnam, bbox_inches="tight")
print('Written file', fnam) 

for n in range(nt):
    un = u.copy()
    for i in range(nx-1):
        u[i] = un[i] - un[i] * dt/dx * (un[i]-un[i-1]) + \
            nu * dt/(dx**2) * (un[i+1] - 2*un[i] + un[i-1])
    # infer the periodicity
    u[-1] = un[-1] - un[-1] * dt/dx * (un[-1]-un[-2]) + \
            nu * dt/(dx**2) * (un[0] - 2*un[-1] + un[-2])

u_analytical = np.asarray([ufunc(T, xi, nu) for xi in grid])

# Generate a plot of the signal and its power spectrum
fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=300)
ax.plot(grid, ui, c='grey',alpha=0.4)
ax.plot(grid, u_analytical, c=Vermillion,alpha=1,label=r'$Analytical$')
ax.plot(grid, u, c=BluishGreen,alpha=1,label=r'$Computational$')
ax.set_ylim(bottom=0, top=8)
ax.set_xlim(left=0, right=Lx)
ax.legend(loc='best', ncols=2,frameon=False)
ax.tick_params(which='both', direction="in", width=1, length=5, labelsize=12,
                labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                bottom=True, top=False, left=True, right=False, pad=5
                )
ax.set_xlabel(r'$x$',fontsize=15,labelpad=5)
ax.set_ylabel(r'$u$',fontsize=15,labelpad=10)
fig.tight_layout()

fnam = plots_path+'4_burgers_1d_sol.png'
plt.savefig(fnam, bbox_inches="tight")
print('Written file', fnam) 




