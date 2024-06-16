import numpy as np
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

Lx = 2; nx = 41; dx = Lx / (nx - 1)
x = np.linspace(0,2,nx)

nt = 20    #nt is the number of timesteps we want to calculate
#dt = .025  #dt is the amount of time each timestep covers (delta t)

u = np.ones(nx)  #as before, we initialize u with every value equal to 1.
u[:]= 1
u[int(.5 / dx) : int(1 / dx + 1)] = 2  #then set u = 2 between 0.5 and 1 as per our I.C.s
ui = u.copy()

# Plotting initial profile
fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=300)
ax.plot(x, ui, c='grey',alpha=0.4)
ax.set_ylim(bottom=0.95, top=2.05)
ax.set_xlim(left=0, right=2)
ax.tick_params(which='both', direction="in", width=1, length=5, labelsize=12,
                labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                bottom=True, top=False, left=True, right=False, pad=5
                )
fig.tight_layout()
fnam = plots_path+'3_diffusion_1d_init.png'
plt.savefig(fnam, bbox_inches="tight")
print('Written file', fnam) 


nu = 0.3 # diffusion coefficient
sigma = 0.2 # sigma is a numerical parameter
dt = sigma * dx**2 / nu  # dt is defined using sigma 

un = np.ones(nx)
for n in range(nt):
    un = u.copy() # copy the existing values of u into un
    for i in range(1, nx-1): # skip first and last component!
        u[i] = un[i] + nu * dt / dx**2 * (un[i+1] - 2 * un[i] + un[i-1])
            
# Generate a plot of the signal and its power spectrum
fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=300)
ax.plot(x, ui, c='grey',alpha=0.4)
ax.plot(x, u, c='grey')
ax.set_ylim(bottom=0.95, top=2.05)
ax.set_xlim(left=0, right=2)
ax.tick_params(which='both', direction="in", width=1, length=5, labelsize=12,
                labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                bottom=True, top=False, left=True, right=False, pad=5
                )
ax.set_xlabel(r'$x$',fontsize=15,labelpad=5)
ax.set_ylabel(r'$u$',fontsize=15,labelpad=10)
fig.tight_layout()
fnam = plots_path+'3_diffusion_1d_sol.png'
plt.savefig(fnam, bbox_inches="tight")
print('Written file', fnam) 

