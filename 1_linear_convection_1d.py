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

Lx = 2; nx = 41; dx = 2/(nx-1)
x = np.linspace(0,2,nx)
nt = 10; dt = 0.025
c = 1

u = np.ones(nx)
u[int(0.5/dx):int(1/dx+1)] = 2
ui=u.copy()
print(ui)

# Plotting initial profile
fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=300)
ax.plot(x, ui, c='grey',alpha=0.4)
ax.set_ylim(bottom=0.95, top=2.05)
ax.set_xlim(left=0, right=2)
fig.tight_layout()
fnam = plots_path+'1_linear_convec_1d_init.png'
plt.savefig(fnam, bbox_inches="tight")
print('Written file', fnam) 

scheme=2   # 0: Backward Difference 1: Forward Difference 2: Central Difference

un = np.ones(nx)
for n in range(nt):
    un=u.copy()
    if(scheme==0): # Backward Difference
        for i in range(1,nx): 
            u[i] = un[i] - c * dt/dx * ( un[i]-un[i-1] )
    if(scheme==1): # Forward Difference
       for i in range(0,nx-1): 
           u[i] = un[i] - c * dt/dx * ( un[i+1]-un[i] )
    if(scheme==2): # Central Difference
       for i in range(1,nx-1): 
           u[i] = un[i] - c * dt/dx * ( un[i+1]-un[i-1] )
            
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

if(scheme==0): suffix='bd'
if(scheme==1): suffix='fd'
if(scheme==2): suffix='cd'
fnam = plots_path+'1_linear_convec_1d_sol_'+suffix+'.png'
plt.savefig(fnam, bbox_inches="tight")
print('Written file', fnam) 

