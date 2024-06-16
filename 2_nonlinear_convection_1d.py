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
fig.tight_layout()
fnam = plots_path+'2_nonlinear_convec_1d_init.png'
plt.savefig(fnam, bbox_inches="tight")
print('Written file', fnam) 


CFL = 0.9
dt = CFL * dx / max(abs(u))
un = np.ones(nx) #initialize our placeholder array un, to hold the time-stepped solution

scheme=0   # 0: Backward Difference 1: Forward Difference 2: Central Difference

un = np.ones(nx)
for n in range(nt):
    un=u.copy()
    if(scheme==0): # Backward Difference
        for i in range(1,nx): 
            u[i] = un[i] - un[i] * dt/dx * ( un[i]-un[i-1] )
    if(scheme==1): # Forward Difference
       for i in range(0,nx-1): 
           u[i] = un[i] - un[i] * dt/dx * ( un[i+1]-un[i] )
    if(scheme==2): # Upwind 1st Order
        F = lambda c: (max(c/(abs(c)+1e-6), 0), max(-c/(abs(c)+1e-6), 0))    
        for i in range(1, nx-1):
            # Coefficients to the east side of the node (i+1)
            fe1, fe2 = F(u[i])
            # Coefficients to the west side of the node (i-1)
            fw1, fw2 = F(u[i])
            # Differential values on the east side interface
            ue = un[i] * fe1 + un[i+1] * fe2
            # Differential values on the wast side interface
            uw = un[i-1] * fw1 + un[i] * fw2
            u[i] = un[i] - un[i] * dt / dx * (ue - uw)
    

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
if(scheme==2): suffix='uw'
fnam = plots_path+'2_nonlinear_convec_1D_sol_'+suffix+'.png'
plt.savefig(fnam, bbox_inches="tight")
print('Written file', fnam) 

