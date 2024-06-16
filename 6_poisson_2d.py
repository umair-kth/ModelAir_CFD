import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc, cm
from mpl_toolkits.mplot3d import Axes3D 
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


def plot2D(x,y,p,fname):
    fig = plt.figure( figsize=(11,7), dpi=100 )
    ax  = fig.add_subplot(projection='3d')
    X, Y = np.meshgrid(x,y)
    ax.plot_surface( X, Y, p[:], rstride=1, cstride=1, cmap=cm.viridis,
                     linewidth=0, antialiased=False )
    ax.set_xlabel(r"$x$");  ax.set_xlim(0,2)
    ax.set_ylabel(r"$y$");  ax.set_ylim(0,2)
    ax.view_init(30,225)
    ax.tick_params(which='both', direction="out", width=1, length=5, labelsize=12,
                    labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                    bottom=True, top=False, left=True, right=False, pad=5
                    )
    ax.set_xlabel(r'$x$',fontsize=15,labelpad=5)
    ax.set_ylabel(r'$y$',fontsize=15,labelpad=10)
    plt.savefig(fname, bbox_inches="tight")
    print('Written file', fname) 
    

def poisson2D(p,b,dx,dy,target_norm):
    norm=1; small=1e-8; niter=0
    pn = np.zeros_like(p)
    while norm > target_norm:
        pn = p.copy(); niter+=1
        p[1:-1,1:-1] =  ( 
                            (dy**2 * (pn[2:,1:-1] + pn[:-2,1:-1])    +
                             dx**2 * (pn[1:-1,2:] + pn[1:-1,:-2])    -
                             dx**2 * dy**2 * b[1:-1,1:-1])/
                            (2* (dx**2 + dy**2))
                        )
        p[0,:]  = 0               # p=0 at x=0
        p[-1,:] = 0               # p=0 at x=2
        p[:,0]  = 0               # p=0 at y=0
        p[:,-1] = 0               # p=0 at y=2
        norm = (np.sum(np.abs(p)) - np.sum(np.abs(pn)))/(np.sum(np.abs(pn))+small)
    return p, niter

Lx = 2; nx = 31; dx = Lx/(nx-1); x = np.linspace(0, Lx, nx)
Ly = 2; ny = 31; dy = Ly/(ny-1); y = np.linspace(0, Ly, ny)

p  = np.zeros((nx,ny))
p[0,:]  = 0               # p=0 at x=0
p[-1,:] = 0               # p=0 at x=2
p[:,0]  = 0               # p=0 at y=0
p[:,-1] = 0               # p=0 at y=2

b  = np.zeros((nx,ny))
b[int(nx/4), int(3*ny/4)] = -100
b[int(3*nx/4), int(ny/4)] = 100

fname = plots_path+'6_poisson_2d_init.png'
plot2D(x,y,b,fname)

fname = plots_path+'6_poisson_2d_sol.png'
p, niter = poisson2D(p,b,dx,dy,1e-4)
print('# of iterations =', niter)
plot2D(x,y,p,fname)


 
