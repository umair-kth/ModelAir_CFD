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
    ax.set_ylabel(r"$y$");  ax.set_ylim(0,1)
    ax.view_init(30,225)
    ax.tick_params(which='both', direction="out", width=1, length=5, labelsize=12,
                    labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                    bottom=True, top=False, left=True, right=False, pad=5
                    )
    ax.set_xlabel(r'$x$',fontsize=15,labelpad=5)
    ax.set_ylabel(r'$y$',fontsize=15,labelpad=10)
    plt.savefig(fname, bbox_inches="tight")
    print('Written file', fname) 
    

def laplace2D(p,y,dx,dy,target_norm):
    norm=1
    pn = np.empty_like(p)
    while norm > target_norm:
        pn = p.copy()
        p[1:-1,1:-1] =  (
                            (dy**2 * (pn[2:,1:-1] + pn[:-2,1:-1]))   +
                            (dx**2 * (pn[1:-1,2:] + pn[1:-1,:-2]))
                        ) / (2* (dx**2 + dy**2))
        p[0,:] = 0                     # p=0 at x=0
        p[-1,:]= y                     # p=y at x=2
        p[:,0] = p[:,1]                # dp/dy=0 at y=0
        p[:,-1] = p[:,-2]              # dp/dy=0 at y=1
        norm = (np.sum(np.abs(p)) - np.sum(np.abs(pn)))/(np.sum(np.abs(pn)))
    return p


Lx = 2; nx = 31; dx = Lx/(nx-1); x = np.linspace(0, Lx, nx)
Ly = 1; ny = 31; dy = Ly/(ny-1); y = np.linspace(0, Ly, ny)

p = np.zeros((nx,ny))
p[0,:] = 0                     # p=0 at x=0
p[-1,:]= y                     # p=y at x=2
p[:,0] = p[:,1]                # dp/dy=0 at y=0
p[:,-1] = p[:,-2]              # dp/dy=0 at y=1

fname = plots_path+'5_laplace_2d_init.png'
plot2D(x,y,p,fname)

fname = plots_path+'5_laplace_2d_sol.png'
p = laplace2D(p,y,dx,dy,1e-4)
plot2D(x,y,p,fname)


 
