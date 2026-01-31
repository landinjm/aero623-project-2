import numpy as np
import matplotlib.pyplot as plt
from readgri import readgri
import sys


#-----------------------------------------------------------
def readU(fname):
    return np.loadtxt(fname)

#-----------------------------------------------------------
def getField(U, field):
    r, ru, rv, rE = [U[:,i] for i in range(4)]
    g = 1.4
    V = np.sqrt(ru**2 + rv**2)/r
    p = (g-1.)*(rE-0.5*r*V**2)
    c = np.sqrt(g*p/r)
    M = V/c
    s = field.lower()
    if (s == 'mach'):
        return M
    else:
        return []

#-----------------------------------------------------------
def getnodestate(Mesh, U):
    V = Mesh['V']; E = Mesh['E']
    Nv, Ne = V.shape[0], E.shape[0]
    UN = np.zeros(Nv); count = np.zeros(Nv)
    for e in range(Ne):
        for i in range(3):
            n = E[e,i];
            UN[n] += U[e]; count[n] += 1
    UN /= count
    return UN

#-----------------------------------------------------------
def plotmesh(Mesh, fname):
    V = Mesh['V']; E = Mesh['E']; BE = Mesh['BE']
    f = plt.figure(figsize=(12,12))
    plt.triplot(V[:,0], V[:,1], E, 'k-')
    for i in range(BE.shape[0]):
        plt.plot(V[BE[i,0:2],0],V[BE[i,0:2],1], '-', linewidth=1, color='black')
    dosave = not not fname
    plt.axis('equal'); plt.axis('off')
    plt.axis([-0.5, 1.5,-1, 1])
    plt.tick_params(axis='both', labelsize=12)
    f.tight_layout();
    if (dosave): plt.savefig(fname)
    else: plt.show(block=True);
    plt.close(f)
    
#-----------------------------------------------------------
def plotstate(Mesh, U, p, field, frange, fname):
    V = Mesh['V']; E = Mesh['E']; BE = Mesh['BE']
    f = plt.figure(figsize=(12,12))
    F = getField(U, field)
    if (p == 0):
        plt.tripcolor(V[:,0], V[:,1], triangles=E, facecolors=F, shading='flat')
    else:
        vc = np.linspace(frange[0], frange[1], 21) if (len(frange) > 0) else 20
        plt.tricontourf(V[:,0], V[:,1], E, getnodestate(Mesh,F), vc)
    for i in range(BE.shape[0]):
        plt.plot(V[BE[i,0:2],0],V[BE[i,0:2],1], '-', linewidth=2, color='black')
    dosave = not not fname
    plt.axis('equal'); plt.axis('off')
    if (len(frange)>0): plt.clim(frange[0], frange[1])
    plt.set_cmap('jet')
    cbar=plt.colorbar(orientation='horizontal', pad=-.12, fraction=.045)
    cbar.ax.tick_params(labelsize=16)
    plt.axis([-0.5, 1.5,-1, 1])
    plt.tick_params(axis='both', labelsize=12)
    f.tight_layout();
    if (dosave): plt.savefig(fname, bbox_inches='tight',pad_inches=-.2)
    else: plt.show(block=True);
    plt.close(f)
    
#-----------------------------------------------------------
def main():
    if (len(sys.argv) < 2):
        print('Pass at least one argument: Mesh')
    else:
        # fMesh, fU, p, field, fname
        Mesh = readgri(sys.argv[1])
        U = [] if (len(sys.argv) <= 2) else readU(sys.argv[2])
        p = 0 if (len(sys.argv) <= 3) else int(sys.argv[3])
        field = 'Mach' if (len(sys.argv) <= 4) else sys.argv[4]
        fname = [] if (len(sys.argv) <= 5) else sys.argv[5]
        if len(U) > 0:
            frange = []
            if (field.lower() == 'mach'):
                frange = [0.0, 0.5]
            plotstate(Mesh, U, p, field, frange, fname)
        else:
            plotmesh(Mesh, 'mesh.pdf')
    
if __name__ == "__main__":
    main()

