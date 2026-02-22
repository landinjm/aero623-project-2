from enum import auto
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as col
from scipy import sparse


#-----------------------------------------------------------
# Identifies interior and boundary edges given element-to-node
# IE contains (n1, n2, elem1, elem2) for each interior edge
# BE contains (n1, n2, elem) for each boundary edge
def edgehash(E, B):
    Ne = E.shape[0]; Nn = np.amax(E)+1
    H = sparse.lil_matrix((Nn, Nn), dtype=np.int64)
    IE = np.zeros([int(np.ceil(Ne*1.5)),4], dtype=np.int64)
    ni = 0
    for e in range(Ne):
        for i in range(3):
            n1, n2 = E[e,i], E[e,(i+1)%3]
            if (H[n2,n1] == 0):
                H[n1,n2] = e+1
            else:
                eR = H[n2,n1]-1
                IE[ni,:] = n1, n2, e, eR
                H[n2,n1] = 0
                ni += 1
    IE = IE[0:ni,:]
    # boundaries
    nb0 = nb = 0
    for g in range(len(B)): nb0 += B[g].shape[0]
    BE = np.zeros([nb0,4], dtype=np.int64)
    for g in range(len(B)):
        Bi = B[g]
        for b in range(Bi.shape[0]):
            n1, n2 = Bi[b,0], Bi[b,1]
            if (H[n1,n2] == 0): n1,n2 = n2,n1
            BE[nb,:] = n1, n2, H[n1,n2]-1, g
            nb += 1
    return IE, BE

#-----------------------------------------------------------
def readgri(fname):
    f = open(fname, 'r')
    Nn, Ne, dim = [int(s) for s in f.readline().split()]
    # read vertices
    V = np.array([[float(s) for s in f.readline().split()] for n in range(Nn)])
    # read boundaries
    NB = int(f.readline())
    B = []; Bname = []
    for i in range(NB):
        s = f.readline().split(); Nb = int(s[0]); Bname.append(s[2])
        Bi = np.array([[int(s)-1 for s in f.readline().split()] for n in range(Nb)])
        B.append(Bi)
    # read elements
    Ne0 = 0; E = []
    while (Ne0 < Ne):
        s = f.readline().split(); ne = int(s[0])
        Ei = np.array([[int(s)-1 for s in f.readline().split()] for n in range(ne)])
        E = Ei if (Ne0==0) else np.concatenate((E,Ei), axis=0)
        Ne0 += ne
    # make IE, BE structures
    IE, BE = edgehash(E, B)
    Mesh = {'V':V, 'E':E, 'IE':IE, 'BE':BE, 'Bname':Bname }
    return Mesh

#-----------------------------------------------------------
def plotsolution(grifile,solutionfile,sname,outfile):
    # read mesh and solution
    Mesh = readgri(grifile)
    U = np.loadtxt(solutionfile)
    Uarray = np.array(U)
    Ut = Uarray.T
    U = Ut.tolist()
    N = len(U[0])

    # create useful scalars
    gamma = 1.4
    q = np.zeros(N)
    p = np.zeros(N)
    c = np.zeros(N)
    M = np.zeros(N)
    for i in range(N):
        q[i] = np.sqrt(np.square(U[1][i]) + np.square(U[2][i]))/U[0][i]
        p[i] = (gamma-1)*(U[3][i] - U[0][i]/2*np.square(q[i]))
        c[i] = np.sqrt(gamma*p[i]/U[0][i])
        M[i] = q[i]/c[i]

    # choose which plot
    match sname:
      case 'mach':
        S = M;
      case 'pressure':
        S = p;
      case 'entropy':
        S = np.zeros(N)
        for i in range(N):
            S[i] = p[i]/(U[0][i]**gamma);

    # Plot final solution
    tris = []
    for i in range(N):
        nodes = Mesh['V'][Mesh['E'][i]]
        tri = patches.Polygon(nodes)
        tris.append(tri)
    pc = col.PatchCollection(tris,cmap='jet')
    pc.set_array(S)
    #pc.set_clim(vmin=0.71, vmax=0.74)
    fig, ax = plt.subplots()
    ax.add_collection(pc)
    plt.colorbar(pc)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(sname)
    plt.axis('equal')
    plt.savefig(outfile, dpi=300)
    plt.close(fig)

#-----------------------------------------------------------
def ploterror(errorfile,outfile):
    # read error
    error = np.loadtxt(errorfile)

    # plot error
    plt.figure()
    plt.plot(error)
    plt.xlabel('Number of Iterations')
    plt.ylabel('$L_1$')
    plt.yscale('log')
    plt.title('Convergence of $L_1$')
    plt.savefig(outfile, dpi=300)
    plt.close()

#-----------------------------------------------------------#-----------------------------------------------------------
def plotcoefficients(coefficientsfile,direction,outfile):
    # read coefficients
    coefficients = np.loadtxt(coefficientsfile)

    # plot coefficients
    plt.figure()
    plt.plot(coefficients)
    plt.xlabel('Number of Iterations')
    plt.ylabel('$C_'+direction+'$')
    plt.title('Convergence of $C_'+direction+'$')
    plt.savefig(outfile, dpi=300)
    plt.close()
#-----------------------------------------------------------
def plotcp(grifile,solutionfile,outfile):
    # read mesh and solution
    Mesh = readgri(grifile)
    U = np.loadtxt(solutionfile)
    Uarray = np.array(U)
    Ut = Uarray.T
    U = Ut.tolist()
    N = len(U[0])

    # identify top & bottom surfaces
    n = len(Mesh['BE'])
    ntop = 0
    nbot = 0
    for i in range(n):
        if Mesh['Bname'][Mesh['BE'][i][3]] == 'BladeTop':
            if ntop == 0:
                topstart = i
            ntop += 1
        if Mesh['Bname'][Mesh['BE'][i][3]] == 'BladeBottom':
            if nbot == 0:
                botstart = i
            nbot += 1

    # create cp
    gamma = 1.4
    p0 = 1/gamma    # rho0*a0^2/gamma
    pout = 0.7*p0
    Mout2 = (2/(gamma-1))*((p0/pout)**((gamma-1)/gamma)-1)
    qout = 1/2*gamma*pout*Mout2
    top = np.zeros([ntop,3])
    bot = np.zeros([nbot,3])
    stop = np.zeros(ntop)
    sbot = np.zeros(nbot)
    # loop over top
    for i in range(ntop):
        vert1 = Mesh['BE'][i+topstart][0]
        vert2 = Mesh['BE'][i+topstart][1]
        elem = Mesh['BE'][i+topstart][2]
        top[i][0] = (Mesh['V'][vert1][0]+Mesh['V'][vert2][0])/2
        dx = (Mesh['V'][vert1][0]-Mesh['V'][vert2][0])
        dy = (Mesh['V'][vert1][1]-Mesh['V'][vert2][1])
        top[i][1] = np.sqrt(dx**2+dy**2)
        nx = -dy/top[i][1]
        ny = dx/top[i][1]
        dot = U[1][elem] * nx + U[2][elem] * ny
        u_b = (U[1][elem] - dot * nx) / U[0][elem]
        u_v = (U[2][elem] - dot * ny) / U[0][elem]
        q2 = u_b**2 + u_v**2
        p = (gamma-1)*(U[3][elem] - U[0][elem]/2*q2)
        top[i][2] = (p-pout)/qout
    # loop over bottom
    for i in range(nbot):
        vert1 = Mesh['BE'][i+botstart][0]
        vert2 = Mesh['BE'][i+botstart][1]
        elem = Mesh['BE'][i+botstart][2]
        bot[i][0] = (Mesh['V'][vert1][0]+Mesh['V'][vert2][0])/2
        dx2 = (Mesh['V'][vert1][0]-Mesh['V'][vert2][0])**2
        dy2 = (Mesh['V'][vert1][1]-Mesh['V'][vert2][1])**2
        bot[i][1] = np.sqrt(dx2+dy2)
        nx = -dy2/bot[i][1]
        ny = dx2/bot[i][1]
        dot = U[1][elem] * nx + U[2][elem] * ny
        u_b = (U[1][elem] - dot * nx) / U[0][elem]
        u_v = (U[2][elem] - dot * ny) / U[0][elem]
        q2 = u_b**2 + u_v**2
        p = (gamma-1)*(U[3][elem] - U[0][elem]/2*q2)
        bot[i][2] = (p-pout)/qout
    top = sorted(top, key=lambda x: x[0])
    xtop, dstop, cptop = zip(*top)
    bot = sorted(bot, key=lambda x: x[0])
    xbot, dsbot, cpbot = zip(*bot)
    # calc distances along surface
    stop[0] = dstop[0]/2
    sbot[0] = dsbot[0]/2
    for i in range(1,ntop):
        stop[i] = stop[i-1]+(dstop[i-1]+dstop[i])/2
    for i in range(1,nbot):
        sbot[i] = sbot[i-1]+(dsbot[i-1]+dsbot[i])/2

    # Plot final solution
    plt.figure()
    plt.plot(stop/stop[-1],cptop)
    plt.plot(sbot/sbot[-1],cpbot)
    plt.xlabel('s/c')
    plt.ylabel('$C_p$')
    plt.title('Pressure Coefficient Along Blade Surface')
    plt.legend(['Top Surface','Bottom Surface'])
    plt.savefig(outfile, dpi=300)
    plt.close()

