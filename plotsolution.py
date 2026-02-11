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
def plotsolution(grifile,solutionfile,sname):
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
        S = p/(U[0]**gamma);
    print(S)
    S = S/max(S)

    # Plot final solution
    tris = []
    for i in range(N):
        nodes = Mesh['V'][Mesh['E'][i]]
        tri = patches.Polygon(nodes)
        tris.append(tri)
    pc = col.PatchCollection(tris,cmap='jet')
    pc.set_array(S)
    fig, ax = plt.subplots()
    ax.add_collection(pc)
    plt.colorbar(pc)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(sname)
    plt.axis('equal')
    plt.show()

