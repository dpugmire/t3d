
import adios2
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import vtk, sys, math

def example2D(x,y,z) :
    print('create a rectilinear grid: X,Y')
    X, Y = np.meshgrid(x, y)
    print('xy= ', x.shape, y.shape)
    # Define the function for Z
    Z = np.sin(np.sqrt(X**2 + Y**2))
    print('XYZ=', X.shape, Y.shape, Z.shape)
    # Create a plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot the surface
    ax.plot_surface(X, Y, Z, cmap='viridis')
    plt.show()

def example3D(x,y,z) :
    # Create the 3D rectilinear grid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    print('XYZ=', X.shape, Y.shape, Z.shape)
    
    # Flatten the arrays to pass to scatter plot
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = Z.flatten()
    print('XYZ_flat=', X_flat.shape, Y_flat.shape, Z_flat.shape)    

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the grid points
    ax.scatter(X_flat, Y_flat, Z_flat, c='r', marker='o')

    # Set labels
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    # Show plot
    plt.show()

def createxyz(valMin=-1, valMax=1, num=10) :
    x = np.linspace(valMin, valMax, num)
    y = np.linspace(valMin, valMax, num)
    z = np.linspace(valMin, valMax, num)
    return (x,y,z)

def readADIOS(fname) :
    f = adios2.FileReader(fname)
    return f

def readNetcdf(fname) :
    ds = Dataset(fname, mode='r')
    return ds

def createVTK(ds, Ntheta=10, Nzeta=10) :
    theta_zero_mid = False
    zeta_zero_mid = False
    #Nzeta= 6 #100
    #Ntheta=20 #100
    s_idx=50
    s_idx = 199
    full_torus = True

    def get(f, key):
        return f.variables[key][:]

    xm = get(ds,'xm')
    xn = get(ds, 'xn')
    #rmnc,zmns = R,Z  (see: https://princetonuniversity.github.io/STELLOPT/VMEC.html)
    rmnc = get(ds, 'rmnc')
    zmns = get(ds, 'zmns')
    lmns = get(ds, 'lmns')
    phi = get(ds, 'phi')
    nfp = get(ds, 'nfp')

    if theta_zero_mid : tax = np.linspace(-np.pi, np.pi, Ntheta)
    else : tax = np.linspace(0, np.pi*2, Ntheta)
    z0 = np.pi/nfp
    if zeta_zero_mid : zax = np.linspace(-z0, z0, Nzeta)
    else : zax = np.linspace(0, 2*z0, Nzeta)

    varZeta = []
    varTheta = []
    # compute m and n modes
    print('DRP: m nodes= ', xm.shape)
    print('DRP: n nodes= ', xn.shape)
    cos = []
    sin = []
    for zeta in zax:
        for theta in tax:
            x = xm*theta - xn*zeta
            cos.append(np.cos(x))
            sin.append(np.sin(x))
            varZeta.append(zeta)
            varTheta.append(theta)
    print('theta/Zeta= ', len(varTheta), len(varZeta))
    print('     ', min(varTheta), max(varTheta), min(varZeta), max(varZeta))

    # sum of Fourier amplitudes
    print('rmnc/zmns/lmns: ', rmnc.shape, zmns.shape, lmns.shape)
    print('        min/max= : ', (rmnc.min(), rmnc.max()), (zmns.min(), zmns.max()), (lmns.min(), lmns.max()))

    ## calling numpy.sum
    print('numpy.sum: ')
    print('  rmnc[s_idx]=', rmnc[s_idx].shape)
    print('  cos=', cos.shape)
    R = np.sum(rmnc[s_idx] * cos, axis=-1)
    Z = np.sum(zmns[s_idx] * sin, axis=-1)
    L = np.sum(lmns[s_idx] * sin, axis=-1)
    print('Fourier amp: R,Z,L= ', R.shape, Z.shape, L.shape)

    # go to R-Phi-Z
    phi = []
    for zeta in zax:
        #np.ones_like: copy tax and fill it with 1.
        phi.append(zeta*np.ones_like(tax))
        R2 = np.reshape(R, (Nzeta, Ntheta))
        Z2 = np.reshape(Z, (Nzeta, Ntheta))
        L2 = np.reshape(L, (Nzeta, Ntheta))

    varR,varZ,varL= [],[],[]
    if full_torus:
        # apply NFP symmetry
        phi_n = []
        R_n = []
        Z_n = []
        L_n = []
        print('DRP: nfp= ', nfp)
        print(' zn: ', 0, '2pi, nfp= ', nfp)
        #endpoint=False means don't include 2pi
        zn = np.linspace(0, np.pi*2, nfp, endpoint=False)
        for z in zn:
            phi_n.append(np.array(phi)+z)
            R_n.append(R2)
            Z_n.append(Z2)
            L_n.append(L2)
            #print('  z: ', z, len(R2))
            varR.append(R2)
            varZ.append(Z2)
            varL.append(L2)

        #size: R_n 5
        print('R_n: ', len(R_n), len(R_n[0][0]))

        phi = np.concatenate(phi_n, axis=0)
        R2 = np.concatenate(R_n, axis=0)
        Z2 = np.concatenate(Z_n, axis=0)
        L2 = np.concatenate(L_n, axis=0)
        print('dRP: ', R2.shape, phi.shape)

    print('DRP: ', len(varR))
    # convert to xyz
    #print('DRP: compute X2,Y2')
    #sizes: (50,10)
    print('CONV to XYZ R2= ', R2.shape, 'phi= ', phi.shape)
    X2 = R2*np.cos(phi)
    Y2 = R2*np.sin(phi)
    #self.Lambda = L2

    varR = R2.flatten()
    varZ = Z2.flatten()
    varL = L2.flatten()
    print('R,Z,L, RET: ', varR.shape, varZ.shape, varL.shape)
    return X2, Y2, Z2, varR, varZ, varL


def readVarsWORKS(f, Ntheta=10, Nzeta=10) :
    theta_zero_mid = False
    zeta_zero_mid = False
    s_idx=50
    full_torus = True

    xm = f.read('xm')
    xn = f.read('xn')

    #rmnc,zmns = R,Z  (see: https://princetonuniversity.github.io/STELLOPT/VMEC.html)
    rmnc = f.read('rmnc')
    zmns = f.read('zmns')
    lmns = f.read('lmns')
    phi =  f.read('phi')
    nfp =  f.read('nfp')

    if theta_zero_mid : tax = np.linspace(-np.pi, np.pi, Ntheta)
    else : tax = np.linspace(0, np.pi*2, Ntheta)
    z0 = np.pi/nfp
    if zeta_zero_mid : zax = np.linspace(-z0, z0, Nzeta)
    else : zax = np.linspace(0, 2*z0, Nzeta)

    varZeta = []
    varTheta = []
    # compute m and n modes
    print('DRP: m nodes= ', xm.shape)
    print('DRP: n nodes= ', xn.shape)
    cos = []
    sin = []
    for zeta in zax:
        for theta in tax:
            ## xm.shape = 288, x.shape = 288
            x = xm*theta - xn*zeta
            cos.append(np.cos(x))
            sin.append(np.sin(x))
            varZeta.append(zeta)
            varTheta.append(theta)
    print('theta/Zeta= ', len(varTheta), len(varZeta))
    print('     ', min(varTheta), max(varTheta), min(varZeta), max(varZeta))
    print('    cos.len= ', len(cos))

    # sum of Fourier amplitudes
    print('rmnc/zmns/lmns: ', rmnc.shape, zmns.shape, lmns.shape)
    print('        min/max= : ', (rmnc.min(), rmnc.max()), (zmns.min(), zmns.max()), (lmns.min(), lmns.max()))

    print('  xm= ', xm.shape)
    print('numpy.sum: ')
    print('  rmnc[s_idx]=', rmnc[s_idx].shape)
    print('  cos=', len(cos))
    tmp = rmnc[s_idx]*cos
    print('   tmp= ', tmp.shape)
    # axis=-1 means to sum along the last dim.
    # rmnc[s_idx].shape = (288,)
    # cos.shape = 10000
    # (rmnc[s_idx] * cos).shape = (10000, 288)
    # R.shape = 10000 -- sum along the 288 values.
    R = np.sum(rmnc[s_idx] * cos, axis=-1)
    Z = np.sum(zmns[s_idx] * sin, axis=-1)
    L = np.sum(lmns[s_idx] * sin, axis=-1)
    print('Fourier amp: R,Z,L= ', R.shape, Z.shape, L.shape)

    # go to R-Phi-Z
    phi = []
    for zeta in zax:
        #np.ones_like: copy tax and fill it with 1.
        phi.append(zeta*np.ones_like(tax))
        R2 = np.reshape(R, (Nzeta, Ntheta))
        Z2 = np.reshape(Z, (Nzeta, Ntheta))
        L2 = np.reshape(L, (Nzeta, Ntheta))
    #r2,z2,l2 make it 2D again (nZeta, nTheta)
    print('R2,Z2,L2: ', R2.shape, Z2.shape, L2.shape)

    varR,varZ,varL= [],[],[]
    if full_torus:
        # apply NFP symmetry
        phi_n = []
        R_n = []
        Z_n = []
        L_n = []
        print('DRP: nfp= ', nfp)
        #endpoint=False means don't include 2pi
        zn = np.linspace(0, np.pi*2, nfp, endpoint=False)
        print(' zn: ', 0, '2pi, nfp= ', nfp, ' zn.shape= ', zn.shape)
        
        for z in zn:
            phi_n.append(np.array(phi)+z)
            R_n.append(R2)
            Z_n.append(Z2)
            L_n.append(L2)
            #print('  z: ', z, len(R2))
            varR.append(R2)
            varZ.append(Z2)
            varL.append(L2)

        #size: R_n 5
        print('R_n: ', len(R_n), len(R_n[0][0]))

        phi = np.concatenate(phi_n, axis=0)
        R2 = np.concatenate(R_n, axis=0)
        Z2 = np.concatenate(Z_n, axis=0)
        L2 = np.concatenate(L_n, axis=0)
        print('dRP: ', R2.shape, phi.shape)

    print('DRP: ', len(varR))
    # convert to xyz
    #print('DRP: compute X2,Y2')
    #sizes: (50,10)
    print('CONV to XYZ R2= ', R2.shape, 'phi= ', phi.shape)
    X2 = R2*np.cos(phi)
    Y2 = R2*np.sin(phi)
    #self.Lambda = L2
    print('  X2,Y2.shape= ', X2.shape, Y2.shape)

    varR = R2.flatten()
    varZ = Z2.flatten()
    varL = L2.flatten()
    print('R,Z,L, RET: ', varR.shape, varZ.shape, varL.shape)
    return X2, Y2, Z2, varR, varZ, varL

def readVars2(f, Ntheta=10, Nzeta=10) :
    theta_zero_mid = False
    zeta_zero_mid = False
    s_idx=50
    full_torus = True

    xm = f.read('xm')
    xn = f.read('xn')

    #rmnc,zmns = R,Z  (see: https://princetonuniversity.github.io/STELLOPT/VMEC.html)
    rmnc = f.read('rmnc')
    zmns = f.read('zmns')
    lmns = f.read('lmns')
    phi =  f.read('phi')
    nfp =  f.read('nfp')

    if theta_zero_mid : tax = np.linspace(-np.pi, np.pi, Ntheta)
    else : tax = np.linspace(0, np.pi*2, Ntheta)
    z0 = np.pi/nfp
    if zeta_zero_mid : zax = np.linspace(-z0, z0, Nzeta)
    else : zax = np.linspace(0, 2*z0, Nzeta)

    varZeta = []
    varTheta = []
    # compute m and n modes
    print('DRP: m nodes= ', xm.shape)
    print('DRP: n nodes= ', xn.shape)
    cos = []
    sin = []
    for zeta in zax:
        for theta in tax:
            ## xm.shape = 288, x.shape = 288
            x = xm*theta - xn*zeta
            cos.append(np.cos(x))
            sin.append(np.sin(x))
            varZeta.append(zeta)
            varTheta.append(theta)
    print('theta/Zeta= ', len(varTheta), len(varZeta))
    print('     ', min(varTheta), max(varTheta), min(varZeta), max(varZeta))
    print('    cos.len= ', len(cos))

    # sum of Fourier amplitudes
    print('rmnc/zmns/lmns: ', rmnc.shape, zmns.shape, lmns.shape)
    print('        min/max= : ', (rmnc.min(), rmnc.max()), (zmns.min(), zmns.max()), (lmns.min(), lmns.max()))

    print('  xm= ', xm.shape)
    print('numpy.sum: ')
    print('  rmnc[s_idx]=', rmnc[s_idx].shape)
    print('  cos=', len(cos))
    tmp = rmnc[s_idx]*cos
    print('   tmp= ', tmp.shape)
    # axis=-1 means to sum along the last dim.
    # rmnc[s_idx].shape = (288,)
    # cos.shape = 10000
    # (rmnc[s_idx] * cos).shape = (10000, 288)
    # R.shape = 10000 -- sum along the 288 values.
    R = np.sum(rmnc[s_idx] * cos, axis=-1)
    Z = np.sum(zmns[s_idx] * sin, axis=-1)
    L = np.sum(lmns[s_idx] * sin, axis=-1)
    print('Fourier amp: R,Z,L= ', R.shape, Z.shape, L.shape)

    # go to R-Phi-Z
    phi = []
    for zeta in zax:
        #np.ones_like: copy tax and fill it with 1.
        phi.append(zeta*np.ones_like(tax))
        R2 = np.reshape(R, (Nzeta, Ntheta))
        Z2 = np.reshape(Z, (Nzeta, Ntheta))
        L2 = np.reshape(L, (Nzeta, Ntheta))
    #r2,z2,l2 make it 2D again (nZeta, nTheta)
    print('R2,Z2,L2: ', R2.shape, Z2.shape, L2.shape)

    varR,varZ,varL= [],[],[]
    if full_torus:
        # apply NFP symmetry
        phi_n = []
        R_n = []
        Z_n = []
        L_n = []
        print('DRP: nfp= ', nfp)
        #endpoint=False means don't include 2pi
        zn = np.linspace(0, np.pi*2, nfp, endpoint=False)
        print(' zn: ', 0, '2pi, nfp= ', nfp, ' zn.shape= ', zn.shape)

        for z in zn:
            phi_n.append(np.array(phi)+z)
            R_n.append(R2)
            Z_n.append(Z2)
            L_n.append(L2)
            #print('  z: ', z, len(R2))
            varR.append(R2)
            varZ.append(Z2)
            varL.append(L2)

        #size: R_n 5
        print('R_n: ', len(R_n), len(R_n[0][0]))

        phi = np.concatenate(phi_n, axis=0)
        R2 = np.concatenate(R_n, axis=0)
        Z2 = np.concatenate(Z_n, axis=0)
        L2 = np.concatenate(L_n, axis=0)
        print('dRP: ', R2.shape, phi.shape)

    print('DRP: ', len(varR))
    # convert to xyz
    #print('DRP: compute X2,Y2')
    #sizes: (50,10)
    print('CONV to XYZ R2= ', R2.shape, 'phi= ', phi.shape)
    X2 = R2*np.cos(phi)
    Y2 = R2*np.sin(phi)
    #self.Lambda = L2
    print('  X2,Y2.shape= ', X2.shape, Y2.shape)

    varR = R2.flatten()
    varZ = Z2.flatten()
    varL = L2.flatten()
    print('R,Z,L, RET: ', varR.shape, varZ.shape, varL.shape)
    return varL

    return X2, Y2, Z2, varR, varZ, varL

def DotIt(R, phi, doCos) :
    nx = R.shape[0]
    ny = R.shape[1]
    result = []
    for i in range(nx) :
        tmp = []
        for j in range(ny) :
            if doCos : tmp.append(R[i,j] * np.cos(phi[i,j]))
            else     : tmp.append(R[i,j] * np.sin(phi[i,j]))            
        result.append(tmp)
    return np.array(result)
                    
            
            

def readVarsSIMP(f, Ntheta=10, Nzeta=10, s_idx=50) :
    theta_zero_mid = False
    zeta_zero_mid = False
    full_torus = True

    xm = f.read('xm')
    xn = f.read('xn')

    #rmnc,zmns = R,Z  (see: https://princetonuniversity.github.io/STELLOPT/VMEC.html)
    rmnc = f.read('rmnc')
    zmns = f.read('zmns')
    lmns = f.read('lmns')
    phi =  f.read('phi')
    nfp =  f.read('nfp')

    if theta_zero_mid : tax = np.linspace(-np.pi, np.pi, Ntheta)
    else : tax = np.linspace(0, np.pi*2, Ntheta)
    z0 = np.pi/nfp
    if zeta_zero_mid : zax = np.linspace(-z0, z0, Nzeta)
    else : zax = np.linspace(0, 2*z0, Nzeta)
    print('ntz= ', Ntheta, Nzeta)
    print('tax= ', tax)
    print('zax= ', zax)

    varZeta = []
    varTheta = []
    # compute m and n modes
    print('DRP: m nodes= ', xm.shape)
    print('DRP: n nodes= ', xn.shape)
    cos = []
    sin = []
    for zeta in zax:
        for theta in tax:
            ## xm.shape = 288, x.shape = 288
            x = xm*theta - xn*zeta
            cos.append(np.cos(x))
            sin.append(np.sin(x))
            varZeta.append(zeta)
            varTheta.append(theta)
    print('theta/Zeta= ', len(varTheta), len(varZeta))
    print('     ', min(varTheta), max(varTheta), min(varZeta), max(varZeta))
    print('    cos.len= ', len(cos))
    
    # theta/zeta are 10000, cos= 10000

    # sum of Fourier amplitudes
    print('rmnc/zmns/lmns: ', rmnc.shape, zmns.shape, lmns.shape)
    print('        min/max= : ', (rmnc.min(), rmnc.max()), (zmns.min(), zmns.max()), (lmns.min(), lmns.max()))

    print('  xm= ', xm.shape)
    print('numpy.sum: ')
    print('  rmnc[s_idx]=', rmnc[s_idx].shape)
    print('  cos=', len(cos))
    tmp = rmnc[s_idx]*cos
    # tmp.shape = (10,000, 288)
    # tmp.flatten() = (2,880,0000)
    print('   tmp= ', tmp.shape)
    tmp = tmp.flatten()

    R_, Z_, L_ = ([],[],[])
    N = Ntheta*Nzeta
    n = xm.shape[0]
    for i in range(N) :
        r,z,l = (0,0,0)
        for j in range(n) :
            #r = r + tmp[i*n + j]
            r = r + rmnc[s_idx][j] * cos[i][j]
            z = z + zmns[s_idx][j] * sin[i][j]
            l = l + lmns[s_idx][j] * sin[i][j]
        R_.append(r)
        Z_.append(z)
        L_.append(l)
    print('RZL_= ', np.array(R_).shape)

    # axis=-1 means to sum along the last dim.
    # rmnc[s_idx].shape = (288,)
    # cos.shape = 10000
    # (rmnc[s_idx] * cos).shape = (10000, 288)
    # R.shape = 10000 -- sum along the 288 values.
    #R = np.sum(rmnc[s_idx] * cos, axis=-1)
    #Z = np.sum(zmns[s_idx] * sin, axis=-1)
    #L = np.sum(lmns[s_idx] * sin, axis=-1)
    R = np.array(R_)
    Z = np.array(Z_)
    L = np.array(L_)
    print('Fourier amp: R,Z,L= ', R.shape, Z.shape, L.shape)
    #print(R[34])

    # go to R-Phi-Z
    phiVec = []
    for zeta in zax:
        #np.ones_like: copy tax and fill it with 1.
        phiVec.append(zeta*np.ones_like(tax))
    #phiVec.shape = nZeta x nTheta
    print('phiVec.shape= ', np.array(phiVec).shape)
    #print('phiVec[1,3]/[5,3]=', phiVec[1][3], phiVec[5][3])
    #c++ good

    #R is 10000
    print('R: ', R.shape)

    R2 = np.reshape(R, (Nzeta, Ntheta))
    print('R2.shape= ', R2.shape)
    Z2 = np.reshape(Z, (Nzeta, Ntheta))
    L2 = np.reshape(L, (Nzeta, Ntheta))
    #r2,z2,l2 make it 2D again (nZeta, nTheta)
    #print('R2/Z2/L2(3,4) = ', R2[3][4], Z2[3][4], L2[3][4])
    #c++ good

    ##### R2,Z2,L2.shape = (10,10)
    ##### phiVec.shape = (10,10)
    print('R2,Z2,L2: ', R2.shape, Z2.shape, L2.shape)
    print('phi.shape: ', np.array(phiVec).shape)

    varR,varZ,varL= [],[],[]
    if full_torus:
        # apply NFP symmetry
        phi_n = []
        R_n = []
        Z_n = []
        L_n = []
        phi_n_, R_n_, Z_n_, L_n_ = ([],[],[],[])
        
        print('DRP: nfp= ', nfp)
        #endpoint=False means don't include 2pi
        zn = np.linspace(0, np.pi*2, nfp, endpoint=False)
        print(' zn: ', 0, '2pi, nfp= ', nfp, ' zn.shape= ', zn.shape)

# OLD CODE
        for z in zn:
            print('z= ', z)
            phi_n.append(np.array(phiVec)+z)
            R_n.append(R2)
            Z_n.append(Z2)
            L_n.append(L2)
            #print('  z: ', z, len(R2))
            varR.append(R2)
            varZ.append(Z2)
            varL.append(L2)
        #phi_n.shape = zn x nZeta x nTheta
        #R_n.shape = zn * nZeta x nTheta
        print('phi_n: ', np.array(phi_n).shape) #5,10,10
        print('R_n: ', np.array(R_n).shape) # 5,10,10
        #print('vals(3,4,4): ', phi_n[3][4][4], R_n[3][4][4], Z_n[3][4][4], L_n[3][4][4])
        #c++ good

        ##new
#        print('R_.shape=', len(R_), len(phi_), Nzeta, Ntheta)
#        for z in zn:
#            for i in range(Nzeta) :
#                for j in range(Ntheta) :
#                    phi_n_.append(phi_[i*Ntheta + j]+z)
#                    R_n_.append(R_[i*Ntheta +j])
#                    Z_n_.append(Z_[i*Ntheta +j])
#                    L_n_.append(L_[i*Ntheta +j])

        #size: R_n 5,Nzeta,Ntheta --> 5*Nzeta, ntheta
        phiVec, R2, Z2, L2  = [],[],[],[]

        print('***** concat: ', np.array(phi_n).shape, np.array(R_n).shape)
        phiVec = np.concatenate(phi_n, axis=0)
        R2 = np.concatenate(R_n, axis=0)
        Z2 = np.concatenate(Z_n, axis=0)
        L2 = np.concatenate(L_n, axis=0)
        print('phiVec.shape= ', phiVec.shape) #50,10
        print('R2.shape= ', R2.shape)   #50,10
        print('ZN: ', zn)
        #print('vals(33,4)= ', phiVec[33][4], R2[33][4], Z2[33][4], L2[33][4])
        #c++ good

        print('phi_n: ->np ', np.array(phi_n).shape)
        print('dRP: ', np.array(phiVec).shape)

    print('After full torus:')
    print('R2,Z2,L2: ', R2.shape, Z2.shape, L2.shape)
    print('phiVec.shape: ', np.array(phi).shape)
    print(R2)

    print('DRP: ', len(varR))
    # convert to xyz
    #print('DRP: compute X2,Y2')
    #sizes: (50,10)
    print('CONV to XYZ R2= ', R2.shape, 'phi= ', np.array(phiVec).shape)
    #X2 = R2*np.cos(phi)
    #Y2 = R2*np.sin(phi)
    #print('0: X2/Y2/Z2: ', X2.shape, Y2.shape, Z2.shape)    
    X2 = DotIt(R2, phiVec, True)
    Y2 = DotIt(R2, phiVec, False)
    print('1: X2/Y2/Z2: ', X2.shape, Y2.shape, Z2.shape)
    #print('X2/Y2/Z2: ', X2[33][4], Y2[33][4], Z2[33][4])
    #c++ good

#    X2,Y2 = ([],[])
#    for i in range(len(R2)) :
#        tmpx,tmpy = ([],[])
#        for j in range(len(R2[0])) :
#            #print('i,j= ', (i,j), phi.shape)
#            tmpx.append(R2[i,j]*math.cos(phi[i,j]))
#            tmpy.append(R2[i,j]*math.sin(phi[i,j]))
#        X2.append(tmpx)
#        Y2.append(tmpy)
    
    #self.Lambda = L2
    print('  X2,Y2.shape= ', len(X2), len(X2[0]))

    varR = R2.flatten()
    varZ = Z2.flatten()
    varL = L2.flatten()
    print('R,Z,L, RET: ', varR.shape, varZ.shape, varL.shape)
    print(' --- ', X2.shape, Y2.shape, Z2.shape)
    print(' ---- ', varR.shape, varZ.shape, varL.shape)
    
    return X2, Y2, Z2, varR, varZ, varL, nfp


def dumpDS(ds, fname) :
    writer = vtk.vtkDataSetWriter()
#    geomFilter = vtk.vtkGeometryFilter()
#    geomFilter.SetInputData(ds)
#    geomFilter.Update()
#    ds = geomFilter.GetOutput()
#    writer = vtk.vtkPolyDataWriter()
    writer.SetFileVersion(42)
    writer.SetFileName(fname)
    writer.SetInputData(ds)
    writer.Write()

    fout = open('/tmp/points_duplicate.txt', 'w')
    pts = ds.GetPoints()
    np = pts.GetNumberOfPoints();
    varL = ds.GetPointData().GetArray('Lambda')

    print('np=', np)
    fout.write('varI, varX, varY, varZ, varL\n')
    for i in range(np) :
        pt = pts.GetPoint(i)
        lVal = varL.GetValue(i)
        fout.write('%d, %.4f, %.4f, %.4f, %.4f\n' % (i, pt[0], pt[1], pt[2], lVal))
    fout.flush()
    fout.close()
        

def getdims(x, idx=0) :
    if not isinstance(x, list) :
        return None
    if len(x) > 0 and isinstance(x[0],list) :
        return [len(x), getdims(x[idx], idx+1)]
    return len(x)

def createGrid(ntheta, nzeta, nfp, X,Y,Z, vars, varNames, fname) :

    print('***************************************')
    print('bummy: ', getdims(X))
    print('createGrid: ', len(X), len(Y), len(Z))
    print(' ntheta/nzeta= ', ntheta, nzeta)
    print(' nfp= ', nfp)
    #print('cmp: ', X[0], Y[0], Z[0])
    #for i in range(50) :
    #    print('   ',i, X[i], Y[i], Z[i], 'diff=', X[i]-X[0])

    dims = getdims(X)
    n = dims[0]
    m = dims[1]

    grid = vtk.vtkUnstructuredGrid()
    pts = vtk.vtkPoints()
    grid.SetPoints(pts)
    idx = 0
    for j in range(m) :
        for i in range(n) :
            pt = (X[i][j], Y[i][j], Z[i][j])
            pts.InsertNextPoint(pt)
            grid.InsertNextCell(vtk.VTK_VERTEX, 1, [idx])
            idx = idx+1
        #print('Pt_',i, pt)

    ptData = grid.GetPointData()
    varIdx = vtk.vtkFloatArray()
    varPln = vtk.vtkFloatArray()
    varTheta = vtk.vtkFloatArray()
    varLambda = vtk.vtkFloatArray()
    varIdx.SetName('idx')
    varPln.SetName('plane')
    varTheta.SetName('theta')
    varLambda.SetName('Lambda')
    idx = 0
    for j in range(m) :
        for i in range(n) :
            varIdx.InsertNextValue(float(idx))
            planeIdx = j #int(i / ntheta)
            varPln.InsertNextValue(float(i))
            varTheta.InsertNextValue(float(j))
            varLambda.InsertNextValue(float(j))
            idx = idx+1
    ptData.AddArray(varIdx)
    ptData.AddArray(varPln)
    ptData.AddArray(varTheta)
    ptData.AddArray(varLambda)    

    nv = len(vars)
    for i in range(nv) :
        ptData = grid.GetPointData()
        var = vtk.vtkFloatArray()
        var.SetName(varNames[i])
        ptData.AddArray(var)
        for j in range(n) :
            var.InsertNextValue(vars[i][j])

    dumpDS(grid, fname)

def createGrids(ntheta, nzeta, nfp, X,Y,Z,L, vars, varNames, fname, srfSelect=None) :

    print('***************************************')
    print('createGrid: ', len(X), len(Y), len(Z))
    print(' ntheta/nzeta= ', ntheta, nzeta)
    print(' nfp= ', nfp)
    print(' X.shape= ', X[0].shape)

    dims = X[0].shape
    #dims = getdims(X[0])

    n = dims[0]
    m = dims[1]

    numSrfs = len(X)

    srfIndex = []
    if srfSelect == None :
        srfIndex = range(numSrfs)
    else : srfIndex = srfSelect

    print('numSrfs: ', numSrfs, ' idx= ', srfIndex)

    grid = vtk.vtkUnstructuredGrid()
    pts = vtk.vtkPoints()
    grid.SetPoints(pts)
    print('surfaces= ', srfSelect, 'numsrfs= ', numSrfs)

    idx = 0
    for s in range(numSrfs) :
        for j in range(m) :
            for i in range(n) :
                pt = (X[s][i][j], Y[s][i][j], Z[s][i][j])
                pts.InsertNextPoint(pt)
                grid.InsertNextCell(vtk.VTK_VERTEX, 1, [idx])
                idx = idx+1
        #print('Pt_',i, pt)

    ptData = grid.GetPointData()
    varIdx = vtk.vtkFloatArray()
    varPln = vtk.vtkFloatArray()
    varTheta = vtk.vtkFloatArray()
    varSrf = vtk.vtkFloatArray()
    varLam = vtk.vtkFloatArray()    
    varIdx.SetName('idx')
    varPln.SetName('plane')
    varTheta.SetName('theta')
    varSrf.SetName('surface')
    varLam.SetName('Lambda')    
    for s in range(numSrfs) :
        idx = 0
        for j in range(m) :
            for i in range(n) :
                varIdx.InsertNextValue(float(idx))
                planeIdx = j #int(i / ntheta)
                varPln.InsertNextValue(float(i))
                varTheta.InsertNextValue(float(j))
                varSrf.InsertNextValue(float(srfIndex[s]))
                varLam.InsertNextValue(L[s][idx])
                idx = idx+1

    ptData.AddArray(varIdx)
    ptData.AddArray(varPln)
    ptData.AddArray(varTheta)
    ptData.AddArray(varSrf)
    ptData.AddArray(varLam)    

    #nv = len(vars)
    #for i in range(nv) :
    #    ptData = grid.GetPointData()
    #    var = vtk.vtkFloatArray()
    #    var.SetName(varNames[i])
    #    ptData.AddArray(var)
    #    for j in range(n) :
    #        var.InsertNextValue(vars[i][j])

    dumpDS(grid, fname)



def createSrf(X,Y,Z,L, nTheta, nZeta, nfp) :
    print('**************************************')
    print('** createSrf')
    srf = vtk.vtkUnstructuredGrid()
    nPlanes = (nZeta) * nfp
    ptsPerPlane = nTheta

    print('nPlanes= ', nPlanes)
    print('ptsPerPlane= ', ptsPerPlane)

    np = nPlanes*ptsPerPlane
    print('np= ', np, 'X.len= ', len(X))

    points = vtk.vtkPoints()
    srf.SetPoints(points)
    planeIdxVar = vtk.vtkFloatArray()
    planeIdxVar.SetName('planeIdx')
    srf.GetPointData().AddArray(planeIdxVar)
    lambdaVar = vtk.vtkFloatArray()
    lambdaVar.SetName('Lambda')
    srf.GetPointData().AddArray(lambdaVar)

    idx = 0
    for p in range(nPlanes) :
        for i in range(ptsPerPlane) :
            pt = (X[p][i], Y[p][i], Z[p][i])
            points.InsertNextPoint(pt)
            planeIdxVar.InsertNextValue(float(p))
            lambdaVar.InsertNextValue(L[idx])
            print('idx= ', idx, pt)
            idx = idx+1

    idx = 0
    print('nPlanes= ', nPlanes)
    if False :
        for p in range(nPlanes-1) :
            p0 = p*ptsPerPlane
            if p == nPlanes-1 :
                p1 = 0
            else:
                p1 = (p+1)*ptsPerPlane

            ptIds = [p0,p1]
            srf.InsertNextCell(vtk.VTK_LINE, 2, ptIds)
        return srf

    zoneId = 0
    zID = vtk.vtkFloatArray()
    zID.SetName('ZoneId')
    srf.GetCellData().AddArray(zID)

    for p in range(nPlanes) :
        offset0 = p*ptsPerPlane
        offset1 = (p+1)*ptsPerPlane
        #if p == 4 : break
        if p == nPlanes-1 : offset1 = 0
        for i in range(ptsPerPlane-1) :
            #if zoneId > 2 : break
            p0 = offset0 + i
            p1 = p0+1
            p2 = offset1 + i
            p3 = p2+1
            ptIds = [p0,p3,p1]
            #print('p,i: ', (p,i), p0,p1,p2,p3)
            srf.InsertNextCell(vtk.VTK_TRIANGLE, 3, ptIds)
            ptIds = [p0,p2,p3]
            srf.InsertNextCell(vtk.VTK_TRIANGLE, 3, ptIds)
            zID.InsertNextValue(zoneId)
            zID.InsertNextValue(zoneId+1)
            zoneId = zoneId+2

    # remove / merge redundant points
    cleaner = vtk.vtkStaticCleanUnstructuredGrid()
    cleaner.SetInputData(srf)
    cleaner.Update()
    srf = cleaner.GetOutput()
    return srf


def createSrfs(X,Y,Z,L, nTheta, nZeta, nfp, surfaces) :
    print('**************************************')
    print('** createSrf')
    nPlanes = (nZeta) * nfp
    ptsPerPlane = nTheta
    numSrfs = len(X)
    print('numSrfs= ', numSrfs)

    print('nPlanes= ', nPlanes)
    print('ptsPerPlane= ', ptsPerPlane)

    np = nPlanes*ptsPerPlane
    print('np= ', np, 'X.len= ', len(X))

    srf = vtk.vtkUnstructuredGrid()
    points = vtk.vtkPoints()
    srf.SetPoints(points)
    planeIdxVar = vtk.vtkFloatArray()
    planeIdxVar.SetName('planeIdx')
    srf.GetPointData().AddArray(planeIdxVar)
    lambdaVar = vtk.vtkFloatArray()
    lambdaVar.SetName('Lambda')
    ptIdxVar = vtk.vtkFloatArray()
    ptIdxVar.SetName('point_index')
    srf.GetPointData().AddArray(ptIdxVar)
    srf.GetPointData().AddArray(lambdaVar)
    surfaceVar = vtk.vtkFloatArray()
    surfaceVar.SetName('surface')
    srf.GetPointData().AddArray(surfaceVar)
    srfIdxVar = vtk.vtkFloatArray()
    srfIdxVar.SetName('surface_index')
    srf.GetPointData().AddArray(srfIdxVar)

    for s in range(numSrfs) :
        idx = 0
        for p in range(nPlanes) :
            for i in range(ptsPerPlane) :
                pt = (X[s][p][i], Y[s][p][i], Z[s][p][i])
                points.InsertNextPoint(pt)
                planeIdxVar.InsertNextValue(float(p))
                lambdaVar.InsertNextValue(L[s][idx])
                srfIdxVar.InsertNextValue(float(s))
                surfaceVar.InsertNextValue(float(surfaces[s]))
                ptIdxVar.InsertNextValue(idx)
                print('idx= ', idx, pt)
                idx = idx+1
    print('numpts= ', points.GetNumberOfPoints(), ':', ptsPerPlane)


    idx = 0
    print('nPlanes= ', nPlanes)
    if False :
        for p in range(nPlanes-1) :
            p0 = p*ptsPerPlane
            if p == nPlanes-1 :
                p1 = 0
            else:
                p1 = (p+1)*ptsPerPlane

            ptIds = [p0,p1]
            srf.InsertNextCell(vtk.VTK_LINE, 2, ptIds)
        return srf

    zoneId = 0
    zID = vtk.vtkFloatArray()
    zID.SetName('ZoneId')
    srf.GetCellData().AddArray(zID)

    doQuad = True
    srfOffset = 0
    for s in range(numSrfs) :
        for p in range(nPlanes) :
            offset0 = p*ptsPerPlane + srfOffset
            offset1 = (p+1)*ptsPerPlane + srfOffset
            if p == nPlanes-1 : offset1 = 0
            for i in range(ptsPerPlane-1) :
                p0 = offset0 + i
                p1 = p0+1
                p2 = offset1 + i
                p3 = p2+1

                if doQuad :
                    ptIds = [p0,p1,p3, p2]
                    srf.InsertNextCell(vtk.VTK_QUAD, 4, ptIds)
                    zID.InsertNextValue(zoneId)
                    zoneId = zoneId + 1
                else :
                    ptIds = [p0,p3,p1]
                    srf.InsertNextCell(vtk.VTK_TRIANGLE, 3, ptIds)
                    ptIds = [p0,p2,p3]
                    srf.InsertNextCell(vtk.VTK_TRIANGLE, 3, ptIds)
                    zID.InsertNextValue(zoneId)
                    zID.InsertNextValue(zoneId+1)
                    zoneId = zoneId+2
                    
        srfOffset = srfOffset + (nPlanes * ptsPerPlane)

    # remove / merge redundant points
    #cleaner = vtk.vtkStaticCleanUnstructuredGrid()
    #cleaner.SetInputData(srf)
    #cleaner.Update()
    #srf = cleaner.GetOutput()
    return srf


fname = './w7x-gx/wout_w7x.nc'
fnameAdios = './w7x-gx/wout_w7x.bp'
print(fname)
print(fnameAdios)
#ds = readNetcdf(fname)
f = readADIOS(fnameAdios)
ntheta = 10
nzeta = 20
ntheta = 20
nzeta = 50
## DRP
ntheta = 50
nzeta = 50

#X,Y,Z,vR,vZ,vL = createVTK(ds, ntheta, nzeta)

#X,Y,Z,vR,vZ,vL = readVars2(f, ntheta, nzeta)
#SIMP = readVarsSIMP(f, ntheta, nzeta)

def toNPArray(x) :
    if isinstance(x, list) :
        return np.array(x).flatten()
    else : return x.flatten()


# testing individual arrays.
if False :
    x0 = toNPArray(readVars2(f, ntheta, nzeta))
    x1 = toNPArray(readVarsSIMP(f, ntheta, nzeta))
    print('\n\n\n*******************************************************')
    print('****** RESULT: ', x0.shape, x1.shape)

    if x0.shape[0] != x1.shape[0] :

        print('shape not same!!!!')

    if np.allclose(x0, x1, atol=1e-8) :
        print(' SAME')
    else:     print('************************************* XXXXXXXXXXXXX NOT SAME')

    sys.exit()



#X,Y,Z,vR,vZ,vL,nfp = readVarsSIMP(f, ntheta, nzeta, 20)
#X = X.flatten()
#Y = Y.flatten()
#Z = Z.flatten()
#print('R=', len(vR))

##Read in multiple surfaces
X,Y,Z,vR,vZ,vL = ([],[],[],[],[],[])
nfp = -1
srfSelect = [151]
for sIdx in srfSelect :
    out = readVarsSIMP(f, ntheta, nzeta, sIdx)
    X.append(out[0])
    Y.append(out[1])
    Z.append(out[2])
    vR.append(out[3])
    vZ.append(out[4])
    vL.append(out[5])
    nfp = out[6]

print('LEN= ', len(X), X[0].shape)


createGrids(ntheta, nzeta, nfp, X,Y,Z,vL, [vR,vZ,vL], ['R','Z','L'], 'grid.vtk', srfSelect)
srf = createSrfs(X,Y,Z,vL,  ntheta, nzeta, nfp, srfSelect)
dumpDS(srf, 'srf_duplicate.vtk')
sys.exit()



createGrid(ntheta, nzeta, nfp, X,Y,Z, [vR, vZ, vL], ['R', 'Z', 'L'], 'grid.vtk')

srf = createSrf(X,Y,Z,vL,  ntheta, nzeta, 5)
dumpDS(srf, 'srf.vtk')
sys.exit()

if False :
    print('meow')
    _, ax = plt.subplots(subplot_kw={"projection": "3d"})
    _ = ax.plot_surface(X,Y,Z, color='c')
    ax.set_aspect('equal')
    plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
print('X: ', X.shape)
print('X: ', X.shape)
srf = ax.plot_trisurf(X,Y,Z)
ax.set_aspect('equal')
plt.show()



#x,y,z = createxyz()

#example2D(x,y,z)
#example3D(x,y,z)


