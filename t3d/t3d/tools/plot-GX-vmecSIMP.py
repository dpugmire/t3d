import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

from scipy.interpolate import griddata

from t3d.Geometry import VmecReader
from mayavi import mlab

'''
GX VMEC plotting scripts

T. Qian - 19 March 2024
'''

# GX files
file1 = "etg/t05-p0-r6-a0-0.out.nc"
file2 = "etg/t05-p0-r6-a1-0.out.nc"
file1 = "t05-p0-r6-a0-0.out.nc"
file2 = "t05-p0-r6-a1-0.out.nc"
file1 = "w7x-gx/w7x-gx-flux-tubes/t03-d0.2-p1-r6-j0.out.nc"
#t03: timestep
#d0.2: time
#p0: newton iteration number.
#r = radius Different flux tube. Usually 0-7
#j = perturbation usually 0,1 maybe 2. 0 = unperterubed.
#t10-p0-r3-1.GK.fluxes.bp "-1" = "-j1".
# "wout_w7x.nc" --vmec geometry


file2 = "w7x-gx/w7x-gx-flux-tubes/t03-d0.2-p1-r6-j1.out.nc"

s_idx = 0  # gx species index

# VMEC files
#path = "./w7x-case"
#geo = "eq_180920017_w7x_169_wout.nc"
#path = "/Users/dpn/proj/gx/3dPlot/tmp/t3d./w7x-gx"
#geo = "wout_w7x.nc"

## 10.28.24
#path = '../w7x-gx'
#geo = 'wout_w7x.nc'
#geo = "eq_180920017_w7x_169_wout.nc"
file1 = '../w7x-gx/t05-p0-r6-a0-0.out.nc'
file2 = '../w7x-gx/t05-p0-r6-a1-0.out.nc'

##11.1.24- use the same data as ADIOS
vmecFile = '/Users/dpn/proj/gx/3dPlot/t3d/NERSC/wout_w7x.nc'
file1 = '/Users/dpn/proj/gx/3dPlot/t3d/NERSC/t10-p0-r6-0.out.nc'
file2 = '/Users/dpn/proj/gx/3dPlot/t3d/NERSC/t10-p0-r6-1.out.nc'

N_THETA = 100
N_ZETA = 25

class FluxTube:
    '''
    Reads GX data

    getXYZ(vmec) exports points using wout
    '''
    def __init__(self,fname):
        data = Dataset(fname, mode='r')
        scale = data.groups['Geometry'].variables['theta_scale'][:]
        theta = data.groups['Grids'].variables['theta'][:]*scale
        iota = 1/data.groups['Geometry'].variables['q'][:]

        zeta_center = data.groups['Geometry'].variables['zeta_center'][:]
        alpha = -iota*zeta_center
        zeta = (theta - alpha)/iota
        if zeta.shape[0] > 64 :
            print('fname= ', fname)
            print('theta: ', theta.shape, theta.min(), theta.max())
            print('zeta: ', zeta.shape, zeta.min(), zeta.max())
            print('zeta_center ', zeta_center.shape, zeta_center.min(), zeta_center.max())
            shit()

        self.alpha = alpha
        self.iota = iota
        self.zeta = zeta
        self.theta = theta
        self.data = data

        # heat flux
        print('Read heat flux: s_idx= ', s_idx)
        time = data.groups['Grids'].variables['time'][:]
        print(' time.shape= ', time.shape)

        Qtz = data.groups['Diagnostics'].variables['HeatFlux_zst'][:,s_idx,:]
        print('  turbulent heat flux in gyroBohm units')
        print('Qtz.shape= ', Qtz.shape)
        Q_gx = np.mean(Qtz[int(len(time)/2):,:], axis=0)
        jacobian = data.groups['Geometry'].variables['jacobian'][:]
        grho = data.groups['Geometry'].variables['grho'][:]
        fluxDenom = np.sum(jacobian*grho)

        # to be interpolated
        self.Q = Q_gx*fluxDenom
        self.norm = jacobian*grho
        self.grho = grho
        self.Q_gx = Q_gx
        self.time = time
        print('self.Q.shape= ', self.Q.shape)
        print('self.Q_gx.shape= ', self.Q_gx.shape)

    def getXYZ(self,vmec, psiN=0.434, full_torus=True):

        zeta = self.zeta
        #alpha = self.alpha
        #iota = self.iota

        # get surface
        s = np.argmin(np.abs(vmec.s - psiN))
        #i = np.argmin(np.abs(vmec.iotaf - iota))

        thetaStar = self.theta
        N = len(zeta)
        theta = np.array([vmec.invertTheta(thetaStar[j], zeta=zeta[j],s_idx=s) for j in np.arange(N)])

        print('THETA', theta.shape, theta.min(), theta.max())
        # wrap around a single field period
        zeta_n = zeta % (2*np.pi/vmec.nfp)
        self.zeta_n = zeta_n
        self.theta_v = theta
        print('zeta:', zeta.shape, zeta.min(), zeta.max())
        print('zeta_n:', zeta_n.shape, zeta_n.min(), zeta_n.max())
        print('thetaStar', thetaStar.shape, thetaStar.min(), thetaStar.max())
        print('theta_v:', self.theta_v.shape, self.theta_v.min(), self.theta_v.max())

        X,Y,Z = (None, None, None)
        return X,Y,Z

def pad(data,zeta,theta, threshold=0.8):
    '''
    Use periodicity to add boundary data before interpolation
    '''

    # pad theta
    t_max = np.pi * threshold
    arg1 = np.argwhere(theta > t_max)[:,0]
    arg2 = np.argwhere(theta < -t_max)[:,0]
    t1 = np.concatenate([theta, theta[arg1]-2*np.pi, theta[arg2]+2*np.pi])
    z1 = np.concatenate([zeta, zeta[arg1], zeta[arg2]])
    Q1 = np.concatenate([data, data[arg1], data[arg2]])
    print('pad theta:')
    print('  theta/zeta.shape= ', theta.shape, zeta.shape)
    print('  arg1/arg2.shape', arg1.shape, arg2.shape)
    print('     t1/z1.shape= ', t1.shape, z1.shape)

    # pad zeta
    z_max = z0 * threshold
    arg1 = np.argwhere(z1 > z_max)[:,0]
    arg2 = np.argwhere(z1 < -z_max)[:,0]
    t2 = np.concatenate([t1, t1[arg1], t1[arg2]])
    z2 = np.concatenate([z1, z1[arg1]-2*z0, z1[arg2]+2*z0])
    Q2 = np.concatenate([Q1, Q1[arg1], Q1[arg2]])
    print('pad zeta:')
    print('  theta/zeta.shape= ', t1.shape, z1.shape)
    print('  arg1/arg2.shape', arg1.shape, arg2.shape)
    print('     t2/z2.shape= ', t2.shape, z2.shape)

    return Q2,z2,t2


# main function
#fname = f"{path}/{geo}"
fname = vmecFile
print('reading: ', fname)
#vmec = VmecReader(f"{path}/{geo}")
vmec = VmecReader(fname)
print('reading flux: ', file1, file2)
f1 = FluxTube(file1)
f2 = FluxTube(file2)

# DRP
# load GX data onto VMEC surfaces
f1.getXYZ(vmec,full_torus=False)
f2.getXYZ(vmec,full_torus=False)
print('f1/f2.Q_gx.shape= ', f1.Q_gx.shape, f2.Q_gx.shape)
print('f1.zeta_n.shape= ', f1.zeta_n.shape, f2.zeta_n.shape)
print('zeta_n0= ', f1.zeta_n.shape, f1.zeta_n.min(), f1.zeta_n.max())
print('zeta_n1= ', f2.zeta_n.shape, f2.zeta_n.min(), f2.zeta_n.max())
#shit()
print('f1/f2.theta_v.shape= ', f1.theta_v.shape, f2.theta_v.shape)

zeta = np.concatenate([f1.zeta_n, f2.zeta_n])
z0 = np.pi/vmec.nfp
zeta = (zeta + z0) % (2*z0) - z0
theta = np.concatenate([f1.theta_v, f2.theta_v])
print('f1.theta_v:', f1.theta_v.size)
print(f1.theta_v.min(), f1.theta_v.max())
#shit()

Q_gx = np.concatenate([f1.Q_gx,f2.Q_gx])  # only for plotting, not for computation
Q = np.concatenate([f1.Q,f2.Q])  # Q_gx * sum1D( norm ), norm = jacobian * grho
norm = np.concatenate([f1.norm,f2.norm])  # jacobian * grad rho
print('********************************')
print('Q1: ', f1.Q.shape, f1.Q.min(), f1.Q.max())
print('Q2: ', f2.Q.shape, f2.Q.min(), f2.Q.max())
print('Qgx1: ', f1.Q_gx.shape, f1.Q_gx.min(), f1.Q_gx.max())
print('Qgx2: ', f2.Q_gx.shape, f2.Q_gx.min(), f2.Q_gx.max())
print('norm1: ', f1.norm.shape, f1.norm.min(), f1.norm.max())
print('norm2: ', f2.norm.shape, f2.norm.min(), f2.norm.max())

## Q_gx.shape = 246.  find where z1 > z_max and z1 < -z_max (z_max = z0*threshold)
Q_gx2, z2, t2 = pad(Q_gx,zeta,theta)
Q2, z2, t2 = pad(Q,zeta,theta)
N2, z2, t2 = pad(norm,zeta,theta)
print('pad: Q_gx2: ', Q_gx2.shape, Q_gx2.min(), Q_gx2.max())
#print('Pad(): Q_gx/zeta/theta.shape= ', Q_gx.shape, zeta.shape, theta.shape)
#print('Q_gx2.shape= ', Q_gx2.shape)
#print('Q2.shape= ', Q2.shape)
#print('N2.shape= ', N2.shape)

#  interpolation grid
Ntheta = N_THETA
Nzeta = N_ZETA
tax = np.linspace(-np.pi,np.pi,Ntheta)
zax = np.linspace(-z0,z0,Nzeta)

Z,T = np.meshgrid(zax,tax)
#scipy.interpolate.griddata
#  griddata((2dpoints), values, (target points))
print('griddata: z2/t2.shape', z2.shape, t2.shape, )
print('   Q2.shape= ', Q2.shape)
print('    Z/T.shape= ', Z.shape, T.shape)
Qsamp = griddata((z2,t2), Q2, (Z,T), method='linear')
Nsamp = griddata((z2,t2), N2, (Z,T), method='linear')
print('Qsamp.shape= ', Qsamp.shape, Qsamp.min(), Qsamp.max())

area = 4*np.pi**2/vmec.nfp
dA = area / (Nzeta-1) / (Ntheta-1)

# integrate, but exclude the endpoint
N_int = np.sum(Nsamp[:-1,:-1]) * dA
Q_int = np.sum(Qsamp[:-1,:-1])/N_int * dA


# plot mesh on surf
#DRP
#Qsamp.T = 12,5 array.
qst = Qsamp.T
print('Q.T.shape= ', Qsamp.T.shape)
srf_idx = 43
print('VMEC(zeta,theta, nfp): ', Nzeta, Ntheta, vmec.nfp)
X,Y,Z = vmec.getSurfaceMesh(Nzeta=Nzeta, Ntheta=Ntheta,s_idx=srf_idx,zeta_zero_mid=True, theta_zero_mid=True, full_torus=True)
nfp = vmec.nfp

nfp = 1
if nfp == 1 :
    #grab first part of mesh...
    X = X[0:25, 0:100]
    Y = Y[0:25, 0:100]
    Z = Z[0:25, 0:100]

# apply stellarator symmetry to Q
zn = np.linspace(0,np.pi*2,vmec.nfp,endpoint=False)
Q_n = []
cnt = 0
for z in zn:
    if cnt == nfp :
        break
    Q_n.append(Qsamp.T)
    cnt = cnt+1

print('Qsamp.T: ', Qsamp.T.shape, Qsamp.T.min(), Qsamp.T.max())
Q_PLOT = np.concatenate(Q_n,axis=0) / N_int * area
print('Q.shape ', Q.shape)
q1d = Q_PLOT.flatten()
print('Q_plot: ', q1d.shape, q1d.min(), q1d.max())
#for i in range(0, q1d.shape[0], 210) :
#    print(i, ':', q1d[i])

idxArr = []
for k in range(nfp) :
    _idx = []
    for zi in range(Nzeta) :
        for ti in range(Ntheta) :
            _idx.append(zi)
    idxArr.append(_idx)

IDX_ARR = np.concatenate(idxArr)
IDX_ARR = IDX_ARR.reshape((Nzeta*nfp, Ntheta))
print('plotting.....')
print(IDX_ARR.shape, IDX_ARR.min(), IDX_ARR.max())
print(Q_PLOT.shape, Q_PLOT.min(), Q_PLOT.max())
print('XYZ:', X.shape, Y.shape, Z.shape)

fig = mlab.figure(bgcolor=(1,1,1), fgcolor=(0.,0.,0.), size=(1000,800))
cmap = 'hot'
cmap = 'viridis'
#mesh = mlab.mesh(X, Y, Z, scalars=Q_PLOT, colormap=cmap)
#mesh = mlab.mesh(X.transpose(), Y.transpose(), Z.transpose(), scalars=Q_PLOT.transpose(), colormap=cmap)
mesh = mlab.mesh(X, Y, Z, scalars=IDX_ARR, colormap=cmap)
fig.scene.show_axes = True

legend = mesh.module_manager.scalar_lut_manager
legend.show_scalar_bar = True
legend.scalar_bar.title = 'Q/Qgb'
# legend.data_range=[0,3]
legend.number_of_labels = 5


mlab.show()
