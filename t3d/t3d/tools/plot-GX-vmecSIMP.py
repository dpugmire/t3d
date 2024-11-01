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
path = "./w7x-case"
geo = "eq_180920017_w7x_169_wout.nc"
path = "/Users/dpn/proj/gx/3dPlot/tmp/t3d./w7x-gx"
geo = "wout_w7x.nc"

## 10.28.24
path = '../w7x-gx'
#geo = 'wout_w7x.nc'
geo = "eq_180920017_w7x_169_wout.nc"
file1 = '../w7x-gx/t05-p0-r6-a0-0.out.nc'
file2 = '../w7x-gx/t05-p0-r6-a1-0.out.nc'

N_THETA = 25
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
        dz = (theta[1]-theta[0])/scale
        Lz = len(theta) * dz

        zeta_center = data.groups['Geometry'].variables['zeta_center'][:]
        alpha = -iota*zeta_center
        zeta = (theta - alpha)/iota

        self.alpha = alpha
        self.iota = iota
        self.zeta = zeta
        self.theta = theta
        self.dz = dz
        self.Lz = Lz
        self.data = data

        # heat flux
        print('Read heat flux: s_idx= ', s_idx)
        time = data.groups['Grids'].variables['time'][:]
        print(' time.shape= ', time.shape)
        _Qt = data.groups['Diagnostics'].variables['HeatFlux_st']
        print('raw Qt.shape= ', '(time, s)',  _Qt.shape)
        print('  turbulent heat flux in gyroBohm units')
        Qt = data.groups['Diagnostics'].variables['HeatFlux_st'][:,s_idx]
        print('Qt.shape= ', Qt.shape)
        #shit()
        _Qtz = data.groups['Diagnostics'].variables['HeatFlux_zst']
        Qtz = data.groups['Diagnostics'].variables['HeatFlux_zst'][:,s_idx,:]
        print('raw Qtz.shape= ', '(time, s, theta)', _Qtz.shape)
        print('  turbulent heat flux in gyroBohm units')
        print('Qtz.shape= ', Qtz.shape)
        Q_gx = np.mean(Qtz[int(len(time)/2):,:], axis=0)
        jacobian = data.groups['Geometry'].variables['jacobian'][:]
        _jacobian = data.groups['Geometry'].variables['jacobian']
        print('raw jacobian.shape= (theta)', _jacobian.shape)
        grho = data.groups['Geometry'].variables['grho'][:]
        _grho = data.groups['Geometry'].variables['grho']
        print('raw grho.shape= (theta)', _grho.shape)
        fluxDenom = np.sum(jacobian*grho)

        # to be interpolated
        self.Q = Q_gx*fluxDenom
        self.norm = jacobian*grho
        self.grho = grho
        self.Q_gx = Q_gx
        self.Qt = Qt
        self.time = time
        self.Q_plot1d = Q_gx * Lz / dz
        print('self.Q.shape= ', self.Q.shape)
        print('self.Qt.shape= ', self.Qt.shape)
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

        # wrap around a single field period
        zeta_n = zeta % (2*np.pi/vmec.nfp)
        self.zeta_n = zeta_n
        self.theta_v = theta

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
fname = f"{path}/{geo}"
print('reading: ', fname)
vmec = VmecReader(f"{path}/{geo}")
print('reading flux: ', file1, file2)
f1 = FluxTube(file1)
f2 = FluxTube(file2)

# DRP
# load GX data onto VMEC surfaces
f1.getXYZ(vmec,full_torus=False)
f2.getXYZ(vmec,full_torus=False)
print('f1/f2.Q_gx.shape= ', f1.Q_gx.shape, f2.Q_gx.shape)
print('f1/f2.zeta_n.shape= ', f1.zeta_n.shape, f2.zeta_n.shape)
print('f1/f2.theta_v.shape= ', f1.theta_v.shape, f2.theta_v.shape)

zeta = np.concatenate([f1.zeta_n, f2.zeta_n])
z0 = np.pi/vmec.nfp
zeta = (zeta + z0) % (2*z0) - z0
theta = np.concatenate([f1.theta_v, f2.theta_v])
print('theta:', theta.size)
print(theta)

Q_gx = np.concatenate([f1.Q_gx,f2.Q_gx])  # only for plotting, not for computation
Q = np.concatenate([f1.Q,f2.Q])  # Q_gx * sum1D( norm ), norm = jacobian * grho
norm = np.concatenate([f1.norm,f2.norm])  # jacobian * grad rho
print('********************************')



## Q_gx.shape = 246.  find where z1 > z_max and z1 < -z_max (z_max = z0*threshold)
Q_gx2, z2, t2 = pad(Q_gx,zeta,theta)
Q2, z2, t2 = pad(Q,zeta,theta)
N2, z2, t2 = pad(norm,zeta,theta)
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
print('Qsamp.shape= ', Qsamp.shape)
area = 4*np.pi**2/vmec.nfp
dA = area / (Nzeta-1) / (Ntheta-1)

# integrate, but exclude the endpoint
N_int = np.sum(Nsamp[:-1,:-1]) * dA
Q_int = np.sum(Qsamp[:-1,:-1])/N_int * dA

#Q_plot = Qsamp / area
# such that np.mean(Q_plot) == Q_int



# plot mesh on surf
#DRP
#Qsamp.T = 12,5 array.
qst = Qsamp.T
print('Q.T.shape= ', Qsamp.T.shape)
srf_idx = 43
print('VMEC(zeta,theta, nfp): ', Nzeta, Ntheta, vmec.nfp)
X,Y,Z = vmec.getSurfaceMesh(Nzeta=Nzeta, Ntheta=Ntheta,s_idx=srf_idx,zeta_zero_mid=True, theta_zero_mid=True, full_torus=True)

# apply stellarator symmetry to Q
zn = np.linspace(0,np.pi*2,vmec.nfp,endpoint=False)
Q_n = []
for z in zn:
    Q_n.append(Qsamp.T)

Q_PLOT = np.concatenate(Q_n,axis=0) / N_int * area
print('Q.shape ', Q.shape)

fig = mlab.figure(bgcolor=(1,1,1), fgcolor=(0.,0.,0.), size=(1000,800))
mesh = mlab.mesh(X, Y, Z, scalars=Q_PLOT, colormap='hot')
fig.scene.show_axes = True

legend = mesh.module_manager.scalar_lut_manager
legend.show_scalar_bar = True
legend.scalar_bar.title = 'Q/Qgb'
# legend.data_range=[0,3]
legend.number_of_labels = 5


mlab.show()
