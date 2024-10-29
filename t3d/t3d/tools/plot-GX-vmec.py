import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

from scipy.interpolate import griddata

from t3d.Geometry import VmecReader
from mayavi import mlab

'''
GX VMEC plotting script

T. Qian - 19 March 2024
'''

# GX files
file1 = "etg/t05-p0-r6-a0-0.out.nc"
file2 = "etg/t05-p0-r6-a1-0.out.nc"
file1 = "t05-p0-r6-a0-0.out.nc"
file2 = "t05-p0-r6-a1-0.out.nc"
file1 = "w7x-gx/w7x-gx-flux-tubes/t03-d0.2-p1-r6-j0.out.nc"
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

N_THETA = 5
N_ZETA = 12

# Choose your Plots
plot_1D_Q_trace = False  # plot the two Q(z) as a function of space, averages second half of time trace
plot_1D_Qt = False       # plot the two Q(t) as a function of time

plot_2D_Q_contour = False      # 2D contour map from interpolated points
plot_2D_Q_scatter_a0 = False   # makes separated plots for both a0 and a1
plot_GX_interpolation = False  # plot interpolation output

# Warning: these 3D plots require mayavi
plot_VmecGX_unfolded = False  # plot GX flux tube over VMEC surface
plot_VmecGX_folded = False    # plot GX flux tube over VMEC surface, and fold into single field period

mayavi_3D_Q = True         # mayavi plot of Q on full stellarator
plot_3D_Q_surface = False  # make a 3D mountain plot on a 2D domain
save_mayavi_video = False  # save frames for GIF


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

    def getXYZ(self,vmec, psiN=0.434, full_torus=True):

        zeta = self.zeta
        alpha = self.alpha
        iota = self.iota

        # get surface
        s = np.argmin(np.abs(vmec.s - psiN))
        i = np.argmin(np.abs(vmec.iotaf - iota))

        if False:
            '''
            this plot debugs a potential mismatch between
            iota-vmec iota-gx and psiN-gx
            '''
            fig,axs = plt.subplots(1,1)
            arr = np.arange(vmec.ns)
            axs.plot(arr,vmec.s,label='psi-vmec')
            axs.plot(arr,vmec.iotaf,label='iota-vmec')

            axs.axhline(psiN, color='C0', ls='--', label=f"psi_Noah = {psiN}")
            axs.axhline(iota, color='C1', ls='--', label=f"iota_GX = {iota:.3f}")
            axs.axvline(s, color='C0', ls='--')
            axs.axvline(i, color='C1', ls=':')

            axs.grid()
            axs.legend()
            axs.set_title(vmec.filename)
            axs.set_xlabel("surface index")
            plt.show()

        thetaStar = self.theta
        N = len(zeta)
        theta = np.array([vmec.invertTheta(thetaStar[j], zeta=zeta[j],s_idx=s) for j in np.arange(N)])

        if False:
            '''
            this plot shows theta vs theta*
            '''
            fig,axs = plt.subplots(1,2,figsize=(9,5))
            axs[0].scatter(zeta,thetaStar,label='theta*')
            axs[0].scatter(zeta,theta,label='theta')
            axs[0].grid()
            axs[0].legend()
            axs[0].set_xlabel('zeta GX')
            axs[0].set_ylabel('theta GX')

            zp = zeta % (np.pi*2/vmec.nfp)
            tp = theta % (np.pi*2) - np.pi
            tsp = thetaStar % (np.pi*2) - np.pi
            axs[1].scatter(zp,tsp,label='theta*')
            axs[1].scatter(zp,tp,label='theta')
            axs[1].grid()
            axs[1].legend()
            axs[1].set_xlabel('zeta')
            plt.show()

        # compute m and n modes
        cos = []
        sin = []
        for j in np.arange(N):
            x = vmec.xm*theta[j] - vmec.xn*zeta[j]
            cos.append(np.cos(x))
            sin.append(np.sin(x))

        # sum of Fourier amplitudes
        R = np.sum(vmec.rmnc[s] * cos, axis=-1)
        Z = np.sum(vmec.zmns[s] * sin, axis=-1)

        # wrap around a single field period
        zeta_n = zeta % (2*np.pi/vmec.nfp)

        if full_torus:
            X = R * np.cos(zeta)
            Y = R * np.sin(zeta)
        else:
            X = R * np.cos(zeta_n)
            Y = R * np.sin(zeta_n)

        self.zeta_n = zeta_n
        self.theta_v = theta
        return X,Y,Z


# main function
fname = f"{path}/{geo}"
print('reading: ', fname)
vmec = VmecReader(f"{path}/{geo}")
print('reading flux: ', file1, file2)
f1 = FluxTube(file1)
f2 = FluxTube(file2)

if plot_VmecGX_folded:
    s = 43
    fig = mlab.figure(bgcolor=(1,1,1), fgcolor=(0.,0.,0.), size=(1200,800))
    # mlab.view(azimuth=0,elevation=90, distance = 25)

    c_vmec = (1,1,0)
    c_f1 = (0,0,1)
    c_f2 = (0,1,0)

    # plot vmec surface
    X,Y,Z = vmec.getSurfaceMesh(Ntheta=25, Nzeta=20, full_torus=True, s_idx=s)
    mlab.mesh(X,Y,Z, opacity=0.5, color=c_vmec)

    # plot f1 GX points
    x,y,z = f1.getXYZ(vmec, full_torus=False)

    # embarassingly ad hoc code
    a = np.argwhere(np.diff(y) < 0)[:,0] + 1
    for j in np.arange(len(a)-1):
        s = np.s_[a[j]:a[j+1]]
        mlab.plot3d(x[s],y[s],z[s], tube_radius=0.07, color=c_f1)
    s = np.s_[a[-1]:]
    mlab.plot3d(x[s],y[s],z[s], tube_radius=0.07, color=c_f1)

    N = len(z) // 2
    mlab.points3d(x[N], y[N], z[N], scale_factor=0.5, color=c_f1)

    # plot f2 GX points
    x,y,z = f2.getXYZ(vmec, full_torus=False)

    # embarassingly ad hoc code
    a = np.argwhere(np.diff(y) < 0)[:,0] + 1
    s = np.s_[:a[0]]
    mlab.plot3d(x[s],y[s],z[s], tube_radius=0.07, color=c_f2)
    for j in np.arange(len(a)-1):
        s = np.s_[a[j]:a[j+1]]
        mlab.plot3d(x[s],y[s],z[s], tube_radius=0.07, color=c_f2)
    s = np.s_[a[-1]:]
    mlab.plot3d(x[s],y[s],z[s], tube_radius=0.07, color=c_f2)

    # plot node
    mlab.points3d(x[N], y[N], z[N], scale_factor=0.5, color=c_f2)

    fig.scene.show_axes = True
    mlab.show()

if plot_VmecGX_unfolded:
    s = 43
    fig = mlab.figure(bgcolor=(1,1,1), fgcolor=(0.,0.,0.), size=(1200,800))
    # mlab.view(azimuth=0,elevation=90, distance = 25)

    c_vmec = (1,1,0)
    c_f1 = (0,0,1)
    c_f2 = (0,1,0)

    # plot vmec surface
    X,Y,Z = vmec.getSurfaceMesh(Ntheta=25, Nzeta=20, full_torus=True, s_idx=s)
    mlab.mesh(X,Y,Z, opacity=0.5, color=c_vmec)

    # plot f1 GX points
    x,y,z = f1.getXYZ(vmec)
    mlab.plot3d(x,y,z, tube_radius=0.07, color=c_f1)

    N = len(z) // 2
    mlab.points3d(x[N], y[N], z[N], scale_factor=0.5, color=c_f1)

    # plot f2 GX points
    x,y,z = f2.getXYZ(vmec)
    mlab.plot3d(x,y,z, tube_radius=0.07, color=c_f2)
    mlab.points3d(x[N], y[N], z[N], scale_factor=0.5, color=c_f2)

    fig.scene.show_axes = True
    mlab.show()


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

Q_gx = np.concatenate([f1.Q_gx,f2.Q_gx])  # only for plotting, not for computation
Q = np.concatenate([f1.Q,f2.Q])  # Q_gx * sum1D( norm ), norm = jacobian * grho
norm = np.concatenate([f1.norm,f2.norm])  # jacobian * grad rho
print('********************************')

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

    # pad zeta
    z_max = z0 * threshold
    arg1 = np.argwhere(z1 > z_max)[:,0]
    arg2 = np.argwhere(z1 < -z_max)[:,0]
    t2 = np.concatenate([t1, t1[arg1], t1[arg2]])
    z2 = np.concatenate([z1, z1[arg1]-2*z0, z1[arg2]+2*z0])
    Q2 = np.concatenate([Q1, Q1[arg1], Q1[arg2]])

    return Q2,z2,t2


Q_gx2, z2, t2 = pad(Q_gx,zeta,theta)
Q2, z2, t2 = pad(Q,zeta,theta)
N2, z2, t2 = pad(norm,zeta,theta)
print('Q_gx2.shape= ', Q_gx2.shape)
print('Q2.shape= ', Q2.shape)
print('N2.shape= ', N2.shape)

#  interpolation grid
Ntheta = N_THETA
Nzeta = N_ZETA
tax = np.linspace(-np.pi,np.pi,Ntheta)
zax = np.linspace(-z0,z0,Nzeta)

Z,T = np.meshgrid(zax,tax)
Qsamp = griddata((z2,t2), Q2, (Z,T), method='linear')
Nsamp = griddata((z2,t2), N2, (Z,T), method='linear')
print('Qsamp.shape= ', Qsamp.shape)
area = 4*np.pi**2/vmec.nfp
dA = area / (Nzeta-1) / (Ntheta-1)

# integrate, but exclude the endpoint
N_int = np.sum(Nsamp[:-1,:-1]) * dA
Q_int = np.sum(Qsamp[:-1,:-1])/N_int * dA

Q_plot = Qsamp / area
# such that np.mean(Q_plot) == Q_int

'''
Q_gx is Q_zst, which has spatial and time information (selecting ions)

We denormalize the spatial direction before interpolation, by multplying sum1D(jacob * grho)
Separately we interpolate (jacob * grho), in preparation for a 2D normalization

Then we will sample both interpolations on a 2D grid
The true Q_avg is
sum2D( Qsamp / sum2D(Nsamp) )
'''

if plot_1D_Q_trace:
    # plot GX data as function of space
    plt.figure(figsize=(12,3))
    Qavg1 = np.mean(f1.Q_plot1d)
    Qavg2 = np.mean(f2.Q_plot1d)
    plt.plot(f1.theta, f1.Q_plot1d, 'C0o-', label=r'$\alpha = 0$')
    plt.plot(f2.theta, f2.Q_plot1d, 'C2^-', label=r'$\alpha = \iota \pi/5$')
    plt.axhline(Qavg1, color='C0', ls='--', label=r'$\left< Q/Q_{gb} \right> = $'+f'{Qavg1:.2f}')
    plt.axhline(Qavg2, color='C2', ls='--', label=r'$\left< Q/Q_{gb} \right> = $'+f'{Qavg2:.2f}')
    plt.xlabel(r'Poloidal Angle $\theta$')
    plt.ylabel(r'$Q / Q_{gb}$')
    plt.grid()
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(12,3))
    plt.plot(f1.theta, f1.Q, 'C0o-', label=r'$\alpha = 0$')
    plt.plot(f2.theta, f2.Q, 'C2^-', label=r'$\alpha = \iota \pi/5$')
    plt.xlabel(r'Poloidal Angle $\theta$')
    plt.ylabel(r'Q_gx * flux_denom')
    plt.grid()
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(12,3))
    plt.plot(f1.theta, f1.Q_gx, 'C0o-', label=r'$\alpha = 0$')
    plt.plot(f2.theta, f2.Q_gx, 'C2^-', label=r'$\alpha = \iota \pi/5$')
    plt.xlabel(r'Poloidal Angle $\theta$')
    plt.ylabel(r'Q_gx')
    plt.grid()
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(12,3))
    plt.plot(f1.theta, f1.norm, 'C0o-', label=r'$\alpha = 0$')
    plt.plot(f2.theta, f2.norm, 'C2^-', label=r'$\alpha = \iota \pi/5$')
    plt.xlabel(r'Poloidal Angle $\theta$')
    plt.ylabel('Jacobian * gradrho')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

if plot_1D_Qt:
    # plot GX data as function of time
    plt.figure()
    plt.plot(f1.time, f1.Qt, 'C0', label=r'$\alpha = 0$')
    plt.plot(f2.time, f2.Qt, 'C2', label=r'$\alpha = \iota \pi/5$')
    plt.xlabel(r'time',fontsize=12)
    plt.ylabel(r'$Q / Q_{gb}$', fontsize=12)
    plt.legend(loc=2, fontsize=10)
    plt.grid()
    plt.tight_layout()
    plt.show()

if plot_2D_Q_scatter_a0:
    # scatter plot of GX flux tubes, shows each tube on separate panel
    fig,axs = plt.subplots(1,1, figsize=(5,5))
    zeta1 = (f1.zeta_n + z0) % (2*z0) - z0

    Qmax = np.max(f1.Q_gx/f1.dz)
    axs.scatter(zeta1, f1.theta_v, c=f1.Q_gx/f1.dz,cmap='inferno', vmax=Qmax, label=r'$\alpha = 0$')
    axs.set_xlabel(r'Toroidal Angle $\zeta$', fontsize=12)
    axs.set_ylabel(r'Poloidal Angle $\theta$', fontsize=12)
    axs.set_title(r'$Q / Q_{gb}$', fontsize=14)
    plt.legend(loc=2, fontsize=10)
    plt.grid()

    fig,axs = plt.subplots(1,1, figsize=(5,5))
    zeta2 = (f2.zeta_n + z0) % (2*z0) - z0
    axs.scatter(zeta2, f2.theta_v,c=f2.Q_gx/f2.dz, cmap='inferno', vmax=Qmax, label=r'$\alpha = \iota \pi/5$')
    axs.set_xlabel(r'Toroidal Angle $\zeta$', fontsize=12)
    axs.set_ylabel(r'Poloidal Angle $\theta$', fontsize=12)
    axs.set_title(r'$Q / Q_{gb}$', fontsize=14)
    plt.legend(loc=2, fontsize=10)
    plt.grid()

    plt.show()

if plot_2D_Q_contour:
    # make a 2D contour plot of Q
    #    the value in Q_gx is the integrand of 1D distriution \int Q dz.
    #    so we divide by dz to get a characteristic amplitude

    # plot the raw data points, folded into 2D
    fig,axs = plt.subplots(1,1, figsize=(5,5))
    axs.scatter(zeta,theta,c=Q_gx/f1.dz,cmap='inferno')
    axs.set_xlabel(r'Toroidal Angle $\zeta$', fontsize=12)
    axs.set_ylabel(r'Poloidal Angle $\theta$', fontsize=12)
    axs.set_title(r'$Q / Q_{gb}$', fontsize=14)
    axs.grid()

    # plot a tricontourf of interpolation domain
    plt.figure()
    plt.tricontourf(z2, t2, Q_gx2 / f1.dz,25,cmap='inferno')
    plt.colorbar()
    N = len(f1.zeta_n)
    plt.scatter(z2[:N], f1.theta_v,marker='o', label=r'$\alpha = 0$')
    plt.scatter(z2[N:2*N], f2.theta_v, marker='^', color='C2', label=r'$\alpha = \iota \pi/5$')
    plt.scatter(z2[2*N:], t2[2*N:], marker='x', color='C1')
    plt.xlabel(r'Toroidal Angle $\zeta$', fontsize=12)
    plt.ylabel(r'Poloidal Angle $\theta$', fontsize=12)
    plt.title(r'$Q / Q_{gb}$', fontsize=14)
    plt.legend(loc=2, fontsize=10)

    # draw a box
    plt.plot(zax[-1]*np.ones_like(tax), tax, 'w--')
    plt.plot(-z0*np.ones_like(tax), tax, 'w--')
    plt.plot(zax, -np.pi*np.ones_like(zax), 'w--')
    plt.plot(zax, tax[-1]*np.ones_like(zax), 'w--')

    plt.show()


if plot_GX_interpolation:
    # show interpolated data

    plt.figure()
    plt.contourf(Z,T,Qsamp, 25, cmap='inferno')
    plt.colorbar()
    plt.title("Q_gx * (jacobian * gradrho)")

    plt.figure()
    plt.contourf(Z,T,Nsamp, 25, cmap='inferno')
    plt.plot(Z,T,'C0.')
    plt.colorbar()
    plt.title("jacobian * grad rho")

    plt.figure()
    plt.contourf(Z,T,Q_plot, 25, cmap='inferno')
    # plt.contourf(Z,T,Qsamp/N_int/dA, 25, cmap='inferno')
    plt.colorbar()
    plt.title(r"$\left<Q/Q_{gb}\right>=$ " + f"{Q_int:.2f}", fontsize=14)
    plt.xlabel(r'Toroidal Angle $\zeta$', fontsize=12)
    plt.ylabel(r'Poloidal Angle $\theta$', fontsize=12)

    plt.show()

if plot_3D_Q_surface:

    # make a 3D mountain plot on a 2D domain
    fig = mlab.figure(bgcolor=(1,1,1), fgcolor=(0.,0.,0.), size=(1200,800))
    mlab.surf(zax*3,tax,Qsamp.T/3/N_int/dA,colormap='hot')
    fig.scene.show_axes = True

    fig = mlab.figure(bgcolor=(1,1,1), fgcolor=(0.,0.,0.), size=(1200,800))
    mlab.surf(zax*3,tax,Nsamp.T/4,colormap='hot')
    fig.scene.show_axes = True


if mayavi_3D_Q:
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

    Q = np.concatenate(Q_n,axis=0) / N_int * area
    print('Q.shape ', Q.shape)
    #shit()

    fig = mlab.figure(bgcolor=(1,1,1), fgcolor=(0.,0.,0.), size=(1000,800))
    mesh = mlab.mesh(X, Y, Z, scalars=Q, colormap='hot')
    fig.scene.show_axes = True

    legend = mesh.module_manager.scalar_lut_manager
    legend.show_scalar_bar = True
    legend.scalar_bar.title = 'Q/Qgb'
    # legend.data_range=[0,3]
    legend.number_of_labels = 5

    if save_mayavi_video:
        mlab.view(azimuth=0, elevation=90, distance=25)

        m1 = mesh
        outpath = "Qgx-gif"

        @mlab.animate(delay=10, ui=True)
        def anim():

            t = 0

            # loop through changing the z perspective by 1 degree
            for j in range(180):
                m1.actor.actor.rotate_wxyz(1,0,0,1)
                mlab.savefig(filename=f'{outpath}/{t:04d}.png', magnification=1)
                t += 1
                yield

            # loop through changing the y perspective by 1 degree
            for j in range(180):
                m1.actor.actor.rotate_wxyz(1,0,1,0)
                mlab.savefig(filename=f'{outpath}/{t:04d}.png', magnification=1)
                t += 1
                yield

            # loop through changing the z perspective by 1 degree
            for j in range(180):
                m1.actor.actor.rotate_wxyz(1,0,0,1)
                mlab.savefig(filename=f'{outpath}/{t:04d}.png', magnification=1)
                t += 1
                yield

            # loop through changing the y perspective by 1 degree
            for j in range(180):
                m1.actor.actor.rotate_wxyz(1,0,1,0)
                mlab.savefig(filename=f'{outpath}/{t:04d}.png', magnification=1)
                t += 1
                yield
        anim()

    mlab.show()

# import pdb
# pdb.set_trace()
