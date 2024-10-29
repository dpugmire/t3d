from t3d.tools.profile_plot import plotter
from t3d.Profiles import GridProfile

import numpy as np
import matplotlib.pyplot as plt
import sys

profile_plot = False
skip_Newton_iterations = True

save = False
outdir = "plot-t/"
infiles = sys.argv[1:]


def time_trace(infile):

    # load data
    self = plotter()
    self.read_data(infile)
    data = np.load(infile, allow_pickle=True).tolist()
    t_time = np.array(data['t'])
    t_step = data['t_step_idx']  # time step (ignoring repeats)
    i_max = t_step[-1]
    # get the index before each change
    arr = [t_step.index(i+1) - 1 for i in np.arange(i_max)]

    def plotTrace(data, ax,label=''):

        data = np.array(data)
        tax = np.arange(len(data))
        if skip_Newton_iterations:
            ax.plot(tax[arr], data[arr], label=label)
        else:
            ax.plot(data, label=label)

    P_aux_t = self.Sp_aux_int_MW_tot
    S_tot_t = np.sum(self.Sn_tot_cumint_SI20_i + self.Sn_tot_cumint_SI20_e, axis=-1) * 1e20

    total_pressure = (self.pe + self.pi) * 1e20 * 1.609e-16 / 1e6
    convert = self.geo.volume_integrate
    grid = self.geo.grid
    W_tot_t = np.array([convert(GridProfile(p,grid)) for p in total_pressure])  # MJ
    N_tot_t = np.array([convert(GridProfile(2*n,grid)) for n in self.n]) * 1e20

    tauE_t = W_tot_t / P_aux_t
    tauP_t = N_tot_t / S_tot_t

    P_alpha_t = np.array(self.Palpha_int_MW)
    Q_power_t = 5.03*P_alpha_t/P_aux_t

    n0 = self.n[:,0] * 1e20
    Ti0 = self.Ti[:,0]  # keV
    nTtau_t = n0*Ti0*tauE_t
    ntau_t = n0*tauE_t

    t_rms = self.data['t_rms']
    # calculate sudo density limit
    P_ext_total = P_alpha_t + P_aux_t
    # checked that (R,B) includes scaling
    B0 = np.mean(self.geo.Btor)  # avg over r-axis
    R0 = self.geo.a_minor * self.geo.AspectRatio
    a0 = self.geo.a_minor

    V0 = (np.pi * a0 * a0) * (2*np.pi*R0)
    n_sudo_t = np.sqrt(P_ext_total * B0 / V0)

    P_rad_t = np.array(self.Prad_tot)

    # calculate beta
    p_avg = W_tot_t*1e6 / V0
    mu0 = 4*np.pi/1e7
    beta_t = p_avg / B0**2 * (2 * mu0)

    fig,axs = plt.subplots(5,2,figsize=(12,8))
    # these are NOT filtered to remove Newton iteration
    # actually these have no time axis at all
    axs[0,0].plot(P_alpha_t, label=r'$P_\alpha$ [MW]')
    axs[0,0].plot(P_aux_t, label=r'$P_\mathrm{aux}$ [MW]')
    axs[0,0].plot(P_ext_total, label=r'$P_\alpha + P_\mathrm{aux}$ [MW]')
    axs[0,0].plot(-P_rad_t, label=r'$-P_\mathrm{rad}$ [MW]')
    axs[0,1].plot(t_rms, '--')
    plotTrace(t_rms, axs[0,1], label='rms error')
    axs[1,0].plot(W_tot_t, '--')
    plotTrace(W_tot_t, axs[1,0], label='W_stored [MJ]')
    axs[1,1].plot(N_tot_t, label='total particles')
    axs[2,0].plot(Q_power_t, label='Q')
    axs[2,1].plot(tauE_t, '--')
    plotTrace(tauE_t, axs[2,1], label='tau_E')
    axs[2,1].plot(tauP_t, label='tau_P')
    axs[3,0].plot(ntau_t, label=r'$n_0 \tau_E [s/m^3]$')
    axs[3,1].plot(nTtau_t, label=r'$n_0 T_{i0} \tau_E [s \cdot keV / m^3]$')
    axs[4,0].plot(beta_t, '--')
    plotTrace(beta_t, axs[4,0], label=r'$\beta_{avg}$')
    axs[4,1].plot(n_sudo_t, label=r'$n_{sudo}$')
    axs[4,1].plot(n0/1e20, label=r'$n_{0}$')  # core or avg?

    axs[0,0].set_yscale('log')
    axs[0,1].set_yscale('log')
    axs[2,0].set_yscale('log')
    axs[2,1].set_yscale('log')
    axs[3,0].set_yscale('log')
    axs[3,1].set_yscale('log')
#    axs[4,1].set_yscale('log')

    axs[3,0].axhline(1.5e20, ls='--', color='r', label='lawson')
    axs[3,1].axhline(3e21, ls='--', color='r', label='lawson')
    axs[2,0].axhline(1, ls='--', color='r', label='breakeven')

    [a.legend(loc=1) for a in np.ndarray.flatten(axs)]
    axs[-1,0].set_xlabel('integer step index')
    axs[-1,1].set_xlabel('integer step index')

    fig.suptitle(infile)
    plt.tight_layout()


for fin in infiles:
    time_trace(fin)

    if save:
        out = f"{outdir}plot-t_{fin[:-8]}.png"
        plt.savefig(out)
        print(f"  saved {out}")
        plt.close()

if not save:
    plt.show()

# presently unused
if profile_plot:
    fig, axs = plt.subplots(4,3,figsize=(12,9))

    t_idx = -1
    plots.plot_state_profiles(axs[0,0], profile='Ti', t_stop=t_idx)
    plots.plot_state_profiles(axs[0,1], profile='Te', t_stop=t_idx)
    plots.plot_state_profiles(axs[0,2], profile='ne', t_stop=t_idx)

    plots.plot_source_total(axs[1,0], profile='Ti', t=t_idx)
    plots.plot_source_total(axs[1,1], profile='Te', t=t_idx)
    plots.plot_source_total(axs[1,2], profile='ne', t=t_idx)

    plots.plot_sink_terms(axs[2,0], profile='Ti', t=t_idx)
    plots.plot_sink_terms(axs[2,1], profile='Te', t=t_idx)
    plots.plot_sink_terms(axs[2,2], profile='ne', t=t_idx)

    plots.plot_flux(axs[-1,0], profile='Ti', t_stop=t_idx)
    plots.plot_flux(axs[-1,1], profile='Te', t_stop=t_idx)
    plots.plot_flux(axs[-1,2], profile='ne', t_stop=t_idx)

    fig.suptitle(infile)
    plt.tight_layout()
