from t3d.tools.profile_plot import plotter
import matplotlib.pylab as pylab
import numpy as np
import sys

import matplotlib.pyplot as plt
'''
This script makes a 3 column plot showing profiles, power balance, and Q(LT)
for Ti, Te, and n.

T Qian
5 March 2024
'''

plots = plotter()

infile = sys.argv[1]
t_idx = -1

plots.read_data(infile)
plots.plot_electrons = True  # archaic, for backwards compatibility

plots.warm_map = pylab.cm.autumn(np.linspace(1, 0.25, plots.N))
plots.cool_map = pylab.cm.Blues(np.linspace(0.25, 1, plots.N))
plots.green_map = pylab.cm.YlGn(np.linspace(0.25, 1, plots.N))

fig, axs = plt.subplots(4,3,figsize=(12,9))

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
plt.show()
