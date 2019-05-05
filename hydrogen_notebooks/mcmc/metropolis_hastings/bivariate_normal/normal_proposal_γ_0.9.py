# %%
%load_ext autoreload
%autoreload 2

import numpy
from matplotlib import pyplot
from glyfish import metropolis_hastings as mh
from glyfish import bivariate_normal_distribution as bv
from glyfish import hamiltonian_monte_carlo as hmc
from glyfish import config
from glyfish import gplot

%matplotlib inline

pyplot.style.use(config.glyfish_style)

# %%

nsample = 10000
x0 = numpy.array([1.0, -1.0])

stepsize = 0.25

μ1 = 0.0
μ2 = 0.0
σ1 = 1.0
σ2 = 1.0
γ = 0.9

target_pdf = bv.metropolis_hastings_target_pdf(μ1, μ2, σ1, σ2, γ)

q1range = [-3.1*σ1, 3.1*σ1]
q2range = [-3.1*σ2, 3.1*σ2]

file_prefix = "mh-bivariate-normal-γ-0.9"

# %%

bv.contour_plot(μ1, μ2, σ1, σ2, γ, [0.01, 0.1, 0.2, 0.3], "hamiltonian_monte_carlo", "metropolis-hastings-bivariate-normal-target-pdf")

# %%

q, accepted = mh.component_metropolis_hastings(target_pdf, mh.normal_proposal, mh.normal_generator, stepsize, nsample=nsample, x0=x0)
accepted = accepted / 2.0

# %%

title = f"MH Bivariate Normal" + f": γ={γ}, nsample={nsample}, accepted={int(100.0*float(accepted)/float(nsample))}%"
hmc.pdf_samples_contour(bv.pdf(μ1, μ2, σ1, σ2, γ), q[:,0], q[:,1], q1range, q2range, [0.01, 0.05, 0.15, 0.2, 0.25, 0.3, 0.35], [r"$q_1$", r"$q_2$"], title, f"{file_prefix}-samples-contour")

# %%

title = f"MH Bivariate Normal "  + r"$q_1$" + f": γ={γ}, nsample={nsample}, accepted={int(100.0*float(accepted)/float(nsample))}%"
gplot.pdf_samples(title, bv.marginal(0.0, σ1), q[:,0], "hamiltonian_monte_carlo",  f"{file_prefix}-samples-1")

# %%

title = f"MH Bivariate Normal "  + r"$q_2$" + f": γ={γ}, nsample={nsample}, accepted={int(100.0*float(accepted)/float(nsample))}%"
gplot.pdf_samples(title, bv.marginal(0.0, σ2), q[:,1], "hamiltonian_monte_carlo",  f"{file_prefix}-samples-2")

# %%

title = f"MH Bivariate Normal "  + r"$q_1$" + f": γ={γ}, nsample={nsample}, accepted={int(100.0*float(accepted)/float(nsample))}%"
vals = q[:,0]
time = range(9000, 9500)
hmc.time_series(title, vals[time], time, [min(vals), max(vals)], f"{file_prefix}-timeseries-1")

# %%

title = f"MH Bivariate Normal " + r"$q_2$" + f": γ={γ}, nsample={nsample}, accepted={int(100.0*float(accepted)/float(nsample))}%"
vals = q[:,1]
time = range(9000, 9500)
hmc.time_series(title, vals[time], time, [min(vals), max(vals)], f"{file_prefix}-timeseries-2")

# %%

title = f"MH Bivariate Normal " + r"$q_1$" + f": γ={γ}, nsample={nsample}, accepted={int(100.0*float(accepted)/float(nsample))}%"
time = range(0, len(q[:,0]))
hmc.cumulative_mean(title, q[:,0], time, 0.0, [-1.0, 1.0], f"{file_prefix}-cummulative-mean-1")

# %%

title = f"MH Bivariate Normal " + r"$q_2$" + f": γ={γ}, nsample={nsample}, accepted={int(100.0*float(accepted)/float(nsample))}%"
time = range(0, len(q[:,1]))
hmc.cumulative_mean(title, q[:,1], time, 0.0, [-1.0, 1.0], f"{file_prefix}-cummulative-mean-1")

# %%

title = f"MH Bivariate Normal " + r"$q_1$" + f": γ={γ}, nsample={nsample}, accepted={int(100.0*float(accepted)/float(nsample))}%"
time = range(0, len(q[:,0]))
hmc.cumulative_standard_deviation(title, q[:,0], time, 1.0, [0.1, 1.5], f"{file_prefix}-cummulative-sigma-1")

# %%

title = f"MH Bivariate Normal " + r"$q_2$" + f": γ={γ}, nsample={nsample}, accepted={int(100.0*float(accepted)/float(nsample))}%"
time = range(0, len(q[:,1]))
hmc.cumulative_standard_deviation(title, q[:,1], time, 1.0, [0.1, 1.5], f"{file_prefix}-cummulative-sigma-2")

# %%

title = f"MH Bivariate Normal " + r"$q_1, q_2$" + f": γ={γ}, nsample={nsample}, accepted={int(100.0*float(accepted)/float(nsample))}%"
hmc.cumulative_correlation(title, q[:,0], q[:,1], time, γ, f"{file_prefix}-position-cummulative-correlation")

# %%

title = f"MH Bivariate Normal " + r"$q_1$" + f": γ={γ}, nsample={nsample}, accepted={int(100.0*float(accepted)/float(nsample))}%"
max_lag = 25
hmc.autocor(title, q[:,0], max_lag, f"{file_prefix}-autocorrelation-1")

# %%

title = f"MH Bivariate Normal " + r"$q_2$" + f": γ={γ}, nsample={nsample}, accepted={int(100.0*float(accepted)/float(nsample))}%"
max_lag = 25
hmc.autocor(title, q[:,1], max_lag, f"{file_prefix}-autocorrelation-2")
