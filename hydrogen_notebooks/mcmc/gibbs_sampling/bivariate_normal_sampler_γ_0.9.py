# %%

%load_ext autoreload
%autoreload 2

import numpy

from matplotlib import pyplot
import scipy
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

q1range = [-3.1*σ1, 3.1*σ1]
q2range = [-3.1*σ2, 3.1*σ2]

bv_q1q2_generator = bv.conditional_pdf_xy_generator(μ1, μ2, σ1, σ2, γ)
bv_q2q1_generator = bv.conditional_pdf_yx_generator(μ1, μ2, σ1, σ2, γ)

file_prefix = "gibbs-bivariate-normal-γ-0.9"

# %%

def gibbs_sample(nsample, x0, f_xy_sample, f_yx_sample):
    samples = numpy.zeros((nsample, 2))
    samples[0] = x0
    for i in range(1, nsample):
        samples[i, 0] = f_xy_sample(samples[i-1, 1])
        samples[i, 1] = f_yx_sample(samples[i, 0])
    return samples

# %%

bv.contour_plot(μ1, μ2, σ1, σ2, γ, [0.01, 0.1, 0.2, 0.3], "hamiltonian_monte_carlo", "gibbs-bivariate-normal-target-pdf")

# %%

q = gibbs_sample(nsample, x0, bv_q1q2_generator, bv_q2q1_generator)

# %%

title = f"Gibbs Bivariate Normal" + f": γ={γ}, nsample={nsample}"
hmc.pdf_samples_contour(bv.pdf(μ1, μ2, σ1, σ2, γ), q[:,0], q[:,1], q1range, q2range, [0.01, 0.05, 0.15, 0.2, 0.25, 0.3, 0.35], [r"$q_1$", r"$q_2$"], title, f"{file_prefix}-samples-contour")

# %%

title = f"Gibbs Bivariate Normal "+ r"$q_1$" + f": γ={γ}, nsample={nsample}"
gplot.pdf_samples(title, bv.marginal(0.0, σ1), q[:,0], "hamiltonian_monte_carlo",  f"{file_prefix}-samples-1")

# %%

title = f"Gibbs Bivariate Normal "+ r"$q_2$" + f": γ={γ}, nsample={nsample}"
gplot.pdf_samples(title, bv.marginal(0.0, σ1), q[:,1], "hamiltonian_monte_carlo",  f"{file_prefix}-samples-2")

# %%

title = f"Gibbs Bivariate Normal "  + r"$q_1$" + f": γ={γ}, nsample={nsample}"
vals = q[:,0]
time = range(9000, 9500)
hmc.time_series(title, vals[time], time, [min(vals), max(vals)], f"{file_prefix}-timeseries-1")

# %%

title = f"Gibbs Bivariate Normal "  + r"$q_2$" + f": γ={γ}, nsample={nsample}"
vals = q[:,1]
time = range(9000, 9500)
hmc.time_series(title, vals[time], time, [min(vals), max(vals)], f"{file_prefix}-timeseries-2")

# %%

title = f"Gibbs Bivariate Normal " + r"$q_1$" + f": γ={γ}, nsample={nsample}"
time = range(0, len(q[:,0]))
hmc.cumulative_mean(title, q[:,0], time, 0.0, [-1.0, 1.0], f"{file_prefix}-cummulative-mean-1")

# %%

title = f"Gibbs Bivariate Normal " + r"$q_2$" + f": γ={γ}, nsample={nsample}"
time = range(0, len(q[:,1]))
hmc.cumulative_mean(title, q[:,1], time, 0.0, [-1.0, 1.0], f"{file_prefix}-cummulative-mean-2")

# %%

title = f"Gibbs Bivariate Normal " + r"$q_1$" + f": γ={γ}, nsample={nsample}"
time = range(0, len(q[:,0]))
hmc.cumulative_standard_deviation(title, q[:,0], time, 1.0, [0.1, 1.5], f"{file_prefix}-cummulative-sigma-1")

# %%

title = f"Gibbs Bivariate Normal " + r"$q_2$" + f": γ={γ}, nsample={nsample}"
time = range(0, len(q[:,1]))
hmc.cumulative_standard_deviation(title, q[:,1], time, 1.0, [0.1, 1.5], f"{file_prefix}-cummulative-sigma-2")

# %%

title = f"Gibbs Bivariate Normal " + r"$q_1, q_2$" + f": γ={γ}, nsample={nsample}"
hmc.cumulative_correlation(title, q[:,0], q[:,1], time, γ, f"{file_prefix}-position-cummulative-correlation")

# %%

title = f"Gibbs Bivariate Normal " + r"$q_1$" + f": γ={γ}, nsample={nsample}"
max_lag = 25
hmc.autocor(title, q[:,0], max_lag, f"{file_prefix}-autocorrelation-1")

# %%

title = f"Gibbs Bivariate Normal " + r"$q_2$" + f": γ={γ}, nsample={nsample}"
max_lag = 25
hmc.autocor(title, q[:,1], max_lag, f"{file_prefix}-autocorrelation-2")
