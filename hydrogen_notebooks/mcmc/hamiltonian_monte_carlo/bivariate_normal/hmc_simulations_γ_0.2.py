%load_ext autoreload
%autoreload 2

import numpy
from matplotlib import pyplot
from glyfish import config
from glyfish import gplot
from glyfish import hamiltonian_monte_carlo as hmc
from glyfish import bivariate_normal_distribution as bv

%matplotlib inline

pyplot.style.use(config.glyfish_style)

# %%
# momentum verlet integrator validation

q0 = [1.0, -1.0]

m1 = 1.0
m2 = 1.0

σ1 = 1.0
σ2 = 1.0

γ = 0.2
α = 1 / (1.0 - γ**2)

q1range = [-3.1*σ1, 3.1*σ1]
q2range = [-3.1*σ2, 3.1*σ2]

p1range = [-3.1*m1, 3.1*m1]
p2range = [-3.1*m2, 3.1*m2]

ε = 0.01
ω_plus = numpy.complex(0.0, numpy.sqrt(α*(1.0 + γ)))
ω_minus = numpy.complex(0.0, numpy.sqrt(α*(1.0 - γ)))
t_plus = 2.0*numpy.pi / numpy.abs(ω_plus)
t_minus = 2.0*numpy.pi / numpy.abs(ω_minus)

tmax = 2.0*t_minus/3.0
nsample = 10000

U = hmc.bivariate_normal_U(γ, σ1, σ2)
K = hmc.bivariate_normal_K(m1, m2)
dUdq = hmc.bivariate_normal_dUdq(γ, σ1, σ2)
dKdp = hmc.bivariate_normal_dKdp(m1, m2)
momentum_generator = hmc.bivariate_normal_momentum_generator(m1, m2)
potential_pdf = hmc.potential_distribution(U)
momentum_pdf = hmc.momentum_distribution(K)

file_prefix = "hmc-bivariate-normal-γ-0.2"

# %%

H, p, q, accepted = hmc.HMC(q0, U, K, dUdq, dKdp, hmc.momentum_verlet_integrator, momentum_generator, nsample, tmax, ε)

# %%

bv.contour_plot(0.0, 0.0, σ1, σ2, γ,  [0.01, 0.05, 0.1, 0.15, 0.2], "hamiltonian_monte_carlo", f"{file_prefix}-target-distribution-1")

# %%

title = f"HMC Normal" + f": γ={γ}, nsample={nsample}, accepted={int(100.0*float(accepted)/float(nsample))}%, " + r"$t_{max}$=" + f"{format(tmax, '2.2f')}"
hmc.pdf_samples_contour(potential_pdf, q[:,0], q[:,1], q1range, q2range, [0.01, 0.05, 0.1, 0.15, 0.2], [r"$q_1$", r"$q_2$"], title, f"{file_prefix}-position-samples-contour-1")

# %%

title = f"HMC Normal" + f": γ={γ}, nsample={nsample}, accepted={int(100.0*float(accepted)/float(nsample))}%, " + r"$t_{max}$=" + f"{format(tmax, '2.2f')}"
hmc.pdf_samples_contour(momentum_pdf, p[:,0], p[:,1], p1range, p2range, [0.01, 0.025, 0.05, 0.1, 0.15, 0.2], [r"$p_1$", r"$p_2$"], title, f"{file_prefix}-momentum-samples-contour-1")

# %%

title = f"HMC Normal " + r"$q_1$" + f": γ={γ}, nsample={nsample}, accepted={int(100.0*float(accepted)/float(nsample))}%, " + r"$t_{max}$=" + f"{format(tmax, '2.2f')}"
gplot.pdf_samples(title, bv.marginal(0.0, 1.0), q[:,0], "hamiltonian_monte_carlo",  f"{file_prefix}-position-samples-1")

# %%

title = f"HMC Normal " + r"$q_2$" + f": γ={γ}, nsample={nsample}, accepted={int(100.0*float(accepted)/float(nsample))}%, " + r"$t_{max}$=" + f"{format(tmax, '2.2f')}"
gplot.pdf_samples(title, bv.marginal(0.0, 1.0), q[:,1], "hamiltonian_monte_carlo",  f"{file_prefix}-position-samples-2")

# %%

title = f"HMC Bivariate Normal " + r"$q_1$" + f": γ={γ}, nsample={nsample}, accepted={int(100.0*float(accepted)/float(nsample))}%, " + r"$t_{max}$=" + f"{format(tmax, '2.2f')}"
vals = q[:,0]
time = range(9000, 9500)
hmc.time_series(title, vals[time], time, [min(vals), max(vals)], f"{file_prefix}-position-timeseries-1")

# %%

title = f"HMC Bivariate Normal " + r"$q_2$" + f": γ={γ}, nsample={nsample}, accepted={int(100.0*float(accepted)/float(nsample))}%, " + r"$t_{max}$=" + f"{format(tmax, '2.2f')}"
vals = q[:,1]
time = range(9000, 9500)
hmc.time_series(title, vals[time], time, [min(vals), max(vals)], f"{file_prefix}-position-timeseries-2")

# %%

title = f"HMC Bivariate Normal " + r"$q_1$" + f": γ={γ}, nsample={nsample}, accepted={int(100.0*float(accepted)/float(nsample))}%, " + r"$t_{max}$=" + f"{format(tmax, '2.2f')}"
time = range(0, len(q[:,0]))
hmc.cumulative_mean(title, q[:,0], time, 0.0, [-0.5, 0.5], f"{file_prefix}-position-cummulative-mean-1")

# %%

title = f"HMC Bivariate Normal " + r"$q_2$" + f": γ={γ}, nsample={nsample}, accepted={int(100.0*float(accepted)/float(nsample))}%, " + r"$t_{max}$=" + f"{format(tmax, '2.2f')}"
time = range(0, len(q[:,1]))
hmc.cumulative_mean(title, q[:,1], time, 0.0, [-0.5, 0.5], f"{file_prefix}-position-cummulative-mean-2")

# %%

title = f"HMC Bivariate Normal " + r"$q_1$" + f": γ={γ}, nsample={nsample}, accepted={int(100.0*float(accepted)/float(nsample))}%, " + r"$t_{max}$=" + f"{format(tmax, '2.2f')}"
time = range(0, len(q[:,0]))
hmc.cumulative_standard_deviation(title, q[:,0], time, 1.0, [0.5, 1.5], f"{file_prefix}-position-cummulative-sigma-1")

# %%

title = f"HMC Bivariate Normal " + r"$q_2$" + f": γ={γ}, nsample={nsample}, accepted={int(100.0*float(accepted)/float(nsample))}%, " + r"$t_{max}$=" + f"{format(tmax, '2.2f')}"
time = range(0, len(q[:,1]))
hmc.cumulative_standard_deviation(title, q[:,1], time, 1.0, [0.5, 1.5], f"{file_prefix}-position-cummulative-sigma-2")

# %%

title = f"HMC Bivariate Normal " + r"$q_1, q_2$" + f": γ={γ}, nsample={nsample}, accepted={int(100.0*float(accepted)/float(nsample))}%, " + r"$t_{max}$=" + f"{format(tmax, '2.2f')}"
hmc.cumulative_correlation(title, q[:,0], q[:,1], time, γ, f"{file_prefix}-position-cummulative-correlation")

# %%

title = f"HMC Bivariate Normal " + r"$q_1$" + f": γ={γ}, nsample={nsample}, accepted={int(100.0*float(accepted)/float(nsample))}%, " + r"$t_{max}$=" + f"{format(tmax, '2.2f')}"
max_lag = 25
hmc.autocor(title, q[:,0], max_lag, f"{file_prefix}-position-autocorrelation-1")

# %%

title = f"HMC Bivariate Normal " + r"$q_2$" + f": γ={γ}, nsample={nsample}, accepted={int(100.0*float(accepted)/float(nsample))}%, " + r"$t_{max}$=" + f"{format(tmax, '2.2f')}"
max_lag = 25
hmc.autocor(title, q[:,1], max_lag, f"{file_prefix}-position-autocorrelation-2")
