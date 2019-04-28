%load_ext autoreload
%autoreload 2

import numpy
from matplotlib import pyplot
from glyfish import config
from glyfish import gplot
from glyfish import hamiltonian_monte_carlo as hmc

%matplotlib inline

pyplot.style.use(config.glyfish_style)

# %%
# momentum verlet integrator validation

q0 = [1.0, -1.0]

m1 = 1.0
m2 = 1.0

σ1 = 1.0
σ2 = 1.0

γ = 0.0
α = 1 / (1.0 - γ**2)

ε = 0.01
ω_plus = numpy.complex(0.0, numpy.sqrt(α*(1.0 + γ)))
ω_minus = numpy.complex(0.0, numpy.sqrt(α*(1.0 - γ)))
t_plus = 2.0*numpy.pi / numpy.abs(ω_plus)
t_minus = 2.0*numpy.pi / numpy.abs(ω_minus)

nsteps = int(t_minus/(2.0*ε))
nsample = 10000

U = hmc.bivariate_normal_U(γ, σ1, σ2)
K = hmc.bivariate_normal_K(m1, m2)
dUdq = hmc.bivariate_normal_dUdq(γ, σ1, σ2)
dKdp = hmc.bivariate_normal_dKdp(m1, m2)
momentum_generator = hmc.bivariate_normal_momentum_generator(m1, m2)

# %%

H, p, q, accepted = hmc.HMC(q0, U, K, dUdq, dKdp, hmc.momentum_verlet_integrator, momentum_generator, nsample, nsteps, ε)

# %%

title = f"HMC Bivariate Normal: γ={γ}, nsample={nsample}, accepted={accepted}"
xrange = [-3.0*σ1, 3.0*σ1]
yrange = [-3.0*σ2, 3.0*σ2]
hmc.distribution_samples(q[:,0], p[:,0], xrange, yrange,  [r"$q_1$", r"$q_2$"], title, "hmc-normal-phase-space-histogram-2")

# %%

# %%

vals = q[:,0]
title = f"HMC Bivariate Normal: γ={γ}, nsample={nsample}, accepted={accepted}"
time = range(2000, 2500)
hmc.time_series(title, vals[time], time, [min(vals), max(vals)], "hmc-normal-position-timeseries-2")
