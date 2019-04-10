%load_ext autoreload
%autoreload 2

import numpy
from matplotlib import pyplot
from glyfish import config
from glyfish import hamiltonian_monte_carlo as hmc
from glyfish import gplot

%matplotlib inline

pyplot.style.use(config.glyfish_style)

# %%
# Here Hamiltonian Monte Carlo is used to generate samples for single variable
# Unit Normal distribution. The Potential and Kinetic Energy are given by assuming a mass of 1

U = lambda q: q**2/2.0
K = lambda p: p**2/2.0
dUdq = lambda q: q
dKdp = lambda p: p
mass = 1.0
target_pdf = lambda q: numpy.exp(-K(q))/numpy.sqrt(2.0*numpy.pi)

# %%

t = 5.0
ε = 0.1
nsteps = int(t/ε)
nsample = 10000
q0 = 1.0

H, p, q, accept = hmc.HMC(q0, mass, U, K, dUdq, dKdp, hmc.momentum_verlet_hmc, nsample, nsteps, ε)
title = f"Normal Sampled HMC"
gplot.pdf_samples(title, target_pdf, q, "hamiltonian_monte_carlo", "normal_sampled_pdf-1")
