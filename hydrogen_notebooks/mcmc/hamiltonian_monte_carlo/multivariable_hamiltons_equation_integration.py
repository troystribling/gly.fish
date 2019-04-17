%load_ext autoreload
%autoreload 2

import numpy
from matplotlib import pyplot
from glyfish import config
from glyfish import gplot
from glyfish import hamiltonian_monte_carlo as hmc
from glyfish import stats

%matplotlib inline

pyplot.style.use(config.glyfish_style)

# %%
# Momentum Verlet integration of Hamiltons's equations
def momentum_verlet(p0, q0, ndim, dUdq, dKdp, nsteps, ε):
    ps = numpy.zeros((2, nsteps+1))
    qs = numpy.zeros((2, nsteps+1))
    ps[0] = p0
    qs[0] = q0

    p = p0
    q = q0

    for i in range(nsteps):
        for j in range(ndim):
            ΔU = dUdq(q, j)
            p[j] = p[j] - ε*ΔU/2.0
            q[j] = q[j] + ε*dKdp(p, j)
            qs[i+1][j] = q[j]
            ΔU = dUdq(q, j)
            p[j] = p[j] - ε*ΔU/2.0
            ps[i+1][j] = p[j]

    return ps, qs

# %%
# Here Hamiltonian Monte Carlo is used to generate samples for single variable
# Unit Normal distribution. The Potential and Kinetic Energy are given by assuming a mass of 1

U = lambda q: q**2/2.0
K = lambda p: p**2/2.0
dUdq = lambda q: q
dKdp = lambda p: p
