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
    p = numpy.zeros((nsteps+1, ndim))
    q = numpy.zeros((nsteps+1, ndim))
    p[0] = p0
    q[0] = q0

    for i in range(nsteps):
        for j in range(ndim):
            ΔU = dUdq(q[i], j)
            p[i+1][j] = p[i][j] - ε*ΔU/2.0
            q[i+1][j] = q[i][j] + ε*dKdp(p[i+1], j)
            ΔU = dUdq(q[i+1], j)
            p[i+1][j] = p[i+1][j] - ε*ΔU/2.0

    return p, q

# %%
# Here Hamiltonian Monte Carlo is used to generate samples for single variable
# Unit Normal distribution. The Potential and Kinetic Energy are given by assuming a mass of 1

def U(γ):
    def f(q):
        return (q[0]**2 + q[1]**2 + γ*q[0]*q[1])
    return f

def K():
    def f(p):
        return numpy.sum(p**2) / 2.0
    return f

def dUdq(γ):
    def f(q, i):
        qshift = stats.shift(q, i - 1)
        return -(qshift[0] + γ*qshift[1])
    return f

def dKdp():
    def f(p, i):
        return p[i]
    return f

# %%

t = 5.0
ε = 0.1
nsteps = int(t/ε)
p0 = [-1.0, -1.0]
q0 = [1.0, 1.0]

p, q = momentum_verlet(p0, q0, 1, dUdq(0.0), dKdp(), nsteps, ε)
title = f"Hamilton's Equations (Momentum Verlet): Δt={ε}, nsteps={nsteps}"

q
