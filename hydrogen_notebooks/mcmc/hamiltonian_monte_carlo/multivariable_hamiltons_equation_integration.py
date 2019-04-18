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
            p[i+1][j] = p[i][j] - ε*dUdq(q[i], j)/2.0
            q[i+1][j] = q[i][j] + ε*dKdp(p[i+1], j)
            p[i+1][j] = p[i+1][j] - ε*dUdq(q[i+1], j)/2.0

    return p, q

# %%
# Here Hamiltonian Monte Carlo is used to generate samples for single variable
# Unit Normal distribution. The Potential and Kinetic Energy are given by assuming a mass of 1

def U(γ, σ1, σ2):
    def f(q):
        return (q[0]**2 + q[1]**2 + γ*q[0]*q[1])
    return f

def K(m1, m2):
    def f(p):
        return numpy.sum(p**2) / 2.0
    return f

def dUdq(γ, σ1, σ2):
    def f(q, i):
        if i == 0:
            return q[0]*σ1**2 + q[1]*γ*σ1*σ2
        elif i == 1:
            return q[1]*σ2**2 + q[0]*γ*σ1*σ2
    return f

def dKdp(m1, m2):
    def f(p, i):
        if i == 0:
            return p[0]/m1
        elif i == 1:
            return p[1]/m2
    return f

# %%
# Integration terms

t = 2.0*numpy.pi
ε = 0.1
nsteps = int(t/ε)
p0 = [-1.0, -1.0]
q0 = [1.0, 1.0]

m1 = 1.0
m2 = 1.0

σ1 = 1.0
σ2 = 1.0
γ = 0.0

# %%

p, q = momentum_verlet(p0, q0, 2, dUdq(γ, σ1, σ2), dKdp(m1, m2), nsteps, ε)
title = f"Hamilton's Equations (Momentum Verlet): Δt={ε}, nsteps={nsteps}"
hmc.phase_space_plot(p[:,0], q[:,0], title, "bivariate_normal_dim_0_plot_1")
