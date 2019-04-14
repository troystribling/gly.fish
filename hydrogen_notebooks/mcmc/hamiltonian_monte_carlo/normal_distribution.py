%load_ext autoreload
%autoreload 2

import numpy
from matplotlib import pyplot
from glyfish import config
from glyfish import gplot

%matplotlib inline

pyplot.style.use(config.glyfish_style)

# %%
# Momentum Verlet integration of Hamiltons's equations used by HMC algorithm

def momentum_verlet_hmc(p0, q0, dUdq, dKdp, nsteps, ε):
    p = p0
    q = q0
    ΔU = dUdq(q)

    for _ in range(nsteps):
        p = p - ε*ΔU/2.0
        q = q + ε*dKdp(p)
        ΔU = dUdq(q)
        p = p - ε*ΔU/2.0

    return p, q

# Hamiltonian Monte Carlo

def HMC(q0, mass, U, K, dUdq, dKdp, integrator, nsample, nsteps, ε):
    current_q = q0
    H = numpy.zeros(nsample)
    qall = numpy.zeros(nsample)
    pall = numpy.zeros(nsample)
    accepted = 0

    for j in range(nsample):

        q = current_q

        # generate momentum sample
        current_p = numpy.random.normal(0.0, numpy.sqrt(mass))

        # integrate hamiltons equations using current_p and current_q to obtain proposal samples p and q
        # and negate p for detailed balance
        p, q = integrator(current_p, current_q, dUdq, dKdp, nsteps, ε)
        p = -p

        # compute acceptance probability
        current_U = U(current_q)
        current_K = K(current_p)
        proposed_U = U(q)
        proposed_K = K(p)
        α = numpy.exp(current_U-proposed_U+current_K-proposed_K)

        # accept or reject proposal
        accept = numpy.random.rand()
        if accept < α:
            current_q = q
            qall[j] = q
            pall[j] = p
            accepted += 1
        else:
            qall[j] = current_q
            pall[j] = current_p

        H[j] = U(current_q) + K(current_p)

    return H, pall, qall, accepted

# %%
# Here Hamiltonian Monte Carlo is used to generate samples for single variable
# Unit Normal distribution. The Potential and Kinetic Energy are given by assuming a mass of 1

U = lambda q: q**2/2.0
K = lambda p: p**2/2.0
dUdq = lambda q: q
dKdp = lambda p: p
mass = 1.0
target_pdf = lambda q: numpy.exp(-U(q))/numpy.sqrt(2.0*numpy.pi)

# %%

t = 5.0
ε = 0.1
nsteps = int(t/ε)
nsample = 10000
q0 = 1.0

H, p, q, accept = HMC(q0, mass, U, K, dUdq, dKdp, momentum_verlet_hmc, nsample, nsteps, ε)
title = f"Normal Sampled HMC"
gplot.pdf_samples(title, target_pdf, q, "hamiltonian_monte_carlo", "normal_sampled_pdf-1")
