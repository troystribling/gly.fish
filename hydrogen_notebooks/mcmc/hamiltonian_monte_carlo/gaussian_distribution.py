%load_ext autoreload
%autoreload 2

import numpy
from matplotlib import pyplot
from glyfish import config

%matplotlib inline

pyplot.style.use(config.glyfish_style)

# %%
# Here Hamiltonian Monte Carlo is used to generate samples for single variable
# Unit Normal distribution. The Potential and Kinetic Energy are given by assuming a mass of 1

U = lambda q: q**2/2.0
K = lambda p: p**2/2.0
dUdq = lambda q: q

# Euler Dicretization integration Hamiltons's equations

def euler(p0, q0, nintegrate, ε):
    ps = [p0]
    qs = [q0]

    pprev = p0
    qprev = q0

    for i in range(nintegrate):
        p = pprev - ε*dUdq(qprev)
        ps.append(p)
        q = qprev + ε*pprev
        qs.append(q)

        pprev = p
        qprev = q

    return ps, qs

# Leapfrog integration of Hamilton's equations

def leapfrog(p0, q0, nintegrate, ε):
    ps = [p0]
    qs = [q0]

    p = p0
    q = q0

    p = p - ε*dUdq(q)/2.0

    for i in range(nintegrate):
        q = q + ε*p
        qs.append(q)
        if (i != nintegrate-1):
            p = p - ε*dUdq(q)/2.0
            ps.append(p)

    p = p - ε*dUdq(q)/2.0
    ps.append(p)

    return ps, qs

# Hamiltonian Monte Carlo with leapfrog

def HMC(U, K, dUdq, nsample, ε, nintegrate):
    p_μ = 0.0
    p_σ = 1.0

    current_q = q_0
    current_p = p_0

    H = numpy.zeros(nsample)
    qall = numpy.zeros(nsample)
    pall = numpy.zeros(nsample)
    accept = 0.0

    for j in range(nsample):
        q = current_q
        p = current_p

        p = numpy.random.normal(p_μ, p_σ)
        current_p = p

        # leap frog
        p = p - ε*dUdq(q)/2.0

        for i in range(nintegrate):
            q = q + ε*p
            if (i != nintegrate-1):
                p = p - ε*dUdq(q)

        p = p - ε*dUdq(q)/2.0

        # negate momentum
        p = -p
        current_U = U(current_q)
        current_K = K(current_p)
        proposed_U = U(q)
        proposed_K = K(p)

        # compute acceptance probability
        α = numpy.exp(current_U-proposed_U+current_K-proposed_K)

        # accept or reject proposal
        accept = numpy.random.rand()
        if accept < α:
            current_q = q
            qall[i] = q
            pall[i] = p
            accept += 1
        else:
            qall[i] = current_q
            pall[i] = current_p

        H[j] = U(current_q) + K(current_p)

    return H, pall, qall, accept
