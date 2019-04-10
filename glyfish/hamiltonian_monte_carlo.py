import numpy
from scipy import stats
from scipy import special

# Euler integration of Hamiltons's equations

def euler(p0, q0, dUdq, dKdp, nsteps, ε):
    ps = numpy.zeros(nsteps+1)
    qs = numpy.zeros(nsteps+1)
    ps[0] = p0
    qs[0] = q0

    pprev = p0
    qprev = q0

    for i in range(nsteps):
        p = pprev - ε*dUdq(qprev)
        ps[i+1] = p
        q = qprev + ε*dKdp(pprev)
        qs[i+1] = q

        pprev = p
        qprev = q

    return ps, qs

# Euler-Cromer integration of Hamiltons's equations

def euler_cromer(p0, q0, dUdq, dKdp, nsteps, ε):
    ps = numpy.zeros(nsteps+1)
    qs = numpy.zeros(nsteps+1)
    ps[0] = p0
    qs[0] = q0

    pprev = p0
    qprev = q0

    for i in range(nsteps):
        p = pprev - ε*dUdq(qprev)
        ps[i+1] = p
        q = qprev + ε*dKdp(p)
        qs[i+1] = q

        pprev = p
        qprev = q

    return ps, qs

# Momentum Verlet integration of Hamiltons's equations

def momentum_verlet(p0, q0, dUdq, dKdp, nsteps, ε):
    ps = numpy.zeros(nsteps+1)
    qs = numpy.zeros(nsteps+1)
    ps[0] = p0
    qs[0] = q0

    p = p0
    q = q0
    ΔU = dUdq(q)

    for i in range(nsteps):
        p = p - ε*ΔU/2.0
        q = q + ε*dKdp(p)
        qs[i+1] = q
        ΔU = dUdq(q)
        p = p - ε*ΔU/2.0
        ps[i+1] = p

    return ps, qs

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
