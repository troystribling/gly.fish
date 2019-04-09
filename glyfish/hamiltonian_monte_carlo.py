import numpy
from scipy import stats
from scipy import special

# Euler integration of Hamiltons's equations

def euler(p0, q0, nintegrate, ε, dUdq, dKdp):
    ps = numpy.zeros(nintegrate+1)
    qs = numpy.zeros(nintegrate+1)
    ps[0] = p0
    qs[0] = q0

    pprev = p0
    qprev = q0

    for i in range(nintegrate):
        p = pprev - ε*dUdq(qprev)
        ps[i+1] = p
        q = qprev + ε*dKdp(pprev)
        qs[i+1] = q

        pprev = p
        qprev = q

    return ps, qs

# Euler-Cromer integration of Hamiltons's equations

def euler_cromer(p0, q0, nsteps, ε, dUdq, dKdp):
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

def momentum_verlet(p0, q0, nsteps, ε, dUdq, dKdp):
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

# Hamiltonian Monte Carlo

def HMC(U, K, dUdq, dKdp, nsample, ε, nsteps):
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

        for i in range(nsteps):
            q = q + ε*p
            if (i != nsteps-1):
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
