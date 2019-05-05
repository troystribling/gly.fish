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

def potential_energy(σ):
    def f(q):
        return q**2/(2.0 * σ**2)
    return f

def kinetic_energy(mass):
    def f(p):
        return p**2/(2.0 * mass)
    return f

def dUdq(σ):
    def f(q):
        return q/σ**2
    return f

def dKdp(mass):
    def f(p):
        return p/mass
    return f

def target_pdf(σ):
    u = potential_energy(σ)
    def f(q):
        return numpy.exp(-u(q))/numpy.sqrt(2.0*numpy.pi*σ**2.0)
    return f

def momentum_pdf(mass):
    k = kinetic_energy(mass)
    def f(p):
        return numpy.exp(-k(p))/numpy.sqrt(2.0*numpy.pi*mass)
    return f

# %%
# Parameters

# position variance
σ = 1.0

# momentum variance
mass = 1.0

# Hamilton's equation integration total time and time step
t = 5.0
ε = 0.1
nsteps = int(t/ε)

# Simulation parameters
nsample = 10000
q0 = 1.0

xrange = [-3.0*σ, 3.0*σ]
yrange = [-3.0*mass, 3.0*mass]

# %%

pdf = target_pdf(σ)
x = numpy.linspace(-3.0*σ, 3.0*σ, 500)
hmc.univariate_pdf_plot(pdf, x, "q", f"Normal Target PDF, σ={σ}", "hmc-normal-target-pdf-1")

# %%

pdf = momentum_pdf(mass)
x = numpy.linspace(-3.0*mass, 3.0*mass, 500)
hmc.univariate_pdf_plot(pdf, x, "p", f"Momentum PDF, mass={mass}", "hmc-momentum-pdf-1")

# %%

pdf = hmc.canonical_distribution(potential_energy(σ), kinetic_energy(mass))
hmc.pdf_contour_plot(pdf, [0.01, 0.025, 0.05, 0.1, 0.15, 0.2], xrange, yrange, ["q", "p"], "Canonical Distribution", "hmc-normal-target-phase-space-1")

# %%

H, p, q, accept = HMC(q0, mass, potential_energy(σ), kinetic_energy(mass), dUdq(σ), dKdp(mass), momentum_verlet_hmc, nsample, nsteps, ε)

# %%

title = f"HMC Normal: σ={σ}, nsample={nsample}, accepted={accept}"
hmc.pdf_samples_contour(pdf, p, q, xrange, yrange, [0.01, 0.025, 0.05, 0.1, 0.15, 0.2], ["q", "p"], title, "hmc-normal-phase-space-histogram-1")

# %%

hmc.distribution_samples(p, q, xrange, yrange,  ["q", "p"], title, "hmc-normal-phase-space-histogram-2")

# %%

title = f"HMC Normal Target: Δt={ε}, nsteps={nsteps}, nsample={nsample}, accepted={accept}"
gplot.pdf_samples(title, target_pdf(σ), q, "hamiltonian_monte_carlo", "hmc-normal-sampled-pdf-1")

# %%

title = f"HMC Normal Target: Δt={ε}, nsteps={nsteps}, nsample={nsample}, accepted={accept}"
time = range(0, len(q))
hmc.time_series(title, q, time, [min(q), max(q)], "hmc-normal-position-timeseries-1")

# %%

title = f"HMC Normal Target: Δt={ε}, nsteps={nsteps}, nsample={nsample}, accepted={accept}"
time = range(9000, 9500)
hmc.time_series(title, q[time], time, [min(q), max(q)], "hmc-normal-position-timeseries-2")

# %%

title = f"HMC Normal Target: Δt={ε}, nsteps={nsteps}, nsample={nsample}, accepted={accept}"
time = range(0, len(q))
hmc.cumulative_mean(title, q, time, 0.0, [-0.5, 0.5], "hmc-normal-position-cummulative-mean-1")

# %%

title = f"HMC Normal Target: Δt={ε}, nsteps={nsteps}, nsample={nsample}, accepted={accept}"
time = range(0, len(q))
hmc.cumulative_standard_deviation(title, q, time, 1.0, [0.5, 2.0], "hmc-normal-position-cummulative-sigma-1")

# %%

title = f"HMC Normal Target: Δt={ε}, nsteps={nsteps}, nsample={nsample}, accepted={accept}"
max_lag = 25
hmc.autocor(title, q, max_lag, "hmc-normal-position-autocorrelation-1")
