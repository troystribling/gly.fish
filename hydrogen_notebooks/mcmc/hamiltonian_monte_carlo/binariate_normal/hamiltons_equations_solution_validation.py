# %%
%load_ext autoreload
%autoreload 2

import numpy
from numpy import linalg
from matplotlib import pyplot
from glyfish import config
from glyfish import hamiltonian_monte_carlo as hmc

%matplotlib inline

pyplot.style.use(config.glyfish_style)

# %%

def hamiltonian_matrix(γ, α):
    m = [[0.0, 0.0, -α, α*γ],
         [0.0, 0.0, α*γ, -α],
         [1.0, 0.0, 0.0, 0.0],
         [0.0, 1.0, 0.0, 0.0]]
    return numpy.matrix(m)

def eigenvector_matrix(γ, α):
    ω_plus = numpy.complex(0.0, numpy.sqrt(α*(1.0 + γ)))
    ω_minus = numpy.complex(0.0, numpy.sqrt(α*(1.0 - γ)))
    m = [[ω_plus, numpy.conj(ω_plus), ω_minus, numpy.conj(ω_minus)],
         [numpy.conj(ω_plus), ω_plus, ω_minus, numpy.conj(ω_minus)],
         [1.0, 1.0, 1.0, 1.0],
         [-1.0, -1.0, 1.0, 1.0]]
    m = numpy.matrix(m)
    _, col = m.shape
    for i in range(0, col):
        m[:,i] = m[:,i] / linalg.norm(m[:,i])
    return m

def eigenvector_matrix_unnormalized(γ, α):
    ω_plus = numpy.complex(0.0, numpy.sqrt(α*(1.0 + γ)))
    ω_minus = numpy.complex(0.0, numpy.sqrt(α*(1.0 - γ)))
    m = [[ω_plus, numpy.conj(ω_plus), ω_minus, numpy.conj(ω_minus)],
         [numpy.conj(ω_plus), ω_plus, ω_minus, numpy.conj(ω_minus)],
         [1.0, 1.0, 1.0, 1.0],
         [-1.0, -1.0, 1.0, 1.0]]
    return numpy.matrix(m)

def eigenvalues(γ, α):
    ω_plus = numpy.complex(0.0, numpy.sqrt(α*(1.0 + γ)))
    ω_minus = numpy.complex(0.0, numpy.sqrt(α*(1.0 - γ)))
    return [ω_plus, numpy.conj(ω_plus), ω_minus, numpy.conj(ω_minus)]

def coordinate_time_series(E, PQ0, λ, time):
    nsteps = len(time)
    PQ = numpy.zeros((nsteps, 4, 1), dtype=complex)
    Einv = linalg.inv(E)
    C = Einv * PQ0
    T = lambda t: numpy.matrix([[numpy.exp(λ[0]*t)], [numpy.exp(λ[1]*t)], [numpy.exp(λ[2]*t)], [numpy.exp(λ[3]*t)]])
    for i in range(nsteps):
        TC = numpy.multiply(C, T(time[i]))
        PQ[i] = E * TC
    return PQ

def phase_space_time_series(title, PQ, time, ylim, file):
    nplots = 2
    figure, axis = pyplot.subplots(nrows=nplots, ncols=1, sharex=True, figsize=(10, 4*nplots))
    axis[0].set_title(title)
    axis[-1].set_xlabel("Time")

    axis[0].set_xlim([time[0], time[-1]])
    axis[0].set_ylim(ylim)
    axis[0].set_ylabel("q")
    axis[0].plot(time, numpy.real(PQ[:, 2, 0]), label=r"$q_1$")
    axis[0].plot(time, numpy.real(PQ[:, 3, 0]), label=r"$q_2$")
    axis[0].legend()

    axis[1].set_xlim([time[0], time[-1]])
    axis[1].set_ylim(ylim)
    axis[1].set_ylabel("p")
    axis[1].plot(time, numpy.real(PQ[:, 0, 0]), label=r"$p_1$")
    axis[1].plot(time, numpy.real(PQ[:, 1, 0]), label=r"$p_2$")
    axis[1].legend()

    config.save_post_asset(figure, "hamiltonian_monte_carlo", file)

def verification_time_series(title, p, q, time, ylim, file):
    pt = [p(t) for t in time]
    qt = [q(t) for t in time]
    nplots = 2
    figure, axis = pyplot.subplots(nrows=nplots, ncols=1, sharex=True, figsize=(10, 4*nplots))
    axis[0].set_title(title)
    axis[-1].set_xlabel("Time")

    axis[0].set_xlim([time[0], time[-1]])
    axis[0].set_ylim(ylim)
    axis[0].set_ylabel("q")
    axis[0].plot(time, qt)

    axis[1].set_xlim([time[0], time[-1]])
    axis[1].set_ylim(ylim)
    axis[1].set_ylabel("p")
    axis[1].plot(time, pt)

    config.save_post_asset(figure, "hamiltonian_monte_carlo", file)

def total_energy(PQ, U, K):
    npts = len(PQ)
    q = numpy.array([[numpy.real(PQ[i,2,0]), numpy.real(PQ[i,3,0])] for i in range(npts)])
    p = numpy.array([[numpy.real(PQ[i,0,0]), numpy.real(PQ[i,1,0])] for i in range(npts)])
    Ut = numpy.array([U(qt) for qt in q])
    Kt =  numpy.array([K(pt) for pt in p])
    return Kt, Ut, Ut + Kt

# %%
# Configuration

γ = 0.9
α = 1 / (1.0 - γ**2)
nsteps = 500

ω_plus = numpy.sqrt(α*(1.0 + γ))
ω_minus = numpy.sqrt(α*(1.0 - γ))
t_plus = 2.0*numpy.pi / numpy.abs(ω_plus)
t_minus = 2.0*numpy.pi / numpy.abs(ω_minus)

U = hmc.bivariate_normal_U(γ, 1.0, 1.0)
K = hmc.bivariate_normal_K(1.0, 1.0)

PQ0 = numpy.matrix([[1.0], [1.0], [1.0], [1.0]])
time = numpy.linspace(0.0, 2.0*t_minus, nsteps)

# %%
# Compute solutions using eigenvalues and eigenvectots computed algebrically

λ = eigenvalues(γ, α)
E = eigenvector_matrix(γ, α)
PQ = coordinate_time_series(E, PQ0, λ, time)
title = f"Calculated Soultion: γ={γ}, " + r"$t_{+}=$" + f"{format(t_plus, '2.5f')}, " + r"$t_{-}=$" + f"{format(t_minus, '2.5f')}"
phase_space_time_series(title, PQ, time, [-2.1, 2.1], "binvariate_normal_verification_calculated_09_1")

# %%
# Compute coefficients for solution using intial conditions
λ = eigenvalues(γ, α)
E = eigenvector_matrix_unnormalized(γ, α)
Einv = linalg.inv(E)
C = UEinv * PQ0
CR = numpy.real(C[2,0])
CI = numpy.imag(C[2,0])

# %%

p = lambda t: -2*ω_minus*(CI*numpy.cos(ω_minus*t) + CR*numpy.sin(ω_minus*t))
q = lambda t: 2*(CR*numpy.cos(ω_minus*t) - CI*numpy.sin(ω_minus*t))
title = f"Analytic Soultion: γ={γ}, " + r"$t_{+}=$" + f"{format(t_plus, '2.5f')}, " + r"$t_{-}=$" + f"{format(t_minus, '2.5f')}"
verification_time_series(title, p, q, time, [-2.1, 2.1], "binvariate_normal_analytic_calculated_09_1")

# %%
H = 4.0*ω_minus**2*(CR**2+CI**2)
title = f"Analytic Soultion: H={format(H, '2.5f')}"
Kt = numpy.array([p(t)**2 for t in time])
Ut = numpy.array([ω_minus**2*q(t)**2 for t in time])

hmc.multicurve(title, [Kt, Ut, Kt+Ut], time, "Time", "Energy", ["K", "U", "H"], (0.8, 0.8), [-0.1, 2.0], "binvariate_normal_analytic_hamiltonian-timeseries-1")

# %%

Kt, Ut, Ht = total_energy(PQ, U, K)
title = f"Verification Soultion: H={format(Ht[0], '2.5f')}"
hmc.multicurve(title, [Kt, Ut, Ht], time, "Time", "Energy", ["K", "U", "H"], (0.8, 0.8), [-0.1, 2.0], "binvariate_normal_verification_hamiltonian-timeseries-1")

# %%
# Configuration

γ = 0.2
α = 1 / (1.0 - γ**2)
nsteps = 500

ω_plus = numpy.sqrt(α*(1.0 + γ))
ω_minus = numpy.sqrt(α*(1.0 - γ))
t_plus = 2.0*numpy.pi / numpy.abs(ω_plus)
t_minus = 2.0*numpy.pi / numpy.abs(ω_minus)

U = hmc.bivariate_normal_U(γ, 1.0, 1.0)
K = hmc.bivariate_normal_K(1.0, 1.0)

PQ0 = numpy.matrix([[-1.0], [-2.0], [1.0], [-1.0]])
time = numpy.linspace(0.0, 2.0*t_minus, nsteps)

# %%
# Compute coefficients for solution using intial conditions

λ = eigenvalues(γ, α)
E = eigenvector_matrix_unnormalized(γ, α)
UEinv = linalg.inv(UE)
C = UEinv * PQ0
C
