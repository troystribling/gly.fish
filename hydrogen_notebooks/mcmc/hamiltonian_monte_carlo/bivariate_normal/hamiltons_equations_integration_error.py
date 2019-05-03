# %%
%load_ext autoreload
%autoreload 2

import numpy
from numpy import linalg
from matplotlib import pyplot
from glyfish import config
from glyfish import hamiltonian_monte_carlo as hmc
from glyfish import hamiltons_equations as he

%matplotlib inline

pyplot.style.use(config.glyfish_style)

# %%

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

def verification_time_series(title, p, q, time, ylim, file):
    nplots = 2
    figure, axis = pyplot.subplots(nrows=nplots, ncols=1, sharex=True, figsize=(10, 4*nplots))
    axis[0].set_title(title)
    axis[-1].set_xlabel("Time")

    q1 = [q[0](t) for t in time]
    q2 = [q[1](t) for t in time]
    axis[0].set_xlim([time[0], time[-1]])
    axis[0].set_ylim(ylim)
    axis[0].set_ylabel("q")
    axis[0].plot(time, q1, label=r"$q_1$")
    axis[0].plot(time, q2, label=r"$q_2$")
    axis[0].legend()

    p1 = [p[0](t) for t in time]
    p2 = [p[1](t) for t in time]
    axis[1].set_xlim([time[0], time[-1]])
    axis[1].set_ylim(ylim)
    axis[1].set_ylabel("p")
    axis[1].plot(time, p1, label=r"$p_1$")
    axis[1].plot(time, p2, label=r"$p_2$")
    axis[1].legend()

    config.save_post_asset(figure, "hamiltonian_monte_carlo", file)

# %%
# Configuration

γ = 0.9
α = 1 / (1.0 - γ**2)
nsteps = 1733

ω_plus = numpy.sqrt(α*(1.0 + γ))
ω_minus = numpy.sqrt(α*(1.0 - γ))
t_plus = 2.0*numpy.pi / numpy.abs(ω_plus)
t_minus = 2.0*numpy.pi / numpy.abs(ω_minus)

U = hmc.bivariate_normal_U(γ, 1.0, 1.0)
K = hmc.bivariate_normal_K(1.0, 1.0)

PQ0 = numpy.matrix([[-1.0], [-2.0], [1.0], [-1.0]])
time = numpy.linspace(0.0, 2.0*t_minus, nsteps)

# %%
# Compute coefficients for solution using intial conditions and plot analytic solutions
λ = eigenvalues(γ, α)
E = eigenvector_matrix_unnormalized(γ, α)
Einv = linalg.inv(E)
C = Einv * PQ0

CR = numpy.real(C[0,0])
CI = numpy.imag(C[0,0])
PI = numpy.imag(C[2,0])

# %%

p1 = lambda t: -2.0*ω_plus*(CR*numpy.sin(ω_plus*t) + CI*numpy.cos(ω_plus*t)) - 2.0*ω_minus*PI*numpy.cos(ω_minus*t)
p2 = lambda t: 2.0*ω_plus*(CR*numpy.sin(ω_plus*t) + CI*numpy.cos(ω_plus*t)) - 2.0*ω_minus*PI*numpy.cos(ω_minus*t)
q1 = lambda t: 2.0*(CR*numpy.cos(ω_plus*t) - CI*numpy.sin(ω_plus*t) - PI*numpy.sin(ω_minus*t))
q2 = lambda t: -2.0*(CR*numpy.cos(ω_plus*t) - CI*numpy.sin(ω_plus*t) + PI*numpy.sin(ω_minus*t))

title = f"Analytic Solution: γ={γ}, " + r"$t_{+}=$" + f"{format(t_plus, '2.5f')}, " + r"$t_{-}=$" + f"{format(t_minus, '2.5f')}"
verification_time_series(title, [p1, p2], [q1, q2], time, [-5.0, 5.0], "hamiltonian-integration-error-bivariate-normal-analytic-phase-space-09-1")

# %%

q1t = [q1(t) for t in time]
q2t = [q2(t) for t in time]
p1t = [p1(t) for t in time]
p2t = [p2(t) for t in time]

# %%
# Integration parameters

p0 = numpy.array([-1.0, -2.0])
q0 = [1.0, -1.0]

m1 = 1.0
m2 = 1.0

σ1 = 1.0
σ2 = 1.0

γ = 0.9
α = 1 / (1.0 - γ**2)

ω_plus = numpy.complex(0.0, numpy.sqrt(α*(1.0 + γ)))
ω_minus = numpy.complex(0.0, numpy.sqrt(α*(1.0 - γ)))
t_plus = 2.0*numpy.pi / numpy.abs(ω_plus)
t_minus = 2.0*numpy.pi / numpy.abs(ω_minus)

ε = 0.01
nsteps = int(2.0*t_minus/ε)
time = numpy.linspace(0.0, 2.0*t_minus, nsteps+1)

U = he.bivariate_normal_U(γ, σ1, σ2)
K = he.bivariate_normal_K(m1, m2)
dUdq = he.bivariate_normal_dUdq(γ, σ1, σ2)
dKdp = he.bivariate_normal_dKdp(m1, m2)

# %%

p_int, q_int = he.momentum_verlet(p0, q0, dUdq, dKdp, nsteps, ε)
title = r"$\sigma_1=$" f"{σ1}, " + r"$\sigma_2=$" f"{σ2}, " + f"γ={γ}, Δt={ε}, steps={nsteps}"
hmc.multicurve(title, [q_int[:,0], q_int[:,1]], time, "Time", "q", [r"$q_1$", r"$q_2$"],  (0.2, 0.9), [-4.25, 4.25], "hamiltonian-integration-error-bivariate-normal-intgrated-position-phase-space-09-1")

# %%

hmc.multicurve(title, [p_int[:,0], p_int[:,1]], time, "Time", "p", [r"$p_1$", r"$p_2$"],  (0.15, 0.8), [-6.0, 6.0], "hamiltonian-integration-error-bivariate-normal-intgrated-momentum-phase-space-09-1")

# %%

q1_error = 100.0*numpy.abs((q_int[:,0] - q1t)/q1t)
q2_error = 100.0*numpy.abs((q_int[:,1] - q2t)/q2t)
p1_error = 100.0*numpy.abs((p_int[:,0] - p1t)/p1t)
p2_error = 100.0*numpy.abs((p_int[:,1] - p2t)/p2t)

# %%

hmc.multicurve(title, [q1_error, q2_error], time, "Time", "q", [r"$q_1$", r"$q_2$"],  (0.15, 0.8), [-0.1, 100.0], "hamiltonian-integration-error-bivariate-normal-intgrated-momentum-phase-space-09-1")

# %%

hmc.multicurve(title, [p1_error, p2_error], time, "Time", "q", [r"$p_1$", r"$p_2$"],  (0.15, 0.8), [-0.1, 500.0], "hamiltonian-integration-error-bivariate-normal-intgrated-momentum-phase-space-09-1")
