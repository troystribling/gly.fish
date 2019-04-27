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

def total_energy(PQ, U, K):
    npts = len(PQ)
    q = numpy.array([[numpy.real(PQ[i,2,0]), numpy.real(PQ[i,3,0])] for i in range(npts)])
    p = numpy.array([[numpy.real(PQ[i,0,0]), numpy.real(PQ[i,1,0])] for i in range(npts)])
    U_t = numpy.array([U(qt) for qt in q])
    K_t =  numpy.array([K(pt) for pt in p])
    return U_t + K_t

# %%
# Configuration

γ = 0.9
α = 1 / (1.0 - γ**2)

ω_plus = numpy.complex(0.0, numpy.sqrt(α*(1.0 + γ)))
ω_minus = numpy.complex(0.0, numpy.sqrt(α*(1.0 - γ)))
t_plus = 2.0*numpy.pi / numpy.abs(ω_plus)
t_minus = 2.0*numpy.pi / numpy.abs(ω_minus)

nsteps = 500

U = hmc.bivariate_normal_U(γ, 1.0, 1.0)
K = hmc.bivariate_normal_K(1.0, 1.0)

# %%
# Compute solutions using eigenvalues and eigenvectots computed from numercically diagonalizing Hamiltonian Matrix
PQ0 = numpy.matrix([[1.0], [1.0], [1.0], [1.0]])
time = numpy.linspace(0.0, 2.0*t_minus, nsteps)

# Diagonalize Hamiltonian Matrix
H = hamiltonian_matrix(γ, α)
λ, E = linalg.eig(H)
PQ = coordinate_time_series(E, PQ0, λ, time)
q = numpy.array([[numpy.real(PQ[i,2,0]), numpy.real(PQ[i,3,0])] for i in range(nsteps)])
p = numpy.array([[numpy.real(PQ[i,0,0]), numpy.real(PQ[i,1,0])] for i in range(nsteps)])

title = f"Numerical Soultion: γ={γ}, " + r"$t_{+}=$" + f"{format(t_plus, '2.4f')}, " + r"$t_{-}=$" + f"{format(t_minus, '2.4f')}"
phase_space_time_series(title, PQ, time, [-2.2, 2.2], "hamiltions-equations-analytic-solution-binvariate-normal-numerical-diag-phase-space-timeseries-1")

# %%

hmc.energy_time_series(title, U, K, p, q, time, (0.4, 0.9), [-0.1, 2.2], "hamiltions-equations-analytic-solution-binvariate-normal-numerical-diag-energy-timeseries-1")

# %%

hmc.phase_space_plot(p[:,0], q[:,0], title, [r"$q_1$", r"$p_1$"], (0.2, 0.7), "hamiltions-equations-analytic-solution-binvariate-normal-numerical-diag-phase-space-1")

# %%
# Compute solutions using eigenvalues and eigenvectots computed algebrically
PQ0 = numpy.matrix([[1.0], [1.0], [1.0], [1.0]])
time = numpy.linspace(0.0, 2.0*t_minus, nsteps)

λ = eigenvalues(γ, α)
E = eigenvector_matrix(γ, α)
PQ = coordinate_time_series(E, PQ0, λ, time)

q = numpy.array([[numpy.real(PQ[i,2,0]), numpy.real(PQ[i,3,0])] for i in range(nsteps)])
p = numpy.array([[numpy.real(PQ[i,0,0]), numpy.real(PQ[i,1,0])] for i in range(nsteps)])

title = f"Analytic Soultion: γ={γ}, " + r"$t_{+}=$" + f"{format(t_plus, '2.5f')}, " + r"$t_{-}=$" + f"{format(t_minus, '2.5f')}"
phase_space_time_series(title, PQ, time, [-4.5, 4.5], "hamiltions-equations-analytic-solution-binvariate-normal-phase-space-timeseries-2")

# %%

hmc.energy_time_series(title, U, K, p, q, time, (0.5, 0.9), [-0.1, 2.2], "hamiltions-equations-analytic-solution-binvariate-normal-energy-timeseries-2")

# %%

hmc.phase_space_plot(p[:,0], q[:,0], title, [r"$q_1$", r"$p_1$"], (0.2, 0.7), "hamiltions-equations-analytic-solution-binvariate-normal-phase-space-2")

# %%

time = numpy.linspace(0.0, 2.0*t_plus, nsteps)
PQ0 = numpy.matrix([[1.0], [-1.0], [1.0], [-1.0]])

λ = eigenvalues(γ, α)
E = eigenvector_matrix(γ, α)
PQ = coordinate_time_series(E, PQ0, λ, time)
q = numpy.array([[numpy.real(PQ[i,2,0]), numpy.real(PQ[i,3,0])] for i in range(nsteps)])
p = numpy.array([[numpy.real(PQ[i,0,0]), numpy.real(PQ[i,1,0])] for i in range(nsteps)])

title = f"Analytic Soultion: γ={γ}, " + r"$t_{+}=$" + f"{format(t_plus, '2.5f')}, " + r"$t_{-}=$" + f"{format(t_minus, '2.5f')}"
phase_space_time_series(title, PQ, time, [-5.5, 5.5], "hamiltions-equations-analytic-solution-binvariate-normal-phase-space-timeseries-3")

# %%

hmc.energy_time_series(title, U, K, p, q, time, (0.5, 0.775), [-0.1, 15.0], "hamiltions-equations-analytic-solution-binvariate-normal-energy-timeseries-3")

# %%

hmc.phase_space_plot(p[:,0], q[:,0], title, [r"$q_1$", r"$p_1$"], (0.2, 0.7), "hamiltions-equations-analytic-solution-binvariate-normal-phase-space-3")

# %%

time = numpy.linspace(0.0, 2.0*t_minus, nsteps)
PQ0 = numpy.matrix([[-1.0], [-2.0], [1.0], [-1.0]])
λ = eigenvalues(γ, α)
E = eigenvector_matrix(γ, α)
PQ = coordinate_time_series(E, PQ0, λ, time)

q = numpy.array([[numpy.real(PQ[i,2,0]), numpy.real(PQ[i,3,0])] for i in range(nsteps)])
p = numpy.array([[numpy.real(PQ[i,0,0]), numpy.real(PQ[i,1,0])] for i in range(nsteps)])

title = f"Analytic Soultion: γ={γ}, " + r"$t_{+}=$" + f"{format(t_plus, '2.5f')}, " + r"$t_{-}=$" + f"{format(t_minus, '2.5f')}"
phase_space_time_series(title, PQ, time, [-5.5, 5.5], "hamiltions-equations-analytic-solution-binvariate-normal-phase-space-timeseries-4")

# %%

hmc.energy_time_series(title, U, K, p, q, time, (0.5, 0.8), [-0.1, 17.0], "hamiltions-equations-analytic-solution-binvariate-normal-energy-timeseries-4")

# %%

hmc.phase_space_plot(p[:,0], q[:,0], title, [r"$q_1$", r"$p_1$"], (0.2, 0.85), "hamiltions-equations-analytic-solution-binvariate-normal-phase-space-4")

# %%

γ = 0.0
α = 1 / (1.0 - γ**2)

ω_plus = numpy.complex(0.0, numpy.sqrt(α*(1.0 + γ)))
ω_minus = numpy.complex(0.0, numpy.sqrt(α*(1.0 - γ)))
t_plus = 2.0*numpy.pi / numpy.abs(ω_plus)
t_minus = 2.0*numpy.pi / numpy.abs(ω_minus)

nsteps = 500

U = hmc.bivariate_normal_U(γ, 1.0, 1.0)
K = hmc.bivariate_normal_K(1.0, 1.0)

# %%

time = numpy.linspace(0.0, 2.0*t_minus, nsteps)
PQ0 = numpy.matrix([[-1.0], [-2.0], [1.0], [-1.0]]) # initial conditions
λ = eigenvalues(γ, α)
E = eigenvector_matrix(γ, α)
PQ = coordinate_time_series(E, PQ0, λ, time)

q = numpy.array([[numpy.real(PQ[i,2,0]), numpy.real(PQ[i,3,0])] for i in range(nsteps)])
p = numpy.array([[numpy.real(PQ[i,0,0]), numpy.real(PQ[i,1,0])] for i in range(nsteps)])

title = f"Analytic Soultion: γ={γ}, " + r"$t_{+}=$" + f"{format(t_plus, '2.5f')}, " + r"$t_{-}=$" + f"{format(t_minus, '2.5f')}"
phase_space_time_series(title, PQ, time, [-3.5, 3.5], "hamiltions-equations-analytic-solution-binvariate-normal-phase-space-timeseries-5")

# %%

hmc.energy_time_series(title, U, K, p, q, time, (0.5, 0.775), [-0.1, 5.0], "hamiltions-equations-analytic-solution-binvariate-normal-energy-timeseries-5")

# %%

hmc.phase_space_plot(p[:,0], q[:,0], title, [r"$q_1$", r"$p_1$"], (0.2, 0.85), "hamiltions-equations-analytic-solution-binvariate-normal-phase-space-5")

# %%

γ = 0.2
α = 1 / (1.0 - γ**2)

ω_plus = numpy.complex(0.0, numpy.sqrt(α*(1.0 + γ)))
ω_minus = numpy.complex(0.0, numpy.sqrt(α*(1.0 - γ)))
t_plus = 2.0*numpy.pi / numpy.abs(ω_plus)
t_minus = 2.0*numpy.pi / numpy.abs(ω_minus)

nsteps = 500

U = hmc.bivariate_normal_U(γ, 1.0, 1.0)
K = hmc.bivariate_normal_K(1.0, 1.0)

# %%

time = numpy.linspace(0.0, 5.0*t_minus, nsteps)
PQ0 = numpy.matrix([[-1.0], [-2.0], [1.0], [-1.0]]) # initial conditions
λ = eigenvalues(γ, α)
E = eigenvector_matrix(γ, α)
PQ = coordinate_time_series(E, PQ0, λ, time)

q = numpy.array([[numpy.real(PQ[i,2,0]), numpy.real(PQ[i,3,0])] for i in range(nsteps)])
p = numpy.array([[numpy.real(PQ[i,0,0]), numpy.real(PQ[i,1,0])] for i in range(nsteps)])

title = f"Analytic Soultion: γ={γ}, " + r"$t_{+}=$" + f"{format(t_plus, '2.5f')}, " + r"$t_{-}=$" + f"{format(t_minus, '2.5f')}"
phase_space_time_series(title, PQ, time, [-3.5, 3.5], "hamiltions-equations-analytic-solution-binvariate-normal-phase-space-timeseries-6")

# %%

hmc.energy_time_series(title, U, K, p, q, time, (0.5, 0.775), [-0.1, 5.0], "hamiltions-equations-analytic-solution-binvariate-normal-energy-timeseries-6")

# %%

hmc.phase_space_plot(p[:,0], q[:,0], title, [r"$q_1$", r"$p_1$"], (0.2, 0.85), "hamiltions-equations-analytic-solution-binvariate-normal-phase-space-5")
