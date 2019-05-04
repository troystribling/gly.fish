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

def total_energy(p, q, U, K):
    Ut = numpy.array([U([q[0][i], q[1][i]]) for i in range(len(q[0]))])
    Kt =  numpy.array([K([p[0][i], p[1][i]]) for i in range(len(p[0]))])
    return Kt, Ut, Ut + Kt

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

q1_error = numpy.abs(q_int[:,0] - q1t)
q2_error = numpy.abs(q_int[:,1] - q2t)
p1_error = numpy.abs(p_int[:,0] - p1t)
p2_error = numpy.abs(p_int[:,1] - p2t)

q1_total_error = 100.0*numpy.sum(q1_error)/numpy.sum(numpy.abs(q1t))
q2_total_error = 100.0*numpy.sum(q2_error)/numpy.sum(numpy.abs(q2t))
p1_total_error = 100.0*numpy.sum(p1_error)/numpy.sum(numpy.abs(p1t))
p2_total_error = 100.0*numpy.sum(p2_error)/numpy.sum(numpy.abs(p2t))

# %%

title = f"Integration Error: γ={γ}, " + r"$\langle\Delta q_1\rangle$=" + f"{format(q1_total_error, '2.2f')}%, " + r"$\langle\Delta q_2\rangle$=" + f"{format(q2_total_error, '2.2f')}%"
hmc.multicurve(title, [q1_error, q2_error], time, "Time", r"$\mid\Delta q\mid$", [r"$q_1$", r"$q_2$"],  (0.15, 0.8), [0.0, 0.05], "hamiltonian-integration-error-bivariate-normal-position-error-09-1")

# %%

title = f"Integration Error: γ={γ}, " + r"$\langle\Delta p_1\rangle$=" + f"{format(p1_total_error, '2.2f')}% , " + r"$\langle\Delta p_2\rangle$=" + f"{format(p2_total_error, '2.2f')}%"
hmc.multicurve(title, [p1_error, p2_error], time, "Time", r"$\mid\Delta p\mid$", [r"$p_1$", r"$p_2$"],  (0.15, 0.8), [0.0, 0.075], "hamiltonian-integration-error-bivariate-normal-intgrated-momentum-phase-space-09-1")

# %%

Kt, Ut, Ht = total_energy([p1t, p2t], [q1t, q2t], U, K)
K_int, Ut_int, Ht_int = total_energy([p_int[:,0], p_int[:,1]], [q_int[:,0], q_int[:,1]], U, K)

H_error = numpy.abs(Ht - Ht_int)
U_error = numpy.abs(Ut - Ut_int)
K_error = numpy.abs(K_int - Kt)

H_total_error = 100.0*numpy.sum(H_error)/numpy.sum(numpy.abs(Ht))
U_total_error = 100.0*numpy.sum(U_error)/numpy.sum(numpy.abs(Ut))
K_total_error = 100.0*numpy.sum(K_error)/numpy.sum(numpy.abs(Kt))

title = f"Integration Error: γ={γ}, " + r"$\langle\Delta H\rangle$=" + f"{format(H_total_error, '2.2f')}% , " + r"$\langle\Delta U\rangle$=" + f"{format(U_total_error, '2.2f')}%, " + r"$\langle\Delta K\rangle$=" + f"{format(K_total_error, '2.2f')}%"
hmc.multicurve(title, [K_error, U_error, H_error], time, "Time", r"$\mid\Delta E\mid$", ["K", "U", "H"], (0.5, 0.85), [-0.01, 0.2], "hamiltonian-integration-error-bivariate-normal-energy-1", 3)

# %%
# Configuration

γ = 0.0
α = 1 / (1.0 - γ**2)
nsteps = 1257

ω_plus = numpy.sqrt(α*(1.0 + γ))
ω_minus = numpy.sqrt(α*(1.0 - γ))
t_plus = 2.0*numpy.pi / numpy.abs(ω_plus)
t_minus = 2.0*numpy.pi / numpy.abs(ω_minus)

U = hmc.bivariate_normal_U(γ, 1.0, 1.0)
K = hmc.bivariate_normal_K(1.0, 1.0)

PQ0 = numpy.matrix([[1.0], [-1.0], [1.0], [-1.0]])
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

p1 = lambda t: -2.0*ω_plus*(CR*numpy.sin(ω_plus*t) + CI*numpy.cos(ω_plus*t))
p2 = lambda t: -p1(t)
q1 = lambda t: 2.0*(CR*numpy.cos(ω_plus*t) - CI*numpy.sin(ω_plus*t))
q2 = lambda t: -q1(t)

title = f"Analytic Solution: γ={γ}, " + r"$t_{+}=$" + f"{format(t_plus, '2.5f')}, " + r"$t_{-}=$" + f"{format(t_minus, '2.5f')}"
verification_time_series(title, [p1, p2], [q1, q2], time, [-2.0, 2.0], "hamiltonian-integration-error-bivariate-normal-analytic-phase-space-09-2")

# %%

q1t = [q1(t) for t in time]
q2t = [q2(t) for t in time]
p1t = [p1(t) for t in time]
p2t = [p2(t) for t in time]

# %%
# Integration parameters

p0 = numpy.array([1.0, -1.0])
q0 = [1.0, -1.0]

m1 = 1.0
m2 = 1.0

σ1 = 1.0
σ2 = 1.0

γ = 0.0
α = 1 / (1.0 - γ**2)

ω_plus = numpy.complex(0.0, numpy.sqrt(α*(1.0 + γ)))
ω_minus = numpy.complex(0.0, numpy.sqrt(α*(1.0 - γ)))
t_plus = 2.0*numpy.pi / numpy.abs(ω_plus)
t_minus = 2.0*numpy.pi / numpy.abs(ω_minus)

ε = 0.01
nsteps = int(2.0*t_minus/ε)
time = numpy.linspace(0.0, 6.0*t_minus, nsteps+1)

U = he.bivariate_normal_U(γ, σ1, σ2)
K = he.bivariate_normal_K(m1, m2)
dUdq = he.bivariate_normal_dUdq(γ, σ1, σ2)
dKdp = he.bivariate_normal_dKdp(m1, m2)

# %%

p_int, q_int = he.momentum_verlet(p0, q0, dUdq, dKdp, nsteps, ε)
title = r"$\sigma_1=$" f"{σ1}, " + r"$\sigma_2=$" f"{σ2}, " + f"γ={γ}, Δt={ε}, steps={nsteps}"
hmc.multicurve(title, [q_int[:,0], q_int[:,1]], time, "Time", "q", [r"$q_1$", r"$q_2$"],  (0.2, 0.8), [-2.0, 2.0], "hamiltonian-integration-error-bivariate-normal-intgrated-position-phase-space-09-2")

# %%

hmc.multicurve(title, [p_int[:,0], p_int[:,1]], time, "Time", "p", [r"$p_1$", r"$p_2$"],  (0.15, 0.8), [-2.0, 2.0], "hamiltonian-integration-error-bivariate-normal-intgrated-momentum-phase-space-09-1")

# %%

q1_error = numpy.abs(q_int[:,0] - q1t)
q2_error = numpy.abs(q_int[:,1] - q2t)
p1_error = numpy.abs(p_int[:,0] - p1t)
p2_error = numpy.abs(p_int[:,1] - p2t)

q1_total_error = 100.0*numpy.sum(q1_error)/numpy.sum(numpy.abs(q1t))
q2_total_error = 100.0*numpy.sum(q2_error)/numpy.sum(numpy.abs(q2t))
p1_total_error = 100.0*numpy.sum(p1_error)/numpy.sum(numpy.abs(p1t))
p2_total_error = 100.0*numpy.sum(p2_error)/numpy.sum(numpy.abs(p2t))

# %%

title = f"Integration Error: γ={γ}, " + r"$\langle\Delta q_1\rangle$=" + f"{format(q1_total_error, '2.2f')}%, " + r"$\langle\Delta q_2\rangle$=" + f"{format(q2_total_error, '2.2f')}%"
hmc.multicurve(title, [q1_error, q2_error], time, "Time", r"$\mid\Delta q\mid$", [r"$q_1$", r"$q_2$"],  (0.15, 0.8), [-0.001, 0.01], "hamiltonian-integration-error-bivariate-normal-position-error-09-2")

# %%

title = f"Integration Error: γ={γ}, " + r"$\langle\Delta p_1\rangle$=" + f"{format(p1_total_error, '2.2f')}% , " + r"$\langle\Delta p_2\rangle$=" + f"{format(p2_total_error, '2.2f')}%"
hmc.multicurve(title, [p1_error, p2_error], time, "Time", r"$\mid\Delta p\mid$", [r"$p_1$", r"$p_2$"],  (0.15, 0.8), [-0.001, 0.01], "hamiltonian-integration-error-bivariate-normal-intgrated-momentum-error-09-2")

# %%

Kt, Ut, Ht = total_energy([p1t, p2t], [q1t, q2t], U, K)
K_int, Ut_int, Ht_int = total_energy([p_int[:,0], p_int[:,1]], [q_int[:,0], q_int[:,1]], U, K)

H_error = numpy.abs(Ht - Ht_int)
U_error = numpy.abs(Ut - Ut_int)
K_error = numpy.abs(K_int - Kt)

H_total_error = 100.0*numpy.sum(H_error)/numpy.sum(numpy.abs(Ht))
U_total_error = 100.0*numpy.sum(U_error)/numpy.sum(numpy.abs(Ut))
K_total_error = 100.0*numpy.sum(K_error)/numpy.sum(numpy.abs(Kt))

title = f"Integration Error: γ={γ}, " + r"$\langle\Delta H\rangle$=" + f"{format(H_total_error, '2.2f')}% , " + r"$\langle\Delta U\rangle$=" + f"{format(U_total_error, '2.2f')}%, " + r"$\langle\Delta K\rangle$=" + f"{format(K_total_error, '2.2f')}%"
hmc.multicurve(title, [K_error, U_error, H_error], time, "Time", r"$\mid\Delta E\mid$", ["K", "U", "H"], (0.5, 0.85), [-0.001, 0.015], "hamiltonian-integration-error-bivariate-normal-energy-2", 3)
