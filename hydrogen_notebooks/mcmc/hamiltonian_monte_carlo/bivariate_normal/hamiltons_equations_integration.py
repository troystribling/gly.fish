%load_ext autoreload
%autoreload 2

import numpy
from matplotlib import pyplot
from glyfish import config
from glyfish import gplot
from glyfish import hamiltonian_monte_carlo as hmc
from glyfish import stats
from glyfish import hamiltons_equations as he

%matplotlib inline

pyplot.style.use(config.glyfish_style)

# %%
# Integration parameters

t = 2.0*numpy.pi
ε = 0.1
nsteps = int(t/ε)

p0 = numpy.array([1.0, -1.0])
q0 = numpy.array([1.0, -1.0])

m1 = 1.0
m2 = 1.0

σ1 = 1.0
σ2 = 1.0
γ = 0.0

U = he.bivariate_normal_U(γ, σ1, σ2)
K = he.bivariate_normal_K(m1, m2)

dUdq = he.bivariate_normal_dUdq(γ, σ1, σ2)
dKdp = he.bivariate_normal_dKdp(m1, m2)
time = numpy.linspace(0.0, t, nsteps+1)

# %%

p, q = he.momentum_verlet(p0, q0, dUdq, dKdp, nsteps, ε)
title = r"$\sigma_1=$" f"{σ1}, " + r"$\sigma_2=$" f"{σ2}, " + f"γ={γ}, Δt={ε}, steps={nsteps}"
hmc.phase_space_plot(p[:,0], q[:,0], title, [r"$q_1$", r"$p_1$"], (0.2, 0.7), "hamiltonian-integration-bivariate-normal-phase-space-plot-1")

# %%

hmc.phase_space_plot(p[:,1], q[:,1], title, [r"$q_2$", r"$p_2$"], (0.2, 0.7), "hamiltonian-integration-bivariate-normal-phase-space-plot-2")

# %%

hmc.phase_space_plot(q[:,0], q[:,1], title, [r"$q_1$", r"$q_2$"], (0.3, 0.9), "hamiltonian-integration-bivariate-normal-phase-space-plot-3")

# %%

hmc.phase_space_plot(p[:,0], p[:,1], title, [r"$p_1$", r"$p_2$"], (0.3, 0.9), "hamiltonian-integration-bivariate-normal-phase-space-plot-4")

# %%

hmc.multicurve(title, [q[:,0], q[:,1]], time, "Time", "q", [r"$q_1$", r"$q_2$"],  (0.8, 0.8), [-2.0, 2.0], "hamiltonian-integration-bivariate-normal-position-timeseries-1")

# %%

hmc.multicurve(title, [p[:,0], p[:,1]], time, "Time", "p", [r"$p_1$", r"$p_2$"],  (0.3, 0.85), [-2.0, 2.0], "hamiltonian-integration-bivariate-normal-momentum-timeseries-1")

# %%

hmc.energy_time_series(title, U, K, p, q, time, (0.5, 0.85), [-0.1, 2.5], "hamiltonian-integration-bivariate-normal-energy-timeseries-1")

# %%
# Integration parameters

p0 = numpy.array([1.0, 1.0])
q0 = numpy.array([1.0, -1.0])

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

p, q = he.momentum_verlet(p0, q0, dUdq, dKdp, nsteps, ε)
title = r"$\sigma_1=$" f"{σ1}, " + r"$\sigma_2=$" f"{σ2}, " + f"γ={γ}, Δt={ε}, steps={nsteps}"
hmc.phase_space_plot(p[:,0], q[:,0], title, [r"$q_1$", r"$p_1$"], (0.2, 0.7), "hamiltonian-integration-bivariate-normal-phase-space-plot-5")

# %%

hmc.phase_space_plot(p[:,1], q[:,1], title, [r"$q_2$", r"$p_2$"], (0.225, 0.85), "hamiltonian-integration-bivariate-normal-phase-space-plot-6")

# %%

hmc.phase_space_plot(q[:,0], q[:,1], title, [r"$q_1$", r"$q_2$"], (0.9, 0.9), "hamiltonian-integration-bivariate-normal-phase-space-plot-7")

# %%

hmc.phase_space_plot(p[:,0], p[:,1], title, [r"$p_1$", r"$p_2$"], (0.8, 0.9), "hamiltonian-integration-bivariate-normal-phase-space-plot-8")

# %%

hmc.multicurve(title, [q[:,0], q[:,1]], time, "Time", "q", [r"$q_1$", r"$q_2$"],  (0.15, 0.8), [-4.25, 4.25], "hamiltonian-integration-bivariate-normal-position-timeseries-2")

# %%

hmc.multicurve(title, [p[:,0], p[:,1]], time, "Time", "p", [r"$p_1$", r"$p_2$"],  (0.2, 0.9), [-5.25, 5.25], "hamiltonian-integration-bivariate-normal-momentum-timeseries-2")

# %%

hmc.energy_time_series(title, U, K, p, q, time, (0.7, 0.775), [-0.1, 15.0], "hamiltonian-integration-bivariate-normal-energy-timeseries-2")

# %%
# Integration parameters
p0 = numpy.array([1.0, -1.0])
q0 = numpy.array([1.0, -1.0])

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
nsteps = int(2.0*t_plus/ε)
time = numpy.linspace(0.0, 2.0*t_plus, nsteps+1)

U = he.bivariate_normal_U(γ, σ1, σ2)
K = he.bivariate_normal_K(m1, m2)

dUdq = he.bivariate_normal_dUdq(γ, σ1, σ2)
dKdp = he.bivariate_normal_dKdp(m1, m2)

# %%

p, q = he.momentum_verlet(p0, q0, dUdq, dKdp, nsteps, ε)
title = r"$\sigma_1=$" f"{σ1}, " + r"$\sigma_2=$" f"{σ2}, " + f"γ={γ}, Δt={ε}, steps={nsteps}"
hmc.phase_space_plot(p[:,0], q[:,0], title, [r"$q_1$", r"$p_1$"], (0.8, 0.85), "hamiltonian-integration-bivariate-normal-phase-space-plot-9")

# %%

hmc.multicurve(title, [q[:,0], q[:,1]], time, "Time", "q", [r"$q_1$", r"$q_2$"],  (0.15, 0.8), [-2.25, 2.25], "hamiltonian-integration-bivariate-normal-position-timeseries-3")

# %%

hmc.multicurve(title, [p[:,0], p[:,1]], time, "Time", "p", [r"$p_1$", r"$p_2$"],  (0.2, 0.9), [-5.0, 5.0], "hamiltonian-integration-bivariate-normal-momentum-timeseries-3")

# %%

hmc.energy_time_series(title, U, K, p, q, time, (0.7, 0.775), [-0.1, 15.0], "hamiltonian-integration-bivariate-normal-energy-timeseries-3")

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

p, q = he.momentum_verlet(p0, q0, dUdq, dKdp, nsteps, ε)
title = r"$\sigma_1=$" f"{σ1}, " + r"$\sigma_2=$" f"{σ2}, " + f"γ={γ}, Δt={ε}, steps={nsteps}"
hmc.phase_space_plot(p[:,0], q[:,0], title, [r"$q_1$", r"$p_1$"], (0.225, 0.85), "hamiltonian-integration-bivariate-normal-phase-space-plot-13")

 # %%

hmc.multicurve(title, [q[:,0], q[:,1]], time, "Time", "q", [r"$q_1$", r"$q_2$"],  (0.2, 0.9), [-4.25, 4.25], "hamiltonian-integration-bivariate-normal-position-timeseries-4")

# %%

hmc.multicurve(title, [p[:,0], p[:,1]], time, "Time", "p", [r"$p_1$", r"$p_2$"],  (0.15, 0.8), [-6.0, 6.0], "hamiltonian-integration-bivariate-normal-momentum-timeseries-4")

# %%

hmc.energy_time_series(title, U, K, p, q, time, (0.5, 0.8), [-0.1, 17.0], "hamiltonian-integration-bivariate-normal-energy-timeseries-4")

# %%
# Integration parameters

p0 = numpy.array([-1.0, -2.0])
q0 = numpy.array([1.0, -1.0])

m1 = 1.0
m2 = 1.0

σ1 = 1.0
σ2 = 1.0

γ = 0.2
α = 1 / (1.0 - γ**2)

ω_plus = numpy.complex(0.0, numpy.sqrt(α*(1.0 + γ)))
ω_minus = numpy.complex(0.0, numpy.sqrt(α*(1.0 - γ)))
t_plus = 2.0*numpy.pi / numpy.abs(ω_plus)
t_minus = 2.0*numpy.pi / numpy.abs(ω_minus)

ε = 0.01
nsteps = int(6.0*t_minus/ε)
time = numpy.linspace(0.0, 2.0*t_minus, nsteps+1)

U = he.bivariate_normal_U(γ, σ1, σ2)
K = he.bivariate_normal_K(m1, m2)
dUdq = he.bivariate_normal_dUdq(γ, σ1, σ2)
dKdp = he.bivariate_normal_dKdp(m1, m2)

# %%

p, q = he.momentum_verlet(p0, q0, dUdq, dKdp, nsteps, ε)
title = r"$\sigma_1=$" f"{σ1}, " + r"$\sigma_2=$" f"{σ2}, " + f"γ={γ}, Δt={ε}, steps={nsteps}"
hmc.phase_space_plot(p[:,0], q[:,0], title, [r"$q_1$", r"$p_1$"], (0.85, 0.2), "hamiltonian-integration-bivariate-normal-phase-space-plot-17")

# %%

hmc.multicurve(title, [q[:,0], q[:,1]], time, "Time", "q", [r"$q_1$", r"$q_2$"],  (0.2, 0.9), [-4.2, 4.2], "hamiltonian-integration-bivariate-normal-position-timeseries-5")

# %%

hmc.multicurve(title, [p[:,0], p[:,1]], time, "Time", "p", [r"$p_1$", r"$p_2$"],  (0.2, 0.9), [-3.2, 3.2], "hamiltonian-integration-bivariate-normal-momentum-timeseries-5")

# %%

hmc.energy_time_series(title, U, K, p, q, time, (0.5, 0.8), [-0.1, 5.0], "hamiltonian-integration-bivariate-normal-energy-timeseries-5")

# %%
# Integration parameters

p0 = numpy.array([-0.35686864, -0.88875008])
q0 = numpy.array([1.0, -1.0])

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
nsteps = int(t_minus/ε)
time = numpy.linspace(0.0, t_minus/ε, nsteps+1)

U = he.bivariate_normal_U(γ, σ1, σ2)
K = he.bivariate_normal_K(m1, m2)
dUdq = he.bivariate_normal_dUdq(γ, σ1, σ2)
dKdp = he.bivariate_normal_dKdp(m1, m2)

# %%

p, q = he.momentum_verlet(p0, q0, dUdq, dKdp, nsteps, ε)
title = r"$\sigma_1=$" f"{σ1}, " + r"$\sigma_2=$" f"{σ2}, " + f"γ={γ}, Δt={ε}, steps={nsteps}"
hmc.phase_space_plot(p[:,0], q[:,0], title, [r"$q_1$", r"$p_1$"], (0.85, 0.2), "hamiltonian-integration-bivariate-normal-phase-space-plot-17")

# %%

hmc.multicurve(title, [q[:,0], q[:,1]], time, "Time", "q", [r"$q_1$", r"$q_2$"],  (0.2, 0.9), [-4.2, 4.2], "hamiltonian-integration-bivariate-normal-position-timeseries-5")

# %%

hmc.multicurve(title, [p[:,0], p[:,1]], time, "Time", "p", [r"$p_1$", r"$p_2$"],  (0.2, 0.9), [-3.2, 3.2], "hamiltonian-integration-bivariate-normal-momentum-timeseries-5")

# %%

hmc.energy_time_series(title, U, K, p, q, time, (0.5, 0.8), [-0.1, 2.0], "hamiltonian-integration-bivariate-normal-energy-timeseries-5")
