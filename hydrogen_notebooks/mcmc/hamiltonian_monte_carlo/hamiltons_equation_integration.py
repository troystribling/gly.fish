%load_ext autoreload
%autoreload 2

import numpy
from matplotlib import pyplot
from glyfish import config
from glyfish import hamiltonian_monte_carlo as hmc

%matplotlib inline

pyplot.style.use(config.glyfish_style)

# %%
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

# %%

def integration_error_plot(p0, q0, nsteps, ε, dUdq, dKdp, plot_file):
    p_euler_cromer, q_euler_cromer = euler_cromer(p0, q0, dUdq, dKdp, nsteps, ε)
    p_euler, q_euler = euler(p0, q0, dUdq, dKdp, nsteps, ε,)
    p_momentum_verlet, q_momentum_verlet = momentum_verlet(p0, q0, dUdq, dKdp, nsteps, ε)

    error_euler_cromer = 100.0*numpy.abs(numpy.sqrt(p_euler_cromer[1:-1]**2 + q_euler_cromer[1:-1]**2) - numpy.sqrt(2.0))/numpy.sqrt(2.0)
    error_euler = 100.0*numpy.abs(numpy.sqrt(p_euler[1:-1]**2 + q_euler[1:-1]**2) - numpy.sqrt(2.0))/numpy.sqrt(2.0)
    error_momentum_verlet = 100.0*numpy.abs(numpy.sqrt(p_momentum_verlet[1:-1]**2 + q_momentum_verlet[1:-1]**2) - numpy.sqrt(2.0))/numpy.sqrt(2.0)

    t_plot = numpy.linspace(0.0, nsteps*ε, nsteps+1)

    figure, axis = pyplot.subplots(figsize=(10, 7))
    axis.set_xlabel("t")
    axis.set_ylabel("% Error")
    axis.set_title(f"Hamiltion's Equations Integration Error: Δt={ε}, nsteps={nsteps}")
    axis.set_xlim([0.0, t])
    axis.plot(t_plot[1:-1], error_euler_cromer, label="Euler Cromer")
    axis.plot(t_plot[1:-1], error_euler, label="Euler")
    axis.plot(t_plot[1:-1], error_momentum_verlet, label="Momentum Verlet")
    config.save_post_asset(figure, "hamiltonian_monte_carlo", plot_file)
    axis.legend(bbox_to_anchor=(0.35, 0.9))

# %%
# Here Hamiltonian Monte Carlo is used to generate samples for single variable
# Unit Normal distribution. The Potential and Kinetic Energy are given by assuming a mass of 1

U = lambda q: q**2/2.0
K = lambda p: p**2/2.0
dUdq = lambda q: q
dKdp = lambda p: p

# %%

hmc.canonical_distribution_contour_plot(K, U, [0.02, 0.1, 0.2, 0.4, 0.6, 0.8], "Canonical Distribution", "gaussian_potential_energy")

# %%

t = 5.0
ε = 0.1
nsteps = int(t/ε)

p, q = euler(-1.0, 1.0, dUdq, dKdp, nsteps, ε)
title = f"Hamilton's Equations (Euler Method): Δt={ε}, nsteps={nsteps}"
hmc.hamiltons_equations_integration_plot(K, U, 0.37, p, q, title, (0.3, 0.95), "hamiltons_equations_integration_euler_method_01_5")

# %%

t = 15.0
ε = 0.1
nsteps = int(t/ε)

p, q = euler(-1.0, 1.0, dUdq, dKdp, nsteps, ε)
title = f"Hamilton's Equations (Euler Method): Δt={ε}, nsteps={nsteps}"
hmc.hamiltons_equations_integration_plot(K, U, 0.37, p, q, title, (0.275, 0.8), "hamiltons_equations_integration_euler_method_01_15")

# %%

t = 5.0
ε = 0.01
nsteps = int(t/ε)

p, q = euler(-1.0, 1.0, dUdq, dKdp, nsteps, ε)
title = f"Hamilton's Equations (Euler Method): Δt={ε}, nsteps={nsteps}"
hmc.hamiltons_equations_integration_plot(K, U, 0.37, p, q, title, (0.3, 0.95), "hamiltons_equations_integration_euler_method_001_5")

# %%

t = 125.0
ε = 0.01
nsteps = int(t/ε)

p, q = euler(-1.0, 1.0, dUdq, dKdp, nsteps, ε)
title = f"Hamilton's Equations (Euler Method): Δt={ε}, nsteps={nsteps}"
hmc.hamiltons_equations_integration_plot(K, U, 0.37, p, q, title, (0.25, 0.8), "hamiltons_equations_integration_euler_method_001_50")

# %%

t = 5.0
ε = 0.1
nsteps = int(t/ε)

p, q = euler_cromer(-1.0, 1.0, dUdq, dKdp, nsteps, ε)
title = f"Hamilton's Equations (Modified Euler Method): Δt={ε}, nsteps={nsteps}"
hmc.hamiltons_equations_integration_plot(K, U, 0.37, p, q, title, (0.3, 0.95), "hamiltons_equations_integration_modified_euler_method_01_5")

# %%

t = 15.0
ε = 0.1
nsteps = int(t/ε)

p, q = euler_cromer(-1.0, 1.0, dUdq, dKdp, nsteps, ε)
title = f"Hamilton's Equations (Modified Euler Method): Δt={ε}, nsteps={nsteps}"
hmc.hamiltons_equations_integration_plot(K, U, 0.37, p, q, title, (0.3, 0.95), "hamiltons_equations_integration_modified_euler_method_01_15")

# %%

t = 30.0
ε = 0.1
nsteps = int(t/ε)

p, q = euler_cromer(-1.0, 1.0, dUdq, dKdp, nsteps, ε)
title = f"Hamilton's Equations (Modified Euler Method): Δt={ε}, nsteps={nsteps}"
hmc.hamiltons_equations_integration_plot(K, U, 0.37, p, q, title, (0.3, 0.95), "hamiltons_equations_integration_modified_euler_method_01_30")

# %%

t = 5.0
ε = 0.1
nsteps = int(t/ε)

p, q = momentum_verlet(-1.0, 1.0, dUdq, dKdp, nsteps, ε)
title = f"Hamilton's Equations (Momentum Verlet): Δt={ε}, nsteps={nsteps}"
hmc.hamiltons_equations_integration_plot(K, U, 0.37, p, q, title, (0.3, 0.95), "hamiltons_equations_integration_momentum_verlet_method_01_5")

# %%

t = 30.0
ε = 0.1
nsteps = int(t/ε)

p, q = momentum_verlet(-1.0, 1.0, dUdq, dKdp, nsteps, ε)
title = f"Hamilton's Equations (Momentum Verlet): Δt={ε}, nsteps={nsteps}"
hmc.hamiltons_equations_integration_plot(K, U, 0.37, p, q, title, (0.3, 0.95), "hamiltons_equations_integration_momentum_verlet_method_01_30")

# %%

t = 5.0
ε = 0.5
nsteps = int(t/ε)

p, q = momentum_verlet(-1.0, 1.0, dUdq, dKdp, nsteps, ε)
title = f"Hamilton's Equations (Momentum Verlet): Δt={ε}, nsteps={nsteps}"
hmc.hamiltons_equations_integration_plot(K, U, 0.37, p, q, title, (0.3, 0.95), "hamiltons_equations_integration_momentum_verlet_method_05_5")

# %%

t = 30.0
ε = 0.5
nsteps = int(t/ε)

p, q = momentum_verlet(-1.0, 1.0, dUdq, dKdp, nsteps, ε)
title = f"Hamilton's Equations (Momentum Verlet): Δt={ε}, nsteps={nsteps}"
hmc.hamiltons_equations_integration_plot(K, U, 0.37, p, q, title, (0.3, 0.95), "hamiltons_equations_integration_momentum_verlet_method_05_30")

# %%

t = 5.0
ε = 0.1
nsteps = int(t/ε)

integration_error_plot(-1.0, 1.0, nsteps, ε, dUdq, dKdp, "hamiltons_equations_integration_error_comparision_01_5")

# %%

t = 30.0
ε = 0.1
nsteps = int(t/ε)

integration_error_plot(-1.0, 1.0, nsteps, ε, dUdq, dKdp, "hamiltons_equations_integration_error_comparision_01_30")
