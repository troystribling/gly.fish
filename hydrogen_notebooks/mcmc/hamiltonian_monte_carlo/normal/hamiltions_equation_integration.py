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
    ps = numpy.zeros(nintegrate+1)
    qs = numpy.zeros(nintegrate+1)
    ps[0] = p0
    qs[0] = q0

    pprev = p0
    qprev = q0

    for i in range(nintegrate):
        p = pprev - ε*dUdq(qprev)
        ps[i+1] = p
        q = qprev + ε*pprev
        qs[i+1] = q

        pprev = p
        qprev = q

    return ps, qs

# Euler Dicretization integration Hamiltons's equations

def euler_cromer(p0, q0, nsteps, ε):
    ps = numpy.zeros(nsteps+1)
    qs = numpy.zeros(nsteps+1)
    ps[0] = p0
    qs[0] = q0

    pprev = p0
    qprev = q0

    for i in range(nsteps):
        p = pprev - ε*dUdq(qprev)
        ps[i+1] = p
        q = qprev + ε*p
        qs[i+1] = q

        pprev = p
        qprev = q

    return ps, qs

# Leapfrog integration of Hamilton's equations
# This is the leapfrog implementaion from AM207 https://am207.info/wiki/hmcexplore.html

def leapfrog(p0, q0, nsteps, ε):
    ps = numpy.zeros(nsteps+1)
    qs = numpy.zeros(nsteps+1)
    ps[0] = p0
    qs[0] = q0

    p = p0
    q = q0

    p = p - ε*dUdq(q)/2.0

    for i in range(nsteps):
        q = q + ε*p
        qs[i+1] = q
        if (i != nsteps-1):
            p = p - ε*dUdq(q)
            ps[i+1] = p

    p = p - ε*dUdq(q)/2.0
    ps[nsteps] = p

    return ps, qs

def momentum_verlet(p0, q0, nsteps, ε):
    ps = numpy.zeros(nsteps+1)
    qs = numpy.zeros(nsteps+1)
    ps[0] = p0
    qs[0] = q0

    p = p0
    q = q0
    ΔU = dUdq(q)

    for i in range(nsteps):
        p = p - ε*ΔU/2.0
        q = q + ε*p
        qs[i+1] = q
        ΔU = dUdq(q)
        p = p - ε*ΔU/2.0
        ps[i+1] = p

    return ps, qs

def canonical_distribution(kinetic_energy, potential_energy):
    def f(p, q):
        return numpy.exp(-kinetic_energy(p) - potential_energy(q))
    return f

def canonical_distribution_mesh(kinetic_energy, potential_energy, npts):
    x1 = numpy.linspace(-3.0, 3.0, npts)
    x2 = numpy.linspace(-3.0, 3.0, npts)
    f = canonical_distribution(kinetic_energy, potential_energy)
    x1_grid, x2_grid = numpy.meshgrid(x1, x2)
    f_x1_x2 = numpy.zeros((npts, npts))
    for i in numpy.arange(npts):
        for j in numpy.arange(npts):
            f_x1_x2[i, j] = f(x1_grid[i,j], x2_grid[i,j])
    return (x1_grid, x2_grid, f_x1_x2)

def canonical_distribution_contour_plot(kinetic_energy, potential_energy, contour_values, title, plot_name):
    npts = 500
    x1_grid, x2_grid, f_x1_x2 = canonical_distribution_mesh(kinetic_energy, potential_energy, npts)
    figure, axis = pyplot.subplots(figsize=(8, 8))
    axis.set_xlabel(r"$q$")
    axis.set_ylabel(r"$p$")
    axis.set_xlim([-3.2, 3.2])
    axis.set_ylim([-3.2, 3.2])
    axis.set_title(title)
    contour = axis.contour(x1_grid, x2_grid, f_x1_x2, contour_values, cmap=config.contour_color_map)
    axis.clabel(contour, contour.levels[::2], fmt="%.3f", inline=True, fontsize=15)
    config.save_post_asset(figure, "hamiltonian_monte_carlo", plot_name)

def hamiltons_equations_integration_plot(kinetic_energy, potential_energy, contour_value, p, q, title, legend_anchor, plot_name):
    npts = 500
    x1_grid, x2_grid, f_x1_x2 = canonical_distribution_mesh(kinetic_energy, potential_energy, npts)
    figure, axis = pyplot.subplots(figsize=(8, 8))
    axis.set_xlabel(r"$q$")
    axis.set_ylabel(r"$p$")
    axis.set_xlim([-3.2, 3.2])
    axis.set_ylim([-3.2, 3.2])
    axis.set_title(title)
    contour = axis.contour(x1_grid, x2_grid, f_x1_x2, [contour_value], cmap=config.contour_color_map, alpha=0.3)
    axis.clabel(contour, contour.levels[::2], fmt="%.3f", inline=True, fontsize=15)
    axis.plot(q, p, lw=1, color="#320075")
    axis.plot(q[0], p[0], marker='o', color="#FF9500", markersize=13.0, label="Start")
    axis.plot(q[-1], p[-1], marker='o', color="#320075", markersize=13.0, label="End")
    axis.legend(bbox_to_anchor=legend_anchor)
    config.save_post_asset(figure, "hamiltonian_monte_carlo", plot_name)

def integration_error_plot(p0, q0, nsteps, ε, plot_file):
    contour_value = p0**2+ q0**2
    p_leapfrog, q_leapfrog = leapfrog(p0, q0, nsteps, ε)
    p_euler_cromer, q_euler_cromer = euler_cromer(p0, q0, nsteps, ε)
    p_euler, q_euler = euler(p0, q0, nsteps, ε)
    p_momentum_verlet, q_momentum_verlet = momentum_verlet(p0, q0, nsteps, ε)

    error_leapfrog = 100.0*numpy.abs(numpy.sqrt(p_leapfrog[1:-1]**2 + q_leapfrog[1:-1]**2) - numpy.sqrt(contour_value))/numpy.sqrt(contour_value)
    error_euler_cromer = 100.0*numpy.abs(numpy.sqrt(p_euler_cromer[1:-1]**2 + q_euler_cromer[1:-1]**2) - numpy.sqrt(2.0))/numpy.sqrt(2.0)
    error_euler = 100.0*numpy.abs(numpy.sqrt(p_euler[1:-1]**2 + q_euler[1:-1]**2) - numpy.sqrt(2.0))/numpy.sqrt(2.0)
    error_momentum_verlet = 100.0*numpy.abs(numpy.sqrt(p_momentum_verlet[1:-1]**2 + q_momentum_verlet[1:-1]**2) - numpy.sqrt(2.0))/numpy.sqrt(2.0)

    t_plot = numpy.linspace(0.0, nsteps*ε, nsteps+1)

    figure, axis = pyplot.subplots(figsize=(10, 7))
    axis.set_xlabel("t")
    axis.set_ylabel("% Error")
    axis.set_title(f"Hamiltion's Equations Integration Error: Δt={ε}, nsteps={nsteps}")
    axis.set_xlim([0.0, t])
    axis.plot(t_plot[1:-1], error_leapfrog, label="Leapfrog")
    axis.plot(t_plot[1:-1], error_euler_cromer, label="Euler Cromer")
    axis.plot(t_plot[1:-1], error_euler, label="Euler")
    axis.plot(t_plot[1:-1], error_momentum_verlet, label="Momentum Verlet")
    config.save_post_asset(figure, "hamiltonian_monte_carlo", plot_file)
    axis.legend(bbox_to_anchor=(0.35, 0.9))

# %%

canonical_distribution_contour_plot(K, U, [0.02, 0.1, 0.2, 0.4, 0.6, 0.8], "Canonical Distribution", "hamiltons-equations-integration-normal-distribution")

# %%

t = 5.0
ε = 0.1
nsteps = int(t/ε)

p, q = euler(-1.0, 1.0, nsteps, ε)
title = f"Hamilton's Equations (Euler Method): Δt={ε}, nsteps={nsteps}"
hamiltons_equations_integration_plot(K, U, 0.37, p, q, title, (0.3, 0.95), "hamiltons-equations-integration-normal-euler-method-01-5")

# %%

t = 15.0
ε = 0.1
nsteps = int(t/ε)

p, q = euler(-1.0, 1.0, nsteps, ε)
title = f"Hamilton's Equations (Euler Method): Δt={ε}, nsteps={nsteps}"
hamiltons_equations_integration_plot(K, U, 0.37, p, q, title, (0.275, 0.8), "hamiltons-equations-integration-normal-euler-method-01-15")

# %%

t = 5.0
ε = 0.01
nsteps = int(t/ε)

p, q = euler(-1.0, 1.0, nsteps, ε)
title = f"Hamilton's Equations (Euler Method): Δt={ε}, nsteps={nsteps}"
hamiltons_equations_integration_plot(K, U, 0.37, p, q, title, (0.3, 0.95), "hamiltons-equations-integration-normal-euler-method-001-5")

# %%

t = 5.0
ε = 0.1
nsteps = int(t/ε)

p, q = euler_cromer(-1.0, 1.0, nsteps, ε)
title = f"Hamilton's Equations (Modified Euler Method): Δt={ε}, nsteps={nsteps}"
hamiltons_equations_integration_plot(K, U, 0.37, p, q, title, (0.275, 0.8), "hamiltons-equations-integration-normal-modified-euler-method-01-5")

# %%

t = 15.0
ε = 0.1
nsteps = int(t/ε)

p, q = euler_cromer(-1.0, 1.0, nsteps, ε)
title = f"Hamilton's Equations (Modified Euler Method): Δt={ε}, nsteps={nsteps}"
hamiltons_equations_integration_plot(K, U, 0.37, p, q, title, (0.3, 0.95), "hamiltons-equations-integration-normal-modified-euler-method-01-15")

# %%

t = 30.0
ε = 0.1
nsteps = int(t/ε)

p, q = euler_cromer(-1.0, 1.0, nsteps, ε)
title = f"Hamilton's Equations (Modified Euler Method): Δt={ε}, nsteps={nsteps}"
hamiltons_equations_integration_plot(K, U, 0.37, p, q, title, (0.25, 0.8), "hamiltons-equations-integration-normal-modified-euler-method-01-30")

# %%

t = 5.0
ε = 0.1
nsteps = int(t/ε)

p, q = leapfrog(-1.0, 1.0, nsteps, ε)
title = f"Hamilton's Equations (Leap Frog Method): Δt={ε}, nsteps={nsteps}"
hamiltons_equations_integration_plot(K, U, 0.37, p, q, title, (0.3, 0.95), "hamiltons-equations-integration-normal-leapfrog-method-01-5")

# %%

t = 30.0
ε = 0.1
nsteps = int(t/ε)

p, q = leapfrog(-1.0, 1.0, nsteps, ε)
title = f"Hamilton's Equations (Leap Frog Method): Δt={ε}, nsteps={nsteps}"
hamiltons_equations_integration_plot(K, U, 0.37, p, q, title, (0.3, 0.95), "hamiltons-equations-integration-normal-leapfrog-method-01-30")

# %%

t = 5.0
ε = 0.5
nsteps = int(t/ε)

p, q = leapfrog(-1.0, 1.0, nsteps, ε)
title = f"Hamilton's Equations (Leap Frog Method): Δt={ε}, nsteps={nsteps}"
hamiltons_equations_integration_plot(K, U, 0.37, p, q, title, (0.3, 0.95), "hamiltons-equations-integration-normal-leapfrog-method-05-5")

# %%

t = 30.0
ε = 0.5
nsteps = int(t/ε)

p, q = leapfrog(-1.0, 1.0, nsteps, ε)
title = f"Hamilton's Equations (Leap Frog Method): Δt={ε}, nsteps={nsteps}"
hamiltons_equations_integration_plot(K, U, 0.37, p, q, title, (0.3, 0.95), "hamiltons-equations-integration-normal-leapfrog-method-05-30")

# %%

t = 5.0
ε = 0.1
nsteps = int(t/ε)

p, q = momentum_verlet(-1.0, 1.0, nsteps, ε)
title = f"Hamilton's Equations (Momentum Verlet): Δt={ε}, nsteps={nsteps}"
hamiltons_equations_integration_plot(K, U, 0.37, p, q, title, (0.3, 0.95), "hamiltons-equations-integration-normal-momentum-verlet-01-5")

# %%

t = 30.0
ε = 0.1
nsteps = int(t/ε)

p, q = momentum_verlet(-1.0, 1.0, nsteps, ε)
title = f"Hamilton's Equations (Momentum Verlet): Δt={ε}, nsteps={nsteps}"
hamiltons_equations_integration_plot(K, U, 0.37, p, q, title, (0.3, 0.95), "hamiltons-equations-integration-normal-momentum-verlet-01-30")

# %%

t = 5.0
ε = 0.5
nsteps = int(t/ε)

p, q = momentum_verlet(-1.0, 1.0, nsteps, ε)
title = f"Hamilton's Equations (Momentum Verlet): Δt={ε}, nsteps={nsteps}"
hamiltons_equations_integration_plot(K, U, 0.37, p, q, title, (0.3, 0.95), "hamiltons-equations-integration-normal-momentum-verlet-05-5")

# %%

t = 30.0
ε = 0.5
nsteps = int(t/ε)

p, q = momentum_verlet(-1.0, 1.0, nsteps, ε)
title = f"Hamilton's Equations (Momentum Verlet): Δt={ε}, nsteps={nsteps}"
hamiltons_equations_integration_plot(K, U, 0.37, p, q, title, (0.3, 0.95), "hamiltons-equations-integration-normal-momentum-verlet-05_-0")

# %%

t = 5.0
ε = 0.1
nsteps = int(t/ε)

integration_error_plot(-1.0, 1.0, nsteps, ε, "hamiltons-equations-integration-normal-error-comparison-01-5")

# %%

t = 30.0
ε = 0.1
nsteps = int(t/ε)

integration_error_plot(-1.0, 1.0, nsteps, ε, "hamiltons-equations-integration-normal-error-comparison-01-30")
