import numpy
from matplotlib import pyplot
from glyfish import config
from glyfish import stats

# Plots

# Momentum Verlet integration of Hamiltons's equations
def momentum_verlet(p0, q0, ndim, dUdq, dKdp, nsteps, ε):
    p = numpy.zeros((nsteps+1, ndim))
    q = numpy.zeros((nsteps+1, ndim))
    p[0] = p0
    q[0] = q0

    for i in range(nsteps):
        for j in range(ndim):
            p[i+1][j] = p[i][j] - ε*dUdq(q, i, j, True)/2.0
            q[i+1][j] = q[i][j] + ε*dKdp(p, i+1, j)
            p[i+1][j] = p[i+1][j] - ε*dUdq(q, i, j, False)/2.0

    return p, q

# Bivariate Normal Distributution Potential Energy and Kinetic Energy

# %%
# Here Hamiltonian Monte Carlo is used to generate samples for single variable
# Unit Normal distribution. The Potential and Kinetic Energy are given by assuming a mass of 1

def bivariate_normal_U(γ, σ1, σ2):
    scale = σ1**2*σ2**2*(1.0 - γ)
    def f(q):
        return ((q[0]*σ1)**2 + (q[1]*σ2)**2 - q[0]*q[1]*σ1*σ2*γ) / (2.0*scale)
    return f

def bivariate_normal_K(m1, m2):
    def f(p):
        return (p[0]**2/m1 + p[1]**2/m2) / 2.0
    return f

def bivariate_normal_dUdq(γ, σ1, σ2):
    scale = σ1**2*σ2**2*(1.0 - γ)
    def f(q, n, i, is_first_step):
        if i == 0:
            if is_first_step:
                return (q[n][0]*σ1**2 - q[n][1]*γ*σ1*σ2) / scale
            else:
                return (q[n+1][0]*σ1**2 - q[n][1]*γ*σ1*σ2) / scale
        elif i == 1:
            if is_first_step:
                return (q[n][1]*σ2**2 - q[n+1][0]*γ*σ1*σ2) / scale
            else:
                return (q[n+1][1]*σ2**2 - q[n+1][0]*γ*σ1*σ2) / scale
    return f

def bivariate_normal_dKdp(m1, m2):
    def f(p, n, i):
        if i == 0:
            return p[n][0]/m1
        elif i == 1:
            return p[n][1]/m2
    return f

# Plots
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

def phase_space_plot(p, q, title, labels, legend_anchor, plot_name):
    figure, axis = pyplot.subplots(figsize=(9, 9))
    axis.set_xlabel(labels[0])
    axis.set_ylabel(labels[1])
    axis.set_title(title)
    axis.plot(q, p, lw=1, color="#320075")
    axis.plot(q[0], p[0], marker='o', color="#FF9500", markersize=13.0, label="Start")
    axis.plot(q[-1], p[-1], marker='o', color="#320075", markersize=13.0, label="End")
    axis.legend(bbox_to_anchor=legend_anchor)
    config.save_post_asset(figure, "hamiltonian_monte_carlo", plot_name)

def univariate_pdf_plot(pdf, x, x_title, title, file):
    figure, axis = pyplot.subplots(figsize=(10, 7))
    axis.set_xlabel(x_title)
    axis.set_ylabel("PDF")
    axis.set_xlim([x[0], x[-1]])
    axis.set_title(title)
    axis.plot(x, [pdf(j) for j in x])
    config.save_post_asset(figure, "hamiltonian_monte_carlo", file)

def grid_pdf(pdf, xrange, yrange, npts):
    x = numpy.linspace(xrange[0], xrange[1], npts)
    y = numpy.linspace(yrange[0], yrange[1], npts)

    x_grid, y_grid = numpy.meshgrid(x, y)
    f = numpy.zeros((npts, npts))
    for i in numpy.arange(npts):
        for j in numpy.arange(npts):
            f[i, j] = pdf(x_grid[i,j], y_grid[i,j])

    dx = (xrange[1] - xrange[0])/npts
    dy = (yrange[1] - yrange[0])/npts

    return f/(dx*dy*numpy.sum(f)), x_grid, y_grid

def canonical_distribution_samples_contour(potential_energy, kinetic_energy, p, q, xrange, yrange, labels, title, file):
    npts = 500
    pdf, x, y = grid_pdf(canonical_distribution(potential_energy, kinetic_energy), xrange, yrange, npts)
    bins = [numpy.linspace(xrange[0], xrange[1], 100), numpy.linspace(yrange[0], yrange[1], 100)]
    figure, axis = pyplot.subplots(figsize=(10, 8))
    axis.set_xlabel(labels[0])
    axis.set_ylabel(labels[1])
    axis.set_title(title)
    hist, _, _, image = axis.hist2d(p, q, normed=True, bins=bins, cmap=config.alternate_color_map)
    contour = axis.contour(x, y, pdf, cmap=config.alternate_contour_color_map)
    axis.clabel(contour, contour.levels[::2], fmt="%.1f", inline=True, fontsize=15)
    figure.colorbar(image)
    config.save_post_asset(figure, "hamiltonian_monte_carlo", file)

def cumulative_mean(title, samples, time, μ, ylim, file):
    nsample = len(time)
    figure, axis = pyplot.subplots(figsize=(10, 7))
    axis.set_xlabel("Time")
    axis.set_ylabel(r"$μ$")
    axis.set_title(title)
    axis.set_ylim(ylim)
    axis.set_xlim([10.0, nsample])
    axis.semilogx(time, numpy.full((len(time)), μ), label="Target μ", color="#000000")
    axis.semilogx(time, stats.cummean(samples))
    config.save_post_asset(figure, "hamiltonian_monte_carlo", file)

def cumulative_standard_deviation(title, samples, time, σ, ylim, file):
    nsample = len(time)
    figure, axis = pyplot.subplots(figsize=(10, 7))
    axis.set_xlabel("Time")
    axis.set_ylabel(r"$σ$")
    axis.set_title(title)
    axis.set_ylim(ylim)
    axis.set_xlim([10.0, nsample])
    axis.semilogx(time, numpy.full((len(time)), σ), label="Target μ", color="#000000")
    axis.semilogx(time, stats.cumsigma(samples))
    config.save_post_asset(figure, "hamiltonian_monte_carlo", file)

def time_series(title, samples, time, ylim, plot):
    figure, axis = pyplot.subplots(figsize=(12, 4))
    axis.set_title(title)
    axis.set_xlabel("Time")
    axis.set_xlim([time[0], time[-1]])
    axis.set_ylim(ylim)
    axis.plot(time, samples, lw="1")
    config.save_post_asset(figure, "hamiltonian_monte_carlo", plot)

def multicurve(title, y, x, x_lab, y_lab, curve_labs, legend_anchor, ylim, plot):
    nplots = len(y)
    figure, axis = pyplot.subplots(figsize=(10, 7))
    axis.set_title(title)
    axis.set_xlabel(x_lab)
    axis.set_ylabel(y_lab)
    axis.set_xlim([x[0], x[-1]])
    axis.set_ylim(ylim)
    for i in range(nplots):
        axis.plot(x, y[i], label=curve_labs[i])
    axis.legend(bbox_to_anchor=legend_anchor)
    config.save_post_asset(figure, "hamiltonian_monte_carlo", plot)

def autocor(title, samples, max_lag, plot):
    figure, axis = pyplot.subplots(figsize=(10, 7))
    axis.set_title(title)
    axis.set_ylabel(r"$\gamma_{\tau}$")
    axis.set_xlabel("Time Lag (τ)")
    axis.set_xlim([0, max_lag])
    ac = stats.autocorrelate(samples)
    axis.plot(range(max_lag), numpy.real(ac[:max_lag]))
    config.save_post_asset(figure, "hamiltonian_monte_carlo", plot)
