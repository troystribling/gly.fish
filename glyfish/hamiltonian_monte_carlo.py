import numpy
from matplotlib import pyplot
from glyfish import config
from glyfish import stats

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
    nplots = len(samples)
    figure, axis = pyplot.subplots(figsize=(12, 4))
    axis.set_title(title)
    axis.set_xlabel("Time")
    axis.set_xlim([time[0], time[-1] + 1])
    axis.set_ylim(ylim)
    axis.plot(time, samples, lw="1")
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
