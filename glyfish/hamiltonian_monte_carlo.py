import numpy
from matplotlib import pyplot
from glyfish import config
from glyfish import stats

# Plots

# Momentum Verlet integration of Hamiltons's equations used by HMC algorithm
def momentum_verlet_integrator(p0, q0, dUdq, dKdp, nsteps, ε):
    ndim = len(p0)
    p = numpy.zeros(ndim)
    q = numpy.zeros(ndim)

    for j in range(ndim):
        p[j] = p0[j]
        q[j] = q0[j]

    for i in range(nsteps):
        for j in range(ndim):
            p[j] = p[j] - ε*dUdq(q, j)/2.0
            q[j] = q[j] + ε*dKdp(p, j)
            p[j] = p[j] - ε*dUdq(q, j)/2.0

    return p, q

# Hamiltonian Monte Carlo

def HMC(q0, U, K, dUdq, dKdp, integrator, momentum_generator, nsample, tmax, ε):
    ndim = len(q0)
    current_q = numpy.zeros(ndim)
    current_p = numpy.zeros(ndim)

    for j in range(ndim):
        current_q[j] = q0[j]

    H = numpy.zeros(nsample)
    qall = numpy.zeros((nsample, ndim))
    pall = numpy.zeros((nsample, ndim))
    accepted = 0

    for i in range(nsample):

        # generate momentum sample
        for j in range(ndim):
            current_p[j] = momentum_generator(j)

        # integrate hamiltons equations using current_p and current_q to obtain proposal samples p and q
        # and negate p for detailed balance
        nsteps = int(numpy.random.rand()*tmax/ε)
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
            qall[i] = q
            pall[i] = p
            accepted += 1
        else:
            qall[i] = current_q
            pall[i] = current_p

        H[i] = U(current_q) + K(current_p)

    return H, pall, qall, accepted

# Bivariate Normal Distributution Potential Energy and Kinetic Energy

# %%
# Here Hamiltonian Monte Carlo is used to generate samples for single variable
# Unit Normal distribution. The Potential and Kinetic Energy are given by assuming a mass of 1

def bivariate_normal_U(γ, σ1, σ2):
    scale = σ1**2*σ2**2*(1.0 - γ**2)
    def f(q):
        return ((q[0]*σ2)**2 + (q[1]*σ1)**2 - 2.0*q[0]*q[1]*σ1*σ2*γ) / (2.0*scale)
    return f

def bivariate_normal_K(m1, m2):
    def f(p):
        return (p[0]**2/m1+ p[1]**2/m2) / 2.0
    return f

def bivariate_normal_dUdq(γ, σ1, σ2):
    scale = σ1**2*σ2**2*(1.0 - γ**2)
    def f(q, n):
        if n == 0:
            return (q[0]*σ2**2 - q[1]*γ*σ1*σ2) / scale
        elif n == 1:
            return (q[1]*σ1**2 - q[0]*γ*σ1*σ2) / scale
    return f

def bivariate_normal_dKdp(m1, m2):
    def f(p, n):
        if n == 0:
            return p[0]/m1
        elif n == 1:
            return p[1]/m2
    return f

def bivariate_normal_momentum_generator(m1, m2):
    def f(n):
        if n == 0:
            return numpy.random.normal(0.0, numpy.sqrt(m1))
        elif n == 1:
            return numpy.random.normal(0.0, numpy.sqrt(m2))
    return f

# Plots

def canonical_distribution(kinetic_energy, potential_energy):
    def f(pq):
        return numpy.exp(-kinetic_energy(pq[0]) - potential_energy(pq[1]))
    return f

def potential_distribution(potential_energy):
    def f(q):
        return numpy.exp(-potential_energy(q))
    return f

def momentum_distribution(kinetic_energy):
    def f(p):
        return numpy.exp(-kinetic_energy(p))
    return f

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
            f[i, j] = pdf([x_grid[i,j], y_grid[i,j]])

    dx = (xrange[1] - xrange[0])/npts
    dy = (yrange[1] - yrange[0])/npts

    return f/(dx*dy*numpy.sum(f)), x_grid, y_grid

def pdf_contour_plot(pdf, contour_values, xrange, yrange, labels, title, plot_name):
    npts = 500
    f_x1_x2, x1_grid, x2_grid = grid_pdf(pdf, xrange, yrange, npts)
    figure, axis = pyplot.subplots(figsize=(8, 8))
    axis.set_xlabel(labels[0])
    axis.set_ylabel(labels[1])
    axis.set_title(title)
    contour = axis.contour(x1_grid, x2_grid, f_x1_x2, contour_values, cmap=config.contour_color_map)
    axis.clabel(contour, contour.levels[::2], fmt="%.3f", inline=True, fontsize=15)
    config.save_post_asset(figure, "hamiltonian_monte_carlo", plot_name)

def pdf_samples_contour(pdf, p, q, xrange, yrange, contour_values, labels, title, file):
    npts = 500
    fxy, x, y = grid_pdf(pdf, xrange, yrange, npts)
    bins = [numpy.linspace(xrange[0], xrange[1], 100), numpy.linspace(yrange[0], yrange[1], 100)]
    figure, axis = pyplot.subplots(figsize=(11, 9))
    axis.set_xlabel(labels[0])
    axis.set_ylabel(labels[1])
    axis.set_title(title)
    hist, _, _, image = axis.hist2d(p, q, normed=True, bins=bins, cmap=config.alternate_color_map)
    contour = axis.contour(x, y, fxy, contour_values, cmap=config.alternate_contour_color_map)
    axis.clabel(contour, contour.levels[::2], fmt="%.3f", inline=True, fontsize=15)
    figure.colorbar(image)
    config.save_post_asset(figure, "hamiltonian_monte_carlo", file)

def distribution_samples(x, y, xrange, yrange, labels, title, file):
    bins = [numpy.linspace(xrange[0], xrange[1], 100), numpy.linspace(yrange[0], yrange[1], 100)]
    figure, axis = pyplot.subplots(figsize=(11, 9))
    axis.set_xlabel(labels[0])
    axis.set_ylabel(labels[1])
    axis.set_title(title)
    hist, _, _, image = axis.hist2d(x, y, normed=True, bins=bins, cmap=config.alternate_color_map)
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
    axis.semilogx(time, numpy.full((len(time)), σ), label="Target σ", color="#000000")
    axis.semilogx(time, stats.cumsigma(samples))
    config.save_post_asset(figure, "hamiltonian_monte_carlo", file)

def cumulative_correlation(title, x, y, time, γ, file):
    nsample = len(time)
    cov = stats.cum_covaraince(x, y)
    sigmax = stats.cumsigma(x)
    sigmay = stats.cumsigma(y)
    γt = numpy.zeros(len(cov))

    for i in range(1, len(cov)):
        γt[i] = cov[i]/(sigmax[i]*sigmay[i])

    figure, axis = pyplot.subplots(figsize=(10, 7))
    axis.set_xlabel("Time")
    axis.set_ylabel(r"$γ$")
    axis.set_title(title)
    axis.set_ylim([-1.1, 1.1])
    axis.set_xlim([10.0, nsample])
    axis.semilogx(time, numpy.full((len(time)), γ), label="Target γ", color="#000000")
    axis.semilogx(time, γt)
    config.save_post_asset(figure, "hamiltonian_monte_carlo", file)

def time_series(title, samples, time, ylim, plot):
    figure, axis = pyplot.subplots(figsize=(12, 4))
    axis.set_title(title)
    axis.set_xlabel("Time")
    axis.set_xlim([time[0], time[-1]])
    axis.set_ylim(ylim)
    axis.plot(time, samples, lw="1")
    config.save_post_asset(figure, "hamiltonian_monte_carlo", plot)

def energy_time_series(title, U, K, p, q, time, legend_anchor, ylim, plot):
    Kt = numpy.array([K(p[i]) for i in range(len(time))])
    Ut = numpy.array([U(q[i]) for i in range(len(time))])
    multicurve(title, [Kt, Ut, Kt+Ut], time, "Time", "Energy", ["K", "U", "H"], legend_anchor, ylim, plot, 3)

def multicurve(title, y, x, x_lab, y_lab, curve_labs, legend_anchor, ylim, plot, ncol=None):
    nplots = len(y)
    figure, axis = pyplot.subplots(figsize=(10, 7))
    axis.set_title(title)
    axis.set_xlabel(x_lab)
    axis.set_ylabel(y_lab)
    axis.set_xlim([x[0], x[-1]])
    axis.set_ylim(ylim)
    for i in range(nplots):
        axis.plot(x, y[i], label=curve_labs[i])
    if ncol is None:
        axis.legend(bbox_to_anchor=legend_anchor)
    else:
        axis.legend(bbox_to_anchor=legend_anchor, ncol=ncol)
    config.save_post_asset(figure, "hamiltonian_monte_carlo", plot)

def autocor(title, samples, max_lag, plot):
    figure, axis = pyplot.subplots(figsize=(10, 7))
    axis.set_title(title)
    axis.set_ylabel(r"$\gamma_{\tau}$")
    axis.set_xlabel("Time Lag (τ)")
    axis.set_xlim([0, max_lag])
    axis.set_ylim([-0.05, 1.0])
    ac = stats.autocorrelate(samples)
    axis.plot(range(max_lag), numpy.real(ac[:max_lag]))
    config.save_post_asset(figure, "hamiltonian_monte_carlo", plot)
