import numpy
from matplotlib import pyplot
from glyfish import stats


def pdf_samples(title, pdf, samples, xrange=None):
    figure, axis = pyplot.subplots(figsize=(12, 5))
    axis.set_xlabel("X")
    axis.set_ylabel("PDF")
    axis.set_title(title)
    _, bins, _ = axis.hist(samples, 50, density=True, color="#336699", alpha=0.6, label=f"Sampled Distribution", edgecolor="#336699", zorder=5)
    if xrange is None:
        delta = (bins[-1] - bins[0]) / 500.0
        xrange = numpy.arange(bins[0], bins[-1], delta)
    sample_distribution = [pdf(val) for val in xrange]
    axis.plot(xrange, sample_distribution, color="#A60628", label=f"Sampled Function", zorder=6)
    axis.legend()


def acceptance(title, x, y, xlim):
    x_optimal = numpy.linspace(xlim[0], xlim[1], 100)
    optimal = numpy.full((len(x_optimal)), 80.0)
    figure, axis = pyplot.subplots(figsize=(12, 5))
    axis.set_xlabel("Step Size")
    axis.set_ylabel("Acceptance %")
    axis.set_title(title)
    axis.set_xlim(xlim)
    axis.set_ylim([1.0, 200.0])
    axis.loglog(x, y, zorder=5, marker='o', color="#336699", markersize=15.0, linestyle="None", markeredgewidth=1.0, alpha=0.5, label="Simulation")
    axis.loglog(x_optimal, optimal, zorder=5, color="#A60628", label="80% Acceptance")
    axis.legend()


def time_series(title, samples, time, ylim):
    nplots = len(samples)
    figure, axis = pyplot.subplots(nrows=nplots, ncols=1, figsize=(12, 9))
    axis[0].set_title(title)
    axis[-1].set_xlabel("Time")
    for i in range(0, nplots):
        axis[i].set_xlim([time[i][0], time[i][-1] + 1])
        axis[i].set_ylim(ylim)
        axis[i].plot(time[i], samples[i], lw="1")


def steps_size_time_series(title, samples, time, stepsize, acceptance, ylim, text_pos):
    nplots = len(samples)
    figure, axis = pyplot.subplots(nrows=nplots, ncols=1, sharex=True, figsize=(12, 3*nplots))
    axis[0].set_title(title)
    axis[-1].set_xlabel("Time")
    for i in range(nplots):
        axis[i].set_xlim([time[0], time[-1] + 1])
        axis[i].set_ylim(ylim)
        axis[i].plot(time, samples[i], lw="1")
        axis[i].text(text_pos[0], text_pos[1], f"stepsize={stepsize[i]}, accepted={format(acceptance[i], '2.0f')}%", fontsize=13)

def step_size_mean(title, samples, time, μ, stepsize):
    nplot = len(samples)
    nsample = len(time)
    figure, axis = pyplot.subplots(figsize=(12, 6))
    axis.set_xlabel("Time")
    axis.set_ylabel(r"$μ$")
    axis.set_title(title)
    axis.set_xlim([10.0, nsample])
    axis.semilogx(time, numpy.full((len(time)), μ), label="Target μ", color="#000000")
    for i in range(nplot):
        axis.semilogx(time, stats.cummean(samples[i]), label=f"stepsize={format(stepsize[i], '2.3f')}", lw=2)
    axis.legend()

def step_size_sigma(title, samples, time, σ, stepsize):
    nplot = len(samples)
    nsample = len(time)
    figure, axis = pyplot.subplots(figsize=(12, 6))
    axis.set_xlabel("Time")
    axis.set_ylabel(r"$σ$")
    axis.set_title(title)
    axis.set_xlim([10.0, nsample])
    axis.semilogx(time, numpy.full((len(time)), σ), label="Target σ", color="#000000")
    for i in range(nplot):
        axis.semilogx(time, stats.cumsigma(samples[i]), label=f"stepsize={format(stepsize[i], '2.3f')}", lw=2)
    axis.legend()

def step_size_autocor(title, samples, stepsize, npts):
    nplot = len(samples)
    figure, axis = pyplot.subplots(figsize=(12, 9))
    axis.set_title(title)
    axis.set_xlabel("Time Lag")
    axis.set_xlim([0, npts])
    for i in range(nplot):
        ac = stats.autocorrelate(samples[i])
        axis.plot(range(npts), numpy.real(ac[:npts]), label=f"stepsize={format(stepsize[i], '2.3f')}")
    axis.legend()
