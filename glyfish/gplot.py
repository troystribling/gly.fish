import numpy
from matplotlib import pyplot
from glyfish import stats

def pdf_samples(title, pdf, samples, xrange=None, ylimit=None):
    figure, axis = pyplot.subplots(figsize=(12, 5))
    axis.set_xlabel("X")
    axis.set_ylabel("PDF")
    axis.set_title(title)
    _, bins, _ = axis.hist(samples, 50, density=True, color="#336699", alpha=0.6, label=f"Sampled Distribution", edgecolor="#336699", zorder=5)
    if xrange is None:
        delta = (bins[-1] - bins[0]) / 500.0
        xrange = numpy.arange(bins[0], bins[-1], delta)
    sample_distribution = [pdf(val) for val in xrange]
    axis.set_xlim([xrange[0], xrange[-1]])
    if ylimit is not None:
        axis.set_ylim(ylimit)
    axis.plot(xrange, sample_distribution, color="#A60628", label=f"Sampled Function", zorder=6)
    axis.legend()


def acceptance(title, x, y, xlim):
    figure, axis = pyplot.subplots(figsize=(12, 5))
    axis.set_xlabel("Step Size")
    axis.set_ylabel("Acceptance %")
    axis.set_title(title)
    axis.set_xlim(xlim)
    axis.set_ylim([0.1, 200.0])
    axis.loglog(x, y, zorder=5, marker='o', color="#336699", markersize=15.0, linestyle="None", markeredgewidth=1.0, alpha=0.5, label="Simulation")


def time_series(title, samples, time, ylim):
    nplots = len(samples)
    figure, axis = pyplot.subplots(figsize=(12, 3))
    axis.set_title(title)
    axis.set_xlabel("Time")
    axis.set_xlim([time[0], time[-1] + 1])
    axis.set_ylim(ylim)
    axis.plot(time, samples, lw="1")

def steps_size_time_series(title, samples, time, stepsize, acceptance, ylim, text_pos):
    nplots = len(samples)
    figure, axis = pyplot.subplots(nrows=nplots, ncols=1, sharex=True, figsize=(12, 3*nplots))
    axis[0].set_title(title)
    axis[-1].set_xlabel("Time")
    bbox = dict(boxstyle='square,pad=1', facecolor="#FFFFFF", edgecolor="lightgrey")
    for i in range(nplots):
        axis[i].set_xlim([time[0], time[-1] + 1])
        axis[i].set_ylim(ylim)
        axis[i].plot(time, samples[i], lw="1")
        axis[i].text(text_pos[0], text_pos[1], f"stepsize={format(stepsize[i], '2.3')}, accepted={format(acceptance[i], '2.0f')}%", fontsize=13, bbox=bbox)

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
