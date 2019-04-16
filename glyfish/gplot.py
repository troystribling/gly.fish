import numpy
from matplotlib import pyplot
from glyfish import stats
from glyfish import config

def pdf_samples(title, pdf, samples, post, plot, xrange=None, ylimit=None):
    figure, axis = pyplot.subplots(figsize=(10, 7))
    axis.set_xlabel(r"$X$")
    axis.set_title(title)
    axis.set_prop_cycle(config.distribution_sample_cycler)
    _, bins, _ = axis.hist(samples, 50, rwidth=0.8, density=True, label=f"Samples", zorder=5)
    if xrange is None:
        delta = (bins[-1] - bins[0]) / 500.0
        xrange = numpy.arange(bins[0], bins[-1], delta)
    sample_distribution = [pdf(val) for val in xrange]
    axis.set_xlim([xrange[0], xrange[-1]])
    if ylimit is not None:
        axis.set_ylim(ylimit)
    axis.plot(xrange, sample_distribution, label=f"Target PDF", zorder=6)
    axis.legend(bbox_to_anchor=(0.75, 0.9))
    config.save_post_asset(figure, post, plot)

def acceptance(title, x, y, xlim, example_idx, post, plot):
    figure, axis = pyplot.subplots(figsize=(10, 7))
    axis.set_xlabel("Step Size")
    axis.set_ylabel("Acceptance %")
    axis.set_title(title)
    axis.set_xlim(xlim)
    axis.set_ylim([0.7, 200.0])
    axis.set_prop_cycle(config.alternate_cycler)
    axis.loglog(x, y, zorder=5, marker='o', markersize=15.0, linestyle="None", markeredgewidth=1.0)
    axis.loglog(x[example_idx[0]], y[example_idx[0]], zorder=5, marker='o', markersize=15.0, linestyle="None", markeredgewidth=1.0, label=f" Small Step Size: ({format(x[example_idx[0]], '.2f')}, {format(y[example_idx[0]], '2.0f')}%)")
    axis.loglog(x[example_idx[1]], y[example_idx[1]], zorder=5, marker='o', markersize=15.0, linestyle="None", markeredgewidth=1.0, label=f" Best Step Size:   ({format(x[example_idx[1]], '.2f')}, {format(y[example_idx[1]], '2.0f')}%)")
    axis.loglog(x[example_idx[2]], y[example_idx[2]], zorder=5, marker='o', markersize=15.0, linestyle="None", markeredgewidth=1.0, label=f" Large Step Size: ({format(x[example_idx[2]], '.2f')}, {format(y[example_idx[2]], '2.0f')}%)")
    axis.legend(bbox_to_anchor=(0.55, 0.6))
    config.save_post_asset(figure, post, plot)

def time_series(title, samples, time, ylim, post, plot):
    nplots = len(samples)
    figure, axis = pyplot.subplots(figsize=(12, 4))
    axis.set_title(title)
    axis.set_xlabel("Time")
    axis.set_xlim([time[0], time[-1] + 1])
    axis.set_ylim(ylim)
    axis.plot(time, samples, lw="1")
    config.save_post_asset(figure, post, plot)

def steps_size_time_series(title, samples, time, stepsize, acceptance, ylim, text_pos, post, plot):
    nplots = len(samples)
    figure, axis = pyplot.subplots(nrows=nplots, ncols=1, sharex=True, figsize=(10, 3*nplots))
    axis[0].set_title(title)
    axis[-1].set_xlabel("Time")
    bbox = dict(boxstyle='square,pad=1', facecolor="#FFFFFF", edgecolor="white", alpha=0.75)
    for i in range(nplots):
        axis[i].set_xlim([time[0], time[-1] + 1])
        axis[i].set_ylim(ylim)
        axis[i].plot(time, samples[i], lw="2")
        axis[i].text(text_pos[0], text_pos[1], f"stepsize={format(stepsize[i], '2.2f')}\naccepted={format(acceptance[i], '2.0f')}%", fontsize=13, bbox=bbox)
    config.save_post_asset(figure, post, plot)

def step_size_mean(title, samples, time, μ, stepsize, post, plot, legend_pos = None):
    nplot = len(samples)
    nsample = len(time)
    figure, axis = pyplot.subplots(figsize=(10, 7))
    axis.set_xlabel("Time")
    axis.set_ylabel(r"$μ$")
    axis.set_title(title)
    axis.set_xlim([10.0, nsample])
    axis.semilogx(time, numpy.full((len(time)), μ), label="Target μ", color="#000000")
    for i in range(nplot):
        axis.semilogx(time, stats.cummean(samples[i]), label=f"stepsize={format(stepsize[i], '2.2f')}")
    if legend_pos is None:
        axis.legend()
    else:
        axis.legend(bbox_to_anchor=legend_pos)
    config.save_post_asset(figure, post, plot)

def cumulative_mean(title, samples, time, μ, stepsize, post, plot, legend_pos = None):
    nplot = len(samples)
    nsample = len(time)
    figure, axis = pyplot.subplots(figsize=(10, 7))
    axis.set_xlabel("Time")
    axis.set_ylabel(r"$μ$")
    axis.set_title(title)
    axis.set_xlim([10.0, nsample])
    axis.semilogx(time, numpy.full((len(time)), μ), label="Target μ", color="#000000")
    for i in range(nplot):
        axis.semilogx(time, stats.cummean(samples[i]), label=f"stepsize={format(stepsize[i], '2.2f')}")
    if legend_pos is None:
        axis.legend()
    else:
        axis.legend(bbox_to_anchor=legend_pos)
    config.save_post_asset(figure, post, plot)

def step_size_sigma(title, samples, time, σ, stepsize, post, plot, legend_pos = None):
    nplot = len(samples)
    nsample = len(time)
    figure, axis = pyplot.subplots(figsize=(10, 7))
    axis.set_xlabel("Time")
    axis.set_ylabel(r"$σ$")
    axis.set_title(title)
    axis.set_xlim([10.0, nsample])
    axis.semilogx(time, numpy.full((len(time)), σ), label="Target σ", color="#000000")
    for i in range(nplot):
        axis.semilogx(time, stats.cumsigma(samples[i]), label=f"stepsize={format(stepsize[i], '2.2f')}")
    if legend_pos is None:
        axis.legend()
    else:
        axis.legend(bbox_to_anchor=legend_pos)
    config.save_post_asset(figure, post, plot)

def step_size_autocor(title, samples, stepsize, npts, post, plot):
    nplot = len(samples)
    figure, axis = pyplot.subplots(figsize=(12, 9))
    axis.set_title(title)
    axis.set_ylabel(r"$\gamma_{\tau}$")
    axis.set_xlabel("Time Lag (τ))")
    axis.set_xlim([0, npts])
    for i in range(nplot):
        ac = stats.autocorrelate(samples[i])
        axis.plot(range(npts), numpy.real(ac[:npts]), label=f"stepsize={format(stepsize[i], '2.2f')}")
    axis.legend(bbox_to_anchor=(0.7, 0.6))
    config.save_post_asset(figure, post, plot)
