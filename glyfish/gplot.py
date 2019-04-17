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
