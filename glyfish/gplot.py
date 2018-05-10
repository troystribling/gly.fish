import numpy
from matplotlib import pyplot
from scipy import stats

def pdf_samples(title, pdf, samples):
    figure, axis = pyplot.subplots(figsize=(12, 5))
    axis.set_xlabel("X")
    axis.set_ylabel("PDF")
    axis.set_title(title)
    _, bins, _ = axis.hist(samples, 50, density=True, color="#336699", alpha=0.6, label=f"Sampled Distribution", edgecolor="#336699", zorder=5)
    delta = (bins[-1] - bins[0]) / 500.0
    sample_distribution = [pdf(val) for val in numpy.arange(bins[0], bins[-1], delta)]
    axis.plot(numpy.arange(bins[0], bins[-1], delta), sample_distribution, color="#A60628", label=f"Sampled Function", zorder=6)
    axis.legend()

def acceptance(title, x, y):
    xlim = [0.005, 20.0]
    x_optimal = numpy.linspace(xlim[0], xlim[1], 100)
    optimal = numpy.full((len(x_optimal)), 80.0)
    figure, axis = pyplot.subplots(figsize=(12, 5))
    axis.set_xlabel("Step Size")
    axis.set_ylabel("Acceptance %")
    axis.set_title(title)
    axis.set_xlim(xlim)
    axis.set_ylim([0.05, 200.0])
    axis.loglog(x, y, zorder=5, marker='o', color="#336699", markersize=15.0, linestyle="None", markeredgewidth=1.0, alpha=0.5)
    axis.loglog(x_optimal, optimal, zorder=5, color="#A60628")
