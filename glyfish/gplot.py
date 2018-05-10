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
    figure, axis = pyplot.subplots(figsize=(12, 5))
    axis.set_xlabel("Step Size")
    axis.set_ylabel("Acceptance %")
    axis.set_title(title)
    axis.set_xlim([0.0, x[-1]])
    axis.set_ylim([0.0, 100.0])
    axis.plot(x, y, zorder=5)
