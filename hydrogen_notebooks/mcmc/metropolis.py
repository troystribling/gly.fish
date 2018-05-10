# %%
%load_ext autoreload
%autoreload 2

import numpy
from matplotlib import pyplot
from scipy import stats
from glyfish import config
from glyfish import metropolis_hastings as mh
from glyfish import gplot

%matplotlib inline

pyplot.style.use(config.glyfish_style)

# %%

def gauss(x):
    return numpy.exp(-0.5*x**2)/numpy.sqrt(2.0*numpy.pi)

def cauchy(x):
    return 1.0/(numpy.pi*(1.0+x**2))

def weibull(x):
    return 0.544*x*numpy.exp(-(x/1.9)**2)

def metropolis(p, nsample=10000, x0=0.0):
    x = x0
    samples = numpy.zeros(nsample)
    for i in range(0, nsample):
        x_star = x + numpy.random.normal()
        accept = numpy.random.rand()
        px_star = p(x_star)
        px = p(x)
        α = px_star / px
        if accept < α:
            x = x_star
        samples[i] = x
    return samples

def sample_plot(samples, sampled_function):
    figure, axis = pyplot.subplots(figsize=(12, 5))
    axis.set_xlabel("Sample")
    axis.set_ylabel("PDF")
    axis.set_title("Metropolis Sampling")
    _, bins, _ = axis.hist(samples, 50, density=True, color="#348ABD", alpha=0.6, label=f"Sampled Distribution", edgecolor="#348ABD", zorder=5)
    delta = (bins[-1] - bins[0]) / 200.0
    sample_distribution = [sampled_function(val) for val in numpy.arange(bins[0], bins[-1], delta)]
    axis.plot(numpy.arange(bins[0], bins[-1], delta), sample_distribution, color="#A60628", label=f"Sampled Function", zorder=6)
    axis.legend()

# %%

samples = metropolis(gauss, nsample=1000000)
sample_plot(samples, gauss)



# %%

samples = metropolis(cauchy, nsample=10000)
sample_plot(samples, cauchy)


# %%

samples = metropolis(weibull, nsample=1000000, x0=0.01)
sample_plot(samples, weibull)
