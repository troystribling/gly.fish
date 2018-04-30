# %%

import numpy
from matplotlib import pyplot
from scipy import stats

%matplotlib inline

# %%

def metropolis_hastings(π, q, qsample, nsample=10000, x0=0.0):
    x = x0
    accepted = 0
    samples = numpy.zeros(nsample)
    for i in range(0, nsample):
        accept = numpy.random.rand()
        x_star = qsample(x)
        px_star = p(x_star)
        px = p(x)
        α = (px_star*q(x_star, x)) / (px*q(x, x_xstar))
        if accept < α:
            accepted += 1
            x = x_star
        samples[i] = x
    return samples, accepted

def metropolis(p, qsample, nsample=10000, x0=0.0):
    x = x0
    samples = numpy.zeros(nsample)
    accepted = 0
    for i in range(0, nsample):
        x_star = qsample(x)
        accept = numpy.random.rand()
        px_star = p(x_star)
        px = p(x)
        α = px_star / px
        if accept < α:
            accepted += 1
            x = x_star
        samples[i] = x
    return samples, accepted

def sample_plot(samples, sampled_function, title):
    figure, axis = pyplot.subplots(figsize=(12, 5))
    axis.set_xlabel("Sample", fontsize=14)
    axis.tick_params(labelsize=13)
    axis.set_ylabel("PDF", fontsize=14)
    axis.set_title("Metropolis Sampling", fontsize=15)
    axis.grid(True, zorder=5)
    _, bins, _ = axis.hist(samples, 50, density=True, color="#348ABD", alpha=0.6, label=f"Sampled Distribution", edgecolor="#348ABD", lw="3", zorder=10)
    delta = (bins[-1] - bins[0]) / 200.0
    sample_distribution = [sampled_function(val) for val in numpy.arange(bins[0], bins[-1], delta)]
    axis.plot(numpy.arange(bins[0], bins[-1], delta), sample_distribution, color="#A60628", label=f"Sampled Function", lw="3", zorder=10)
    axis.legend(fontsize=13)

#%%

def normal_random_walk(x, σ=1.0, μ=0.0):
    return x + numpy.random.normal(μ, σ)

def normal(x, σ=1.0, μ=0.0):
    return numpy.exp(-(x - μ)**2/(2.0*σ**2))/numpy.sqrt(2.0*numpy.pi*σ**2)

#%%

nsample=10000
samples, accepted = metropolis(normal, normal_random_walk, nsample=nsample, x0=0.0)
sample_plot(samples, normal, "Metropolis Sampling")
