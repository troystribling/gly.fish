# %%

import numpy
from matplotlib import pyplot
from scipy import stats

%matplotlib inline

# %%

def metropolis_hastings(p, q, qsample, stepsize, nsample=10000, x0=0.0):
    x = x0
    accepted = 0
    samples = numpy.zeros(nsample)
    for i in range(0, nsample):
        accept = numpy.random.rand()
        x_star = qsample(x, stepsize)
        px_star = p(x_star)
        px = p(x)
        α = (px_star*q(x_star, x)) / (px*q(x, x_star))
        if accept < α:
            accepted += 1
            x = x_star
        samples[i] = x
    return samples, accepted

def metropolis(p, qsample, stepsize, nsample=10000, x0=0.0):
    x = x0
    samples = numpy.zeros(nsample)
    accepted = 0
    for i in range(0, nsample):
        x_star = qsample(x, stepsize)
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
    axis.set_title(title, fontsize=15)
    axis.grid(True, zorder=5)
    _, bins, _ = axis.hist(samples, 50, density=True, color="#348ABD", alpha=0.6, label=f"Sampled Distribution", edgecolor="#348ABD", lw="3", zorder=10)
    delta = (bins[-1] - bins[0]) / 200.0
    sample_distribution = [sampled_function(val) for val in numpy.arange(bins[0], bins[-1], delta)]
    axis.plot(numpy.arange(bins[0], bins[-1], delta), sample_distribution, color="#A60628", label=f"Sampled Function", lw="3", zorder=10)
    axis.legend(fontsize=13)

# %%
# generators

def normal_random_walk_generator(x, stepsize):
    return x + numpy.random.normal(0.0, stepsize)

def normal_generator(x, stepsize):
    return numpy.random.normal(x, stepsize)

def gamma_generator(x, stepsize):
    return scipy.stats.gamma.rvs(x/stepsize, scale=stepsize)

def uniform_generator(x, stepsize):
    return numpy.random.rand()

# %%
# proposed densities

def normal_random_walk(y, x, stepsize):
    ε = ((y -  x)**2) / (2.0 * stepsize**2)
    return numpy.exp(-ε) / numpy.sqrt(2 * numpy.pi * σ**2)

def gamma(y, x, stepsize):
    return scipy.stats.gamma.pdf(y, x/stepsize, scale=stepsize)

def uniform(y, x, stepsize):
    return 1.0

# %%
# sampled densities

def normal(x, μ=0.0, σ=1.0):
    ε = (x - μ)**2/(2.0*σ**2)
    return numpy.exp(-ε)/numpy.sqrt(2.0*numpy.pi*σ**2)

def weibull(x, k, λ):
    return (k/λ)*(x/λ)**(k-1)*numpy.exp(-(x/λ)**k)

def arcsine(x):
    return 1.0/(numpy.pi*numpy.sqrt(x*(1.0 - x)))

def bimodal_normal(x, μ=0.0, σ=1.0):
    return 0.5*(normal(x, σ, μ) + normal(x, σ/2.0, 3.0*μ)/2.0)

#%%

nsample=10000
samples, accepted = metropolis(weibull, normal_random_walk, nsample=nsample, x0=1.0)
sample_plot(samples, weibull, "Metropolis Sampling: Weibull, Randomwalk")

#%%

nsample=10000
samples, accepted = metropolis_hastings(weibull, ar_1_kernel, normal_random_walk, nsample=nsample, x0=1.0)
sample_plot(samples, weibull, "Metropolis-Hastings Sampling: Weibull, Randomwalk")
