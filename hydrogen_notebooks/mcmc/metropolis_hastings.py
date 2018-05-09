# %%

import numpy
from matplotlib import pyplot
from scipy import stats
from glyfish import config

%matplotlib inline

pyplot.style.use(config.glyfish_style)

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
        α = (px_star*q(x_star, x, stepsize)) / (px*q(x, x_star, stepsize))
        if accept < α:
            accepted += 1
            x = x_star
        samples[i] = x
    return samples, accepted

# %%
# generators

def normal_generator(x, stepsize):
    return numpy.random.normal(x, stepsize)

def normal_independence_generator(μ):
    def f(x, stepsize):
        return numpy.random.normal(μ, stepsize)
    return f

def gamma_generator(x, stepsize):
    return stats.gamma.rvs(x/stepsize, scale=stepsize)

def uniform_generator(x, stepsize):
    return numpy.random.rand()

# proposed densities

def normal_proposal(x, y, stepsize):
    ε = ((y - x)**2) / (2.0 * stepsize**2)
    return numpy.exp(-ε) / numpy.sqrt(2 * numpy.pi * stepsize**2)

def gamma_proposal(x, y, stepsize):
    return stats.gamma.pdf(y, x/stepsize, scale=stepsize)

def uniform_proposal(x, y, stepsize):
    return 1.0

# sampled densities

def normal(x, σ=1.0, μ=0.0):
    ε = (x - μ)**2/(2.0*σ**2)
    return numpy.exp(-ε)/numpy.sqrt(2.0*numpy.pi*σ**2)

def weibull(k, λ=1.0):
    def f(x):
        if x < 0.0:
            return 0.0
        return (k/λ)*(x/λ)**(k-1)*numpy.exp(-(x/λ)**k)
    return f

def arcsine(x):
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return 1.0/(numpy.pi*numpy.sqrt(x*(1.0 - x)))

def bimodal_normal(x, μ=1.0, σ=1.0):
    return 0.5*(normal(x, σ, -2.0*μ) + normal(x, σ/2.0, 3.0*μ))

def gamma(a, σ=1.0):
    def f(x):
        if x <= 0 or a <= 0:
            return 0.0
        return stats.gamma.pdf(x, a, scale=σ)
    return f

#%%

nsample=100000
stepsize = 1.0
pdf = weibull(5.0)
samples, accepted = metropolis_hastings(pdf, normal_proposal, normal_generator, stepsize, nsample=nsample, x0=0.001)
accepted_percent = 100.0*float(accepted)/float(nsample)

figure, axis = pyplot.subplots(figsize=(12, 5))
axis.set_xlabel("X")
axis.set_ylabel("PDF")
axis.set_title(f"Weibull Distribution, Normal Proposal, Accepted {format(accepted_percent, '2.0f')}%")
_, bins, _ = axis.hist(samples, 50, density=True, color="#336699", alpha=0.6, label=f"Sampled Distribution", edgecolor="#336699", zorder=5)
delta = (bins[-1] - bins[0]) / 200.0
sample_distribution = [pdf(val) for val in numpy.arange(bins[0], bins[-1], delta)]
axis.plot(numpy.arange(bins[0], bins[-1], delta), sample_distribution, color="#A60628", label=f"Sampled Function", zorder=6)
axis.legend()

# %%

nsample=100000
stepsize = 1.0
pdf = weibull(5.0)
samples, accepted = metropolis_hastings(pdf, normal_proposal, normal_generator, stepsize, nsample=nsample, x0=0.001)
accepted_percent = 100.0*float(accepted)/float(nsample)

time = numpy.linspace(0, nsample - 1, nsample)
start = 5000
end = 5500

figure, axis = pyplot.subplots(figsize=(12, 5))
axis.set_xlabel("Time")
axis.set_ylabel("X")
axis.set_xlim([start, end])
axis.set_title(f"Wiebull Timeseries, Accepted {format(accepted_percent, '2.0f')}%")
axis.plot(time[start:end], samples[start:end], lw="1")

#%%

nsample=100000
stepsize = 1.0
pdf = weibull(5.0)
samples, accepted = metropolis_hastings(pdf, normal_proposal, normal_independence_generator(1.0), stepsize, nsample=nsample, x0=0.001)
accepted_percent = 100.0*float(accepted)/float(nsample)

figure, axis = pyplot.subplots(figsize=(12, 5))
axis.set_xlabel("X")
axis.set_ylabel("PDF")
axis.set_title(f"Weibull Distribution, Normal Proposal, Accepted {format(accepted_percent, '2.0f')}%")
_, bins, _ = axis.hist(samples, 50, density=True, color="#336699", alpha=0.6, label=f"Sampled Distribution", edgecolor="#336699", zorder=5)
delta = (bins[-1] - bins[0]) / 200.0
sample_distribution = [pdf(val) for val in numpy.arange(bins[0], bins[-1], delta)]
axis.plot(numpy.arange(bins[0], bins[-1], delta), sample_distribution, color="#A60628", label=f"Sampled Function", zorder=6)
axis.legend()

# %%

nsample=100000
stepsize = 1.0
pdf = weibull(5.0)
samples, accepted = metropolis_hastings(pdf, normal_proposal, normal_independence_generator(1.0), stepsize, nsample=nsample, x0=0.001)
accepted_percent = 100.0*float(accepted)/float(nsample)

time = numpy.linspace(0, nsample - 1, nsample)
start = 5000
end = 5500

figure, axis = pyplot.subplots(figsize=(12, 5))
axis.set_xlabel("Time")
axis.set_ylabel("X")
axis.set_xlim([start, end])
axis.set_title(f"Wiebull Timeseries, Accepted {format(accepted_percent, '2.0f')}%")
axis.plot(time[start:end], samples[start:end], lw="1")

#%%

nsample=100000
stepsize = 0.1
pdf = gamma(5.0)
samples, accepted = metropolis_hastings(pdf, gamma_proposal, gamma_generator, stepsize, nsample=nsample, x0=1.0)
accepted_percent = 100.0*float(accepted)/float(nsample)

figure, axis = pyplot.subplots(figsize=(12, 5))
axis.set_xlabel("X")
axis.set_ylabel("PDF")
axis.set_title(f"Gamma Distribution, Gamma Proposal, Accepted {format(accepted_percent, '2.0f')}%")
_, bins, _ = axis.hist(samples, 50, density=True, color="#336699", alpha=0.6, label=f"Sampled Distribution", edgecolor="#336699", zorder=5)
delta = (bins[-1] - bins[0]) / 200.0
sample_distribution = [pdf(val) for val in numpy.arange(bins[0], bins[-1], delta)]
axis.plot(numpy.arange(bins[0], bins[-1], delta), sample_distribution, color="#A60628", label=f"Sampled Function", zorder=6)
axis.legend()

#%%

nsample=100000
stepsize = 0.1
pdf = weibull(5.0)
samples, accepted = metropolis_hastings(pdf, gamma_proposal, gamma_generator, stepsize, nsample=nsample, x0=1.0)
accepted_percent = 100.0*float(accepted)/float(nsample)

figure, axis = pyplot.subplots(figsize=(12, 5))
axis.set_xlabel("X")
axis.set_ylabel("PDF")
axis.set_title(f"Weibull Distribution, Gamma Proposal, Accepted {format(accepted_percent, '2.0f')}%")
_, bins, _ = axis.hist(samples, 50, density=True, color="#336699", alpha=0.6, label=f"Sampled Distribution", edgecolor="#336699", zorder=5)
delta = (bins[-1] - bins[0]) / 200.0
sample_distribution = [pdf(val) for val in numpy.arange(bins[0], bins[-1], delta)]
axis.plot(numpy.arange(bins[0], bins[-1], delta), sample_distribution, color="#A60628", label=f"Sampled Function", zorder=6)
axis.legend()
