import numpy
from scipy import stats

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

def normal_proposal(x, y, stepsize):
    ε = ((y - x)**2) / (2.0 * stepsize**2)
    return numpy.exp(-ε) / numpy.sqrt(2 * numpy.pi * stepsize**2)

def gamma_proposal(x, y, stepsize):
    return stats.gamma.pdf(y, x/stepsize, scale=stepsize)

def uniform_proposal(x, y, stepsize):
    return 1.0

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
