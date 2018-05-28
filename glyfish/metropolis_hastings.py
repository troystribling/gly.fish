import numpy
from scipy import stats
from scipy import special

# Metrolois hastings samplind algorithm
def metropolis_hastings(p, q, qsample, stepsize, nsample=10000, x0=0.0):
    x = x0
    accepted = 0
    samples = numpy.zeros(nsample)
    for i in range(nsample):
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

# Proposal generators
def normal_generator(x, stepsize):
    return numpy.random.normal(x, stepsize)

def normal_independence_generator(μ):
    def f(x, stepsize):
        return numpy.random.normal(μ, stepsize)
    return f

def gamma_generator(x, stepsize):
    if x <= 0 or stepsize <= 0:
        return 0.0
    return stats.gamma.rvs(x/stepsize, scale=stepsize)

def uniform_generator(x, stepsize):
    return numpy.random.rand()

# Proposal distributions
def normal_proposal(x, y, stepsize):
    ε = ((y - x)**2) / (2.0 * stepsize**2)
    return numpy.exp(-ε) / numpy.sqrt(2 * numpy.pi * stepsize**2)

def gamma_proposal(x, y, stepsize):
    return stats.gamma.pdf(y, x/stepsize, scale=stepsize)

def uniform_proposal(x, y, stepsize):
    return 1.0

def normal_independence_proposal(μ):
    def f(x, stepsize):
        ε = ((y - μ)**2) / (2.0 * stepsize**2)
        return numpy.exp(-ε) / numpy.sqrt(2 * numpy.pi * stepsize**2)
    return f
