import numpy
from scipy import stats
from scipy import special

# Metropolis Hastings samplind algorithm
def metropolis_hastings(p, q, qsample, stepsize, nsample, x0):
    x = x0
    accepted = 0
    samples = numpy.zeros(nsample)
    for i in range(nsample):
        accept = numpy.random.rand()
        y_star = qsample(x, stepsize)
        py_star = p(y_star)
        px = p(x)
        α = (py_star*q(y_star, x, stepsize)) / (px*q(x, y_star, stepsize))
        if accept < α:
            accepted += 1
            x = y_star
        samples[i] = x
    return samples, accepted

# Component wise Metropolis Hastings samplind algorithm
def component_wise_metropolis_hastings(p, q, qsample, initial_state, ndim, stepsize, nsample, x0):
    accepted = 0
    samples = numpy.zeros((nsample, ndim))
    samples[0] = initial_state
    for i in range(1, nsample):
        x_previous = samples[i-1]
        for j in range(ndim):
            x_current = samples[i]
            x_j = x_previous[j]
            y_star = qsample(x_j, stepsize)
            accept = numpy.random.rand()
            py_star = p(y_star, j, x_previous, x_current),
            px = p(x_j, j, x_previous, x_current)
            α = (py_star*q(y_star, x_j, stepsize)) / (px*q(x_j, y_star, stepsize))
            if accept < α:
                accepted += 1
                x_j = y_star
            samples[i][j] = x_j
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
