import numpy
from scipy import stats
from scipy import special

# distributions

def normal(x, σ=1.0, μ=0.0):
    ε = (x - μ)**2/(2.0*σ**2)
    return numpy.exp(-ε)/numpy.sqrt(2.0*numpy.pi*σ**2)

def weibull(k, λ=1.0):
    def f(x):
        if x < 0.0:
            return 0.0
        return (k/λ)*(x/λ)**(k-1)*numpy.exp(-(x/λ)**k)
    return f

def weibull_mean(k, λ=1.0):
    return λ*special.gamma(1+1.0/k)

def weibull_sigma(k, λ=1.0):
    return numpy.sqrt(λ**2*(special.gamma(1.0+2.0/k) - special.gamma(1.0+1.0/k)**2))

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

# utilities

def cummean(samples):
    nsample = len(samples)
    mean = numpy.zeros(nsample)
    mean[0] = samples[0]
    for i in range(1, len(samples)):
        mean[i] = (float(i) * mean[i - 1] + samples[i])/float(i + 1)
    return mean

def cumsigma(samples):
    nsample = len(samples)
    mean = cummean(samples)
    var = numpy.zeros(nsample)
    var[0] = samples[0]**2
    for i in range(1, len(samples)):
        var[i] = (float(i) * var[i - 1] + samples[i]**2)/float(i + 1)
    return numpy.sqrt(var-mean**2)

def autocorr(sample):
    return numpy.correlate(sample, sample, mode='same')