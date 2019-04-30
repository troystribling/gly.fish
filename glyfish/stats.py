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

def bimodal_normal(x, σ=1.0, μ=1.0):
    return 0.5*(normal(x, σ, -2.0*μ) + normal(x, σ/2.0, 3.0*μ))

def gamma(a, θ=1.0):
    def f(x):
        if x <= 0 or a <= 0:
            return 0.0
        return stats.gamma.pdf(x, a, scale=θ)
    return f

def gamma_mean(a, σ):
    return a * σ

def gamma_sigma(a, σ):
    return numpy.sqrt(a * σ**2)

def bimodal_normal_mean(σ=1.0, μ=1.0):
    return 0.5*μ

def bimodal_normal_sigma(σ=1.0, μ=1.0):
    var = 0.5*(1.25*σ**2 + 13.0*μ**2) - 0.25*μ**2
    return numpy.sqrt(var)

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

def autocorrelate(x):
    n = len(x)
    x_shifted = x - x.mean()
    x_padded = numpy.concatenate((x_shifted, numpy.zeros(n-1)))
    x_fft = numpy.fft.fft(x_padded)
    h_fft = numpy.conj(x_fft) * x_fft
    ac = numpy.fft.ifft(h_fft)
    return ac[0:n]/ac[0]

def shift(a, n):
    result = numpy.zeros(a.shape)
    count = len(a)
    n = n % count
    for i in range(0, n):
        result[count - n + i] = a[i]
    for i in range(n, count):
        result[i-n] = a[i]
    return result
