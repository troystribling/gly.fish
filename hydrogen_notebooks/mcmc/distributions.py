# %%

import numpy
from matplotlib import pyplot
from scipy import stats

%matplotlib inline

# %%
# generators

def ar_1_kernel(x, y, stepsize):
    ε  = ((y -  x)**2) / (2.0 * stepsize**2)
    return numpy.exp(-ε) / numpy.sqrt(2 * numpy.pi * σ**2)

def gamma(x, y, stepsize):
    return scipy.stats.gamma.pdf(x, y/stepsize, scale=stepsize)

# sampled densities

def normal(x, σ=1.0, μ=0.0):
    ε = (x - μ)**2/(2.0*σ**2)
    return numpy.exp(-ε)/numpy.sqrt(2.0*numpy.pi*σ**2)

def weibull(x):
    return 0.544*x*numpy.exp(-(x/1.9)**2)

def cos(x):
    return numpy.cos(x)

# %%
# Gamma distribution

figure, axis = pyplot.subplots(figsize=(12, 5))
axis.set_xlabel("Sample", fontsize=14)
axis.tick_params(labelsize=13)
axis.set_ylabel("PDF", fontsize=14)
axis.set_title(title, fontsize=15)
axis.grid(True, zorder=5)
for i in range(len(pdf)):
    axis.plot(pdfs[i], x, color="#A60628", label=labels[i], lw="3", zorder=10)
axis.legend(fontsize=13)
