# %%

import numpy
from matplotlib import pyplot
from scipy import stats

%matplotlib inline

# %%

def metropolis_hastings(π, q, qsample, nsample=10000, x0=0.0):
    x = x0
    samples = numpy.zeros(nsample)
    for i in range(0, nsample):
        accept = numpy.random.rand()
        x_star = qsample(x)
        px_star = p(x_star)
        px = p(x)
        α = (px_star*q(x, x_star)) / (px*q(x_star, x))
        if accept < α:
            x = x_star
        samples[i] = x
    return samples
