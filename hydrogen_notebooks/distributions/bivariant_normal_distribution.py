# %%
%load_ext autoreload
%autoreload 2

import numpy
from matplotlib import pyplot
from glyfish import config

%matplotlib inline

pyplot.style.use(config.glyfish_style)

# %%

def bivariant_normal_pdf(μ1, μ2, σ1, σ2, ρ):
    def f(x1, x2):
        y1 = (x1 - μ1) / σ1
        y2 = (x2 - μ2) / σ2
        c = 2 * numpy.pi * σ1 * σ2 * numpy.sqrt(1.0 - ρ**2)
        ε = (y1**2 + y2**2 - ρ * y1 * y2) / (2.0 * (1.- ρ**2))
        return numpy.exp(ε) / c
    return f

def bivariant_normal_pdf_iso(μ1, μ2, σ1, σ2, ρ):
    def f(x1, x2):
        y1 = (x1 - μ1) / σ1
        y2 = (x2 - μ2) / σ2
        return (y1**2 + y2**2 - ρ * y1 * y2) / (2.0 * (1.- ρ**2))
     return f

def bivariant_normal_conditional_pdf(μ1, μ2, σ1, σ2, ρ):
    def f(x1, x2):
        y1 = (x1 - μ1)
        y2 = (x2 - μ2)
        c = 2 * numpy.pi * σ1 * numpy.sqrt(1.0 - ρ**2)
        ε = (y1**2 - ρ * σ1 * y2 / σ2) / (2.0 * (1.- ρ**2))
        return numpy.exp(ε) / c
    return f

def normal(x, σ=1.0, μ=0.0):
    ε = (x - μ)**2/(2.0*σ**2)
    return numpy.exp(-ε)/numpy.sqrt(2.0*numpy.pi*σ**2)

def bivariant_normal_conditional_pdf_generator(y, μ1, μ2, σ1, σ2, ρ):
    loc = μ1 + ρ * σ1 * (y - μ2) / σ2
    scale = numpy.sqrt((1.0 - ρ**2) * σ1**2)
    return numpy.random.normal(loc, scale)

# %%
