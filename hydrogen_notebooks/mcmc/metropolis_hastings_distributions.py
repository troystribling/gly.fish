# %%

import numpy
from matplotlib import pyplot
from scipy import stats

%matplotlib inline

# %%
# generators

def normal_random_walk_generator(x, stepsize):
    return x + numpy.random.normal(0.0, stepsize)

def normal_generator(x, stepsize):
    return numpy.random.normal(x, stepsize)

def gamma_generator(x, stepsize):
    return scipy.stats.gamma.rvs(x/nsteps, scale=stepsize)

def uniform_generator(x, stepsize):
    return numpy.random.rand()

# %%
# proposed densities

def normal_random_walk(x, y, stepsize):
    ε = ((y -  x)**2) / (2.0 * stepsize**2)
    return numpy.exp(-ε) / numpy.sqrt(2 * numpy.pi * σ**2)

def gamma(x, y, stepsize):
    return scipy.stats.gamma.pdf(x, y/stepsize, scale=stepsize)

# sampled densities

def normal(x, σ=1.0, μ=0.0):
    ε = (x - μ)**2/(2.0*σ**2)
    return numpy.exp(-ε)/numpy.sqrt(2.0*numpy.pi*σ**2)

def weibull(x, k, λ):
    return (k/λ)*(x/λ)**(k-1)*numpy.exp(-(x/λ)**k)

def arcsine(x):
    return 1.0/(numpy.pi*numpy.sqrt(x*(1.0 - x)))

def bimodal_normal(x, μ=0.0, σ=1.0):
    return 0.5*(normal(x, σ, μ) + normal(x, σ/2.0, 3.0*μ)/2.0)

# %% normal

x = numpy.linspace(-7.0, 7.0, 200)
pyplot.style.use('bmh')
figure, axis = pyplot.subplots(figsize=(12, 5))
axis.set_xlabel("X", fontsize=14)
axis.tick_params(labelsize=13)
axis.set_ylabel("PDF", fontsize=14)
axis.set_ylim([0.0, 1.0])
axis.set_title("Normal Distribution", fontsize=15)
axis.grid(True, zorder=5)
σ = [0.5, 1.0, 2.0]
μ = [-2.0, 0.0, 2.0]
for i in range(3):
    pdf = [normal(j, σ[i], μ[i]) for j in x]
    axis.plot(x, pdf, label=f"σ={σ[i]}, μ={μ[i]}", lw=3, zorder=10)
axis.legend(fontsize=13)

# %% weibull

x = numpy.linspace(0.0, 7.0, 200)
pyplot.style.use('bmh')
figure, axis = pyplot.subplots(figsize=(12, 5))
axis.set_xlabel("X", fontsize=14)
axis.tick_params(labelsize=13)
axis.set_ylabel("PDF", fontsize=14)
axis.set_title("Weibull Distribution", fontsize=15)
axis.grid(True, zorder=5)
k = [1.0, 1.0, 1.0]
λ = [1.0, 2.0, 1.0]
for i in range(3):
    pdf = [weibull(j, k[i], λ[i]) for j in x]
    axis.plot(x, pdf, label=f"k={k[i]}, λ={λ[i]}", lw=3, zorder=10)
axis.legend(fontsize=13)
