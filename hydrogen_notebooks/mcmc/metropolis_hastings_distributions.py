# %%

import numpy
from scipy import stats
from matplotlib import pyplot
from glyfish import config

%matplotlib inline

pyplot.style.use(config.glyfish_style)

# %%
# generators

def normal_random_walk_generator(x, stepsize):
    return x + numpy.random.normal(0.0, stepsize)

def normal_generator(x, stepsize):
    return numpy.random.normal(x, stepsize)

def gamma_generator(x, stepsize):
    return scipy.stats.gamma.rvs(x/stepsize, scale=stepsize)

def uniform_generator(x, stepsize):
    return numpy.random.rand()

# proposed densities

def normal_proposal(x, y, stepsize):
    ε = ((y - x)**2) / (2.0 * stepsize**2)
    return numpy.exp(-ε) / numpy.sqrt(2 * numpy.pi * stepsize**2)

def gamma_proposal(x, y, stepsize):
    return stats.gamma.pdf(x, y/stepsize, scale=stepsize)

# sampled densities

def normal(x, σ=1.0, μ=0.0):
    ε = (x - μ)**2/(2.0*σ**2)
    return numpy.exp(-ε)/numpy.sqrt(2.0*numpy.pi*σ**2)

def weibull(k, λ=1.0):
    def f(x):
        return (k/λ)*(x/λ)**(k-1)*numpy.exp(-(x/λ)**k)
    return f

def arcsine(x):
    return 1.0/(numpy.pi*numpy.sqrt(x*(1.0 - x)))

def bimodal_normal(x, μ=1.0, σ=1.0):
    return 0.5*(normal(x, σ, -2.0*μ) + normal(x, σ/2.0, 3.0*μ))

def gamma(x, k):
    return scipy.stats.gamma.pdf(x, k)

# %%
## normal

x = numpy.linspace(-7.0, 7.0, 500)
figure, axis = pyplot.subplots(figsize=(12, 5))
axis.set_xlabel("X")
axis.set_ylabel("PDF")
axis.set_ylim([0.0, 1.5])
axis.set_title("Normal Distribution")
axis.yaxis.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
σ = [0.3, 0.5, 1.0, 2.0]
μ = [-4.0, -2.0, 0.0, 2.0]
for i in range(len(σ)):
    pdf = [normal(j, σ[i], μ[i]) for j in x]
    axis.plot(x, pdf, label=f"σ={σ[i]}, μ={μ[i]}")
axis.legend()

# %%
## weibull

x = numpy.linspace(0.001, 3.0, 200)
k = [0.5, 1.0, 2.0, 5]

figure, axis = pyplot.subplots(figsize=(12, 5))
axis.set_xlabel("X")
axis.set_ylabel("PDF")
axis.set_xlim([0.0, 3.0])
axis.set_ylim([0.0, 2.0])
axis.yaxis.set_ticks([0.0, 0.5, 1.0, 1.5, 2.0])
axis.set_title(f"Weibull Distribution, λ=1.0")
for i in range(len(k)):
    pdf = weibull(k[i], 1.0)
    pdf_values = [pdf(j) for j in x]
    axis.plot(x, pdf_values, label=f"k={k[i]}")
axis.legend()


# %%
## arcsine

x = numpy.linspace(0.001, 0.999, 200)
pdf = [arcsine(j) for j in x]

figure, axis = pyplot.subplots(figsize=(12, 5))
axis.set_xlabel("X")
axis.set_ylabel("PDF")
axis.set_xlim([0.0, 1.0])
axis.set_title(f"Arcsine Distribution")
axis.plot(x, pdf)

# %%
## bimodal normal

x = numpy.linspace(-7.0, 7.0, 200)
pdf = [bimodal_normal(j, 1.2) for j in x]

figure, axis = pyplot.subplots(figsize=(12, 5))
axis.set_xlabel("X")
axis.set_ylabel("PDF")
axis.yaxis.set_ticks([0.0, 0.1, 0.2, 0.3, 0.4])
axis.set_ylim([0.0, 0.45])
axis.set_title(f"Normal Bimodal Distribution")
axis.plot(x, pdf)

# %%
## gamma disribution
x = numpy.linspace(0.001, 15.0, 200)

k = [0.5, 1.0, 2.0, 5]
figure, axis = pyplot.subplots(figsize=(12, 5))
axis.set_xlabel("X")
axis.set_ylabel("PDF")
axis.set_title(f"Gamma Distribution θ=1.0")
axis.set_ylim([0.0, 0.5])
axis.set_xlim([0.0, 15.0])
for i in range(len(k)):
    pdf = [gamma(j, k[i]) for j in x]
    axis.plot(x, pdf, label=f"k={format(k[i], '.2f')}")
axis.legend()

# %%
## normal generator

stepsize = 1.0
x0 = 0.0
nsamples = 5
yvals = numpy.linspace(-7.0, 7.0, 200)

figure, axis = pyplot.subplots(figsize=(12, 5))
axis.set_xlabel("X")
axis.set_ylabel("PDF")
axis.set_xlim([-7.0, 7.0])
axis.set_ylim([0.0, 0.45])
axis.set_title(f"Normal Proposal Distribution, stepsize={stepsize}, " + r"$X_0$" +f"={x0}")
x = x0
for i in range(nsamples):
    y = normal_generator(x, stepsize)
    pdf = [normal_proposal(x, j, stepsize) for j in yvals]
    axis.plot(yvals, pdf, label=f"step={i}, x={format(x, '.2f')}")
    x = y
axis.legend()

# %%
## normal generator

stepsize = 0.1
x0 = 1.0
nsamples = 5
yvals = numpy.linspace(0.0, 5.0, 200)

figure, axis = pyplot.subplots(figsize=(12, 5))
axis.set_xlabel("X")
axis.set_ylabel("PDF")
axis.set_xlim([0.0, 5.0])
axis.set_title(f"Gamma Proposal Distribution, stepsize={stepsize}, " + r"$X_0$" +f"={x0}")
x = x0
for i in range(nsamples):
    y = gamma_generator(x, stepsize)
    pdf = [gamma_proposal(x, j, stepsize) for j in yvals]
    axis.plot(yvals, pdf, label=f"step={i}, x={format(x, '.2f')}")
    x = y
axis.legend()
