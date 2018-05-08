# %%

import numpy
from scipy import stats
from matplotlib import pyplot
from glyfish import config

%matplotlib inline

pyplot.style.use(config.glyfish_style)

# %%
# generators

def normal_generator(x, stepsize):
    return numpy.random.normal(x, stepsize)

def normal_independence_generator(μ):
    def f(x, stepsize):
        return numpy.random.normal(μ, stepsize)
    return f

def gamma_generator(x, stepsize):
    print(x, stepsize)
    return stats.gamma.rvs(x/stepsize, scale=stepsize)

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
# normal generator

stepsize = 1.0
x0 = 0.0
nsamples = 500

x = numpy.zeros(nsamples)
x[0] = x0
for i in range(1, nsamples):
    x[i] = normal_generator(x[i-1], stepsize)

figure, axis = pyplot.subplots(figsize=(12, 5))
axis.set_xlabel("Time")
axis.set_ylabel("X")
axis.set_xlim([0, nsamples])
axis.set_title(f"Normal Generator, stepsize={format(stepsize, '2.2f')}")
axis.plot(range(nsamples), x, lw="1")

# %%
# normal generator

stepsize = 1.0
x0 = 0.0
nsamples = 50000

x = numpy.zeros(nsamples)
x[0] = x0
for i in range(1, nsamples):
    x[i] = normal_generator(x[i-1], stepsize)

figure, axis = pyplot.subplots(figsize=(12, 5))
axis.set_xlabel("X")
axis.set_ylabel("PDF")
axis.set_title(f"Normal Proposal Distribution, stepsize={format(stepsize, '2.2f')}")
_, bins, _ = axis.hist(x, 50, density=True, color="#336699", alpha=0.6, label=f"Generated Distribution", edgecolor="#336699", zorder=5)
delta = (bins[-1] - bins[0]) / 200.0
sample_distribution = [normal(val, stepsize) for val in numpy.arange(bins[0], bins[-1], delta)]
# axis.plot(numpy.arange(bins[0], bins[-1], delta), sample_distribution, color="#A60628", label=f"Sampled Function", zorder=6)
axis.legend()

# %%
## normal generator and proposal

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
# gamma generator

stepsize = 0.1
x0 = 2.0
nsamples = 100

x = numpy.zeros(nsamples)
x[0] = x0
for i in range(1, nsamples):
    x[i] = gamma_generator(x[i-1], stepsize)

# figure, axis = pyplot.subplots(figsize=(12, 5))
# axis.set_xlabel("Time")
# axis.set_ylabel("X")
# axis.set_xlim([0, nsamples])
# axis.set_title(f"Gamma Generator, stepsize={format(stepsize, '2.2f')}")
# axis.plot(range(nsamples), x, lw="1")

# %%
## gamma generator and proposal

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
