# %%
%load_ext autoreload
%autoreload 2

import numpy
from matplotlib import pyplot
from scipy import stats
from glyfish import config
from glyfish import metropolis_hastings as mh
from glyfish import gplot
from glyfish import stats

%matplotlib inline

pyplot.style.use(config.glyfish_style)

#%%
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
    pdf = [stats.normal(j, σ[i], μ[i]) for j in x]
    axis.plot(x, pdf, label=f"σ={σ[i]}, μ={μ[i]}")
axis.legend()

# %%

x = numpy.linspace(-7.0, 7.0, 500)
figure, axis = pyplot.subplots(figsize=(12, 5))
axis.set_xlabel("X")
axis.set_ylabel("PDF")
axis.set_title("Normal Distribution")
axis.yaxis.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
σ = 2.0
μ = 1.0
nsamples = 10000
pdf = [stats.normal(i, σ, μ) for i in x]
samples = numpy.zeros(nsamples)
for i in range(nsamples):
    samples[i] = mh.normal_generator(μ, σ)
_, bins, _ = axis.hist(samples, 50, density=True, color="#A60628", alpha=0.6, label=f"Sampled Distribution", edgecolor="#A60628", zorder=5)
axis.plot(x, pdf, label=f"σ={σ}, μ={μ}", zorder=6)
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
    pdf = stats.weibull(k[i], 1.0)
    pdf_values = [pdf(j) for j in x]
    axis.plot(x, pdf_values, label=f"k={k[i]}")
axis.legend()


# %%
## arcsine

x = numpy.linspace(0.001, 0.999, 200)
pdf = [stats.arcsine(j) for j in x]

figure, axis = pyplot.subplots(figsize=(12, 5))
axis.set_xlabel("X")
axis.set_ylabel("PDF")
axis.set_xlim([0.0, 1.0])
axis.set_title(f"Arcsine Distribution")
axis.plot(x, pdf)

# %%
## bimodal normal

x = numpy.linspace(-7.0, 7.0, 200)
pdf = [stats.bimodal_normal(j, 1.2) for j in x]

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
    pdf = [stats.gamma(k[i])(j) for j in x]
    axis.plot(x, pdf, label=f"k={format(k[i], '.2f')}")
axis.legend()

# %%
# normal generator time series

stepsize = 1.0
x0 = 0.0
nsamples = 500

x = numpy.zeros(nsamples)
x[0] = x0
for i in range(1, nsamples):
    x[i] = mh.normal_generator(x[i-1], stepsize)

figure, axis = pyplot.subplots(figsize=(12, 5))
axis.set_xlabel("Time")
axis.set_ylabel("X")
axis.set_xlim([0, nsamples])
axis.set_title(f"Normal Generator, stepsize={format(stepsize, '2.2f')}")
axis.plot(range(nsamples), x, lw="1")

# %%
# normal generator distribution

stepsize = 1.0
x0 = 0.0
nsamples = 50000

x = numpy.zeros(nsamples)
x[0] = x0
for i in range(1, nsamples):
    x[i] = mh.normal_generator(x[i-1], stepsize)

figure, axis = pyplot.subplots(figsize=(12, 5))
axis.set_xlabel("X")
axis.set_ylabel("PDF")
axis.set_title(f"Normal Proposal Distribution, stepsize={format(stepsize, '2.2f')}")
_, bins, _ = axis.hist(x, 50, density=True, color="#336699", alpha=0.6, label=f"Generated Distribution", edgecolor="#336699", zorder=5)
delta = (bins[-1] - bins[0]) / 200.0
sample_distribution = [stats.normal(val, stepsize) for val in numpy.arange(bins[0], bins[-1], delta)]
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
    y = mh.normal_generator(x, stepsize)
    pdf = [mh.normal_proposal(x, j, stepsize) for j in yvals]
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
    x[i] = mh.gamma_generator(x[i-1], stepsize)

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
    y = mh.gamma_generator(x, stepsize)
    pdf = [mh.gamma_proposal(x, j, stepsize) for j in yvals]
    axis.plot(yvals, pdf, label=f"step={i}, x={format(x, '.2f')}")
    x = y
axis.legend()
