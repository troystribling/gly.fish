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

# %%
## normal

x = numpy.linspace(-7.0, 7.0, 500)
figure, axis = pyplot.subplots(figsize=(10, 6))
axis.set_xlabel(r"$X$")
axis.set_ylabel("PDF")
axis.set_ylim([0.0, 1.5])
axis.set_title("Normal Distribution")
axis.yaxis.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
σ = [0.3, 0.5, 1.0, 2.0]
μ = [-4.0, -2.0, 0.0, 2.0]
for i in range(len(σ)):
    pdf = [stats.normal(j, σ[i], μ[i]) for j in x]
    axis.plot(x, pdf, label=f"σ={σ[i]}, μ={μ[i]}")
axis.legend(bbox_to_anchor=(0.9, 0.95))
config.save_post_asset(figure, "metropolis_hastings_sampling", "normal_distribution_parameters")

# %%
σ = 2.0
μ = 1.0

x = numpy.linspace(-7.0, 8.0, 500)
figure, axis = pyplot.subplots(figsize=(10, 6))
axis.set_xlabel(r"$X$")
axis.set_ylabel("PDF")
axis.set_xlim([-7.0, 8.0])
axis.set_yticks([0.0, 0.05, 0.1, 0.15, 0.2])
axis.set_title(f"Normal Distribution Samples: σ={σ}, μ={μ}")
axis.set_prop_cycle(config.distribution_sample_cycler)
nsamples = 50000
pdf = [stats.normal(i, σ, μ) for i in x]
samples = [mh.normal_generator(μ, σ) for _ in range(nsamples)]
axis.hist(samples, 50, density=True, rwidth=0.8, label=f"Samples", zorder=5)
axis.plot(x, pdf, label="Target PDF", zorder=6)
axis.legend()
config.save_post_asset(figure, "metropolis_hastings_sampling", "normal_distribution_samples")

# %%
# normal generator time series

stepsize = 1.0
x0 = 0.0
nsamples = 500

figure, axis = pyplot.subplots(3, sharex=True, figsize=(10, 7))
axis[2].set_xlabel("Time")
axis[0].set_title(f"Normal Proposal Markov Chain: stepsize={format(stepsize, '2.2f')}, " + r"$X_0$" +f"={x0}")
for i in range(3):
    x = numpy.zeros(nsamples)
    x[0] = x0
    for j in range(1, nsamples):
        x[j] = mh.normal_generator(x[j-1], stepsize)
    axis[i].set_xlim([0, nsamples])
    axis[i].set_ylim([-30.0, 30.0])
    axis[i].set_yticks([-20.0, -10.0, 0.0, 10.0, 20.0])
    axis[i].plot(range(nsamples), x, lw="2")

config.save_post_asset(figure, "metropolis_hastings_sampling", "normal_proposal_time_series")

# %%
## normal generator and proposal

stepsize = 1.0
x0 = 0.0
nsamples = 5
yvals = numpy.linspace(-7.0, 7.0, 200)

figure, axis = pyplot.subplots(figsize=(10, 7))
axis.set_xlabel(r"$X$")
axis.set_ylabel("PDF")
axis.set_xlim([-7.0, 7.0])
axis.set_ylim([0.0, 0.45])
axis.set_title(f"Normal Proposal Distribution: stepsize={stepsize}, " + r"$X_0$" +f"={x0}")
x = x0
for i in range(nsamples):
    y = mh.normal_generator(x, stepsize)
    pdf = [mh.normal_proposal(x, j, stepsize) for j in yvals]
    axis.plot(yvals, pdf, label=f"step={i}, X={format(x, '.2f')}")
    x = y
axis.legend()
config.save_post_asset(figure, "metropolis_hastings_sampling", "normal_proposal_examples")

# %%
## gamma disribution
x = numpy.linspace(0.001, 8.0, 200)

k = [50.0, 20.0, 10.0, 5.0]
θ = [0.1, 0.1, 0.1, 0.1]
figure, axis = pyplot.subplots(figsize=(10, 7))
axis.set_xlabel("X")
axis.set_ylabel("PDF")
axis.set_title(f"Gamma Distribution")
axis.set_ylim([-0.01, 2.0])
axis.set_xlim([-0.1, 8.0])
for i in range(len(k)):
    pdf = [stats.gamma(k[i], θ[i])(j) for j in x]
    axis.plot(x, pdf, label=f"k={format(k[i], '.2f')}, θ={format(θ[i], '.2f')}")
axis.legend(bbox_to_anchor=(0.9, 0.95))
config.save_post_asset(figure, "metropolis_hastings_sampling", "gamma_distribution_parameters")

# %%
# gamma generator time series

stepsize = 0.01
x0 = 1.0
nsamples = 500

figure, axis = pyplot.subplots(3, sharex=True, figsize=(10, 7))
axis[2].set_xlabel("Time")
axis[0].set_title(f"Gamma Proposal Markov Chain: stepsize={format(stepsize, '2.2f')}, " + r"$X_0$" +f"={x0}")
for i in range(3):
    x = numpy.zeros(nsamples)
    x[0] = x0
    for j in range(1, nsamples):
        x[j] = mh.gamma_generator(x[j-1], stepsize)
    axis[i].set_xlim([0, nsamples])
    axis[i].set_ylim([-0.5, 3.5])
    axis[i].set_yticks([0.0, 1.0, 2.0, 3.0])
    axis[i].plot(range(nsamples), x, lw="2")

config.save_post_asset(figure, "metropolis_hastings_sampling", "gamma_proposal_time_series")

# %%
## gamma generator and proposal

stepsize = 0.1
x0 = 1.0
nsamples = 5
yvals = numpy.linspace(0.0, 3.5, 200)

figure, axis = pyplot.subplots(figsize=(10, 7))
axis.set_xlabel(r"$X$")
axis.set_ylabel("PDF")
axis.set_xlim([0.0, 3.5])
axis.set_title(f"Gamma Proposal Distribution: stepsize={stepsize}, " + r"$X_0$" +f"={x0}")
x = x0
for i in range(nsamples):
    y = mh.gamma_generator(x, stepsize)
    pdf = [mh.gamma_proposal(x, j, stepsize) for j in yvals]
    axis.plot(yvals, pdf, label=f"step={i}, X={format(x, '.2f')}")
    x = y
axis.legend()

# %%
## weibull

x = numpy.linspace(0.001, 3.0, 200)
k = [0.01, 0.5, 1.0, 2.0, 5]

figure, axis = pyplot.subplots(figsize=(10, 6))
axis.set_xlabel(r"$X$")
axis.set_ylabel("PDF")
axis.set_xlim([0.0, 3.0])
axis.set_ylim([0.0, 2.0])
axis.yaxis.set_ticks([0.0, 0.5, 1.0, 1.5, 2.0])
axis.set_title(f"Weibull Distribution: λ=1.0")
for i in range(len(k)):
    pdf = stats.weibull(k[i], 1.0)
    pdf_values = [pdf(j) for j in x]
    axis.plot(x, pdf_values, label=f"k={k[i]}")
axis.legend()
config.save_post_asset(figure, "metropolis_hastings_sampling", "weibull_distribution_parameters")

# %%
## arcsine

x = numpy.linspace(0.001, 0.999, 200)
pdf = [stats.arcsine(j) for j in x]

figure, axis = pyplot.subplots(figsize=(10, 7))
axis.set_xlabel("X")
axis.set_ylabel("PDF")
axis.set_xlim([0.0, 1.0])
axis.set_title(f"Arcsine Distribution")
axis.plot(x, pdf)
config.save_post_asset(figure, "metropolis_hastings_sampling", "arcsine_distribution_parameters")

# %%
## bimodal normal

x = numpy.linspace(-7.0, 7.0, 200)
pdf = [stats.bimodal_normal(j, 1.2) for j in x]

figure, axis = pyplot.subplots(figsize=(10, 7))
axis.set_xlabel(r"$X$")
axis.set_ylabel("PDF")
axis.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
axis.set_ylim([0.0, 0.45])
axis.set_title(f"Normal Bimodal Distribution")
axis.plot(x, pdf)
config.save_post_asset(figure, "metropolis_hastings_sampling", "bimodeal_normal_distribution_parameters")
