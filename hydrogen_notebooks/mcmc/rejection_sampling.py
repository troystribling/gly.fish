# %%
%load_ext autoreload
%autoreload 2

import numpy
import sympy

from matplotlib import pyplot
import scipy
from glyfish import config
from glyfish import stats

%matplotlib inline

pyplot.style.use(config.glyfish_style)

# %%

def rejection_sample(h, y_samples, c):
    nsamples = len(y_samples)
    u = numpy.random.rand(nsamples)
    accepted_mask = (u < h(y_samples) / c)
    return y_samples[accepted_mask], u, accepted_mask

def plot_sampled_pdf(title, f, samples, plot_name):
    figure, axis = pyplot.subplots(figsize=(10, 6))
    axis.set_prop_cycle(config.distribution_sample_cycler)
    axis.set_title(title)
    axis.set_ylabel("PDF")
    axis.set_ylim([0, 2.0])
    axis.set_yticks([0.2, 0.6, 1.0, 1.4, 1.8])
    axis.set_xlim([0.0, 1.6])
    axis.hist(samples, 30, density=True, rwidth=0.8, label=f"Samples", zorder=5)
    x = numpy.linspace(0.0, 1.6, 500)
    axis.plot(x, f(x), label=f"Target PDF", zorder=7)
    axis.legend(bbox_to_anchor=(0.3, 0.85))
    config.save_post_asset(figure, "rejection_sampling", plot_name)

def acceptance_plot(title, h, y_samples, ymax, xmax, legend_loc, plot_name):
    nsamples = len(y_samples)
    x_values = numpy.linspace(0.0, xmax, 500)
    samples, u, accepted_mask = rejection_sample(h, y_samples, ymax)
    rejected_mask = numpy.logical_not(accepted_mask)
    efficiency = 100.0 * (len(samples) / nsamples)
    figure, axis = pyplot.subplots(figsize=(10, 6))
    axis.set_title(title + f", Efficiency={format(efficiency, '2.0f')}%")
    axis.set_ylim([0, 1.05])
    axis.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    axis.set_xlim([0.0, 1.6])
    axis.plot(x_values, h(x_values) / ymax, zorder=5, lw=3, color="#5600C9")
    axis.scatter(y_samples[accepted_mask], u[accepted_mask], label="Accepted Samples", alpha=0.5, zorder=5)
    axis.scatter(y_samples[rejected_mask], u[rejected_mask], label="Rejected Samples", alpha=0.4, zorder=5)
    axis.legend(bbox_to_anchor=legend_loc, framealpha=0.9, edgecolor="#DDDDDD")
    config.save_post_asset(figure, "rejection_sampling", plot_name)

def mean_convergence(title, samples, μ, plot_name):
    nsamples = len(samples)
    time = range(nsamples)
    figure, axis = pyplot.subplots(figsize=(10, 6))
    axis.set_xlabel("Sample Number")
    axis.set_ylabel("μ")
    axis.set_title(title)
    axis.set_xlim([1.0, nsamples])
    axis.set_ylim( [0.5, 1.5])
    cmean = stats.cummean(samples)
    μs = numpy.full(len(samples), μ)
    axis.semilogx(time, μs, label="Target μ")
    axis.semilogx(time, cmean, label="Sampled μ")
    axis.legend(bbox_to_anchor=([0.9, 0.9]))
    config.save_post_asset(figure, "rejection_sampling", plot_name)

def sigma_convergence(title, samples, σ, plot_name):
    nsamples = len(samples)
    time = range(nsamples)
    figure, axis = pyplot.subplots(figsize=(10, 6))
    axis.set_xlabel("Sample Number")
    axis.set_ylabel("σ")
    axis.set_title(title)
    axis.set_xlim([1.0, nsamples])
    axis.set_ylim([0.0, 0.5])
    axis.set_yticks([0.1, 0.2, 0.3, 0.4])
    csigma = stats.cumsigma(samples)
    σs = numpy.full(len(samples), σ)
    axis.semilogx(time, σs, label="Target σ")
    axis.semilogx(time, csigma, label="Sampled σ")
    axis.legend(bbox_to_anchor=([0.95, 0.9]))
    config.save_post_asset(figure, "rejection_sampling", plot_name)

def normal(μ, σ):
    def f(x):
        ε = (x - μ)**2/(2.0*σ**2)
        return numpy.exp(-ε)/numpy.sqrt(2.0*numpy.pi*σ**2)
    return f

# %%
# Wiebull Target Density

k = 5.0
λ = 1.0
weibull_pdf = lambda v: (k/λ)*(v/λ)**(k-1)*numpy.exp(-(v/λ)**k)

# Uniform propsal density rejection method sampled function
# %%
xmax = 1.6
ymax = 2.0

x = numpy.linspace(0.001, xmax, 500)
cdf = numpy.cumsum(weibull_pdf(x)) * xmax/500.0

figure, axis = pyplot.subplots(figsize=(10, 6))
axis.set_xlim([0.0, xmax])
axis.set_ylim([0.0, ymax])
axis.set_yticks([0.2, 0.6, 1.0, 1.4, 1.8])
axis.set_title(f"Sampled Weibull Distribution, k={k}, λ={λ}")
axis.plot(x, weibull_pdf(x), label="PDF")
axis.plot(x, cdf, label=f"CDF")
axis.legend()

config.save_post_asset(figure, "rejection_sampling", "weibull_pdf")

# %%

x_values = numpy.linspace(0.0, xmax, 500)

target_pdf = weibull_pdf
proposal_pdf = 1.0/xmax
target_label = r"$f_X(y)=Weibull(k={k}, λ={λ})$".format(k=k, λ=λ)
proposal_label = r"$f_Y(y)=Uniform(0, {xmax})$".format(xmax=xmax)

h = lambda x: target_pdf(x) * xmax
figure, axis = pyplot.subplots(figsize=(10, 6))
axis.set_ylim([0, 3.75])
axis.set_yticks([0.5, 1.5, 2.5, 3.5])
axis.set_xlim([0.0, 1.6])
axis.set_title("Rejection Sampling Functions")
axis.plot(x_values, target_pdf(x_values), label=target_label)
axis.plot(x_values, numpy.full(len(x_values), 1.0/xmax), label=proposal_label)
axis.plot(x_values, h(x_values), label=r"$h(y)=f_X(y)/f_Y(y)$")
axis.legend(bbox_to_anchor=(0.52, 0.7), framealpha=0.0)
config.save_post_asset(figure, "rejection_sampling", "weibull_uniform_sampled_functions")

# %%

xmax = 1.6
ymax = 2.0
nsamples = 100000
y_samples = numpy.random.rand(nsamples) * xmax

samples, _, _ = rejection_sample(weibull_pdf, y_samples, ymax)

title = f"Sampled Weibull Density, k={k}, λ={λ}, Uniform Proposal"
plot_sampled_pdf(title, weibull_pdf, samples, "weibull_uniform_sampled_distribution")

# %%

xmax = 1.6
ymax = 2.0
nsamples = 10000
y_samples = numpy.random.rand(nsamples) * xmax
title = f"Sampled Weibull Density, Uniform Proposal"
acceptance_plot(title, weibull_pdf, y_samples, ymax, xmax, (0.4, 0.75), "weibull_uniform_efficiency")

# %%

μ = stats.weibull_mean(k, λ)
title = r"Weibull Density, Uniform Proposal, Cumulative μ convergence"
mean_convergence(title, samples, μ, "weibull_uniform_mean_convergence")

# %%

σ = stats.weibull_sigma(k, λ)
title = r"Weibull Density, Uniform Proposal, Cumulative σ convergence"
sigma_convergence(title, samples, σ, "weibull_uniform_sigma_convergence")

# Normal proposal density rejection sampled function
# %%

σ = 0.2
μ = 0.95
x_values = numpy.linspace(0.0, 2.0, 500)
target_pdf = weibull_pdf
proposal_pdf = normal(μ, σ)
target_label = r"$f_X(y)=Weibull(k={k}, λ={λ})$".format(k=k, λ=λ)
proposal_label = r"$f_Y(y)=Normal(μ={μ}, σ={σ})$".format(μ=μ, σ=σ)

h = lambda x: target_pdf(x) / proposal_pdf(x)
figure, axis = pyplot.subplots(figsize=(10, 6))
axis.set_ylim([0, 5.0])
axis.set_yticks([1.0, 2.0, 3.0, 4.0])
axis.set_xlim([0.0, 1.6])
axis.set_title("Rejection Sampling Functions")
axis.plot(x_values, target_pdf(x_values), label=target_label)
axis.plot(x_values, proposal_pdf(x_values), label=proposal_label)
axis.plot(x_values, h(x_values), label=r"$h(y)=f_X(y)/f_Y(y)$")
axis.legend(bbox_to_anchor=(0.95, 0.92))
config.save_post_asset(figure, "rejection_sampling", "weibull_normal_1_sampled_functions")

# %%

xmax = 1.6
nsamples = 100000
y_samples = numpy.random.normal(μ, σ, nsamples)
h = lambda x: weibull_pdf(x) / normal(μ, σ)(x)
ymax = h(x_values).max()

samples, _, accepted_mask = rejection_sample(h, y_samples, ymax)

title = f"Weibull Density, k={k}, λ={λ}, Normal Proposal"
plot_sampled_pdf(title, weibull_pdf, samples, "weibull_normal_1_sampled_distribution")

# %%

nsamples = 10000
y_samples = numpy.random.normal(μ, σ, nsamples)

acceptance_plot(title, h, y_samples, ymax, xmax, (0.6, 0.75), "weibull_normal_1_efficiency")

# %%

μ = stats.weibull_mean(k, λ)
title = r"Weibull Density, Normal Proposal, Cumulative μ convergence"
mean_convergence(title, samples, μ, "weibull_normal_1_mean_convergence")

# %%

σ = stats.weibull_sigma(k, λ)
title = r"Weibull Density, Normal Proposal, Cumulative σ convergence"
sigma_convergence(title, samples, σ, "weibull_normal_1_sigma_convergence")

# Normal proposal density rejection sampled function
# %%

σ = 0.25
μ = 0.9
x_values = numpy.linspace(0.0, xmax, 500)

target_pdf = weibull_pdf
proposal_pdf = normal(μ, σ)
target_label = r"$f_X(y)=Weibull(k={k}, λ={λ})$".format(k=k, λ=λ)
proposal_label = r"$f_Y(y)=Normal(μ={μ}, σ={σ})$".format(μ=μ, σ=σ)

h = lambda x: target_pdf(x) / proposal_pdf(x)
figure, axis = pyplot.subplots(figsize=(10, 6))
axis.set_ylim([0, 2.0])
axis.set_yticks([0.5, 1.0, 1.5, 2.0])
axis.set_xlim([0.0, 1.6])
axis.set_title("Rejection Sampling Functions")
axis.plot(x_values, target_pdf(x_values), label=target_label)
axis.plot(x_values, proposal_pdf(x_values), label=proposal_label)
axis.plot(x_values, h(x_values), label=r"$h(y)=f_X(y)/f_Y(y)$")
axis.legend(bbox_to_anchor=(0.52, 0.7), framealpha=0.0)
config.save_post_asset(figure, "rejection_sampling", "weibull_normal_3_sampled_functions")

# %%

xmax = 1.6
nsamples = 100000
y_samples = numpy.random.normal(μ, σ, nsamples)
h = lambda x: weibull_pdf(x) / normal(μ, σ)(x)
ymax = h(x_values).max()

samples, _, accepted_mask = rejection_sample(h, y_samples, ymax)

title = f"Weibull Density, k={k}, λ={λ}, Normal Proposal"
plot_sampled_pdf(title, weibull_pdf, samples, "weibull_normal_3_sampled_distribution")

# %%

xmax = 1.6
nsamples = 10000
ymax = h(x_values).max()
y_samples = numpy.random.normal(μ, σ, nsamples)
title = f"Weibull Density, k={k}, λ={λ}, Normal Proposal"

acceptance_plot(title, h, y_samples, ymax, xmax, (0.4, 0.73), "weibull_normal_3_efficiency")

# %%

μ = stats.weibull_mean(k, λ)
title = r"Weibull Distribution, Normal Proposal, μ convergence"
mean_convergence(title, samples, μ, "weibull_normal_3_mean_convergence")

# %%

σ = stats.weibull_sigma(k, λ)
title = r"Weibull Distribution, Normal Proposal, σ convergence"
sigma_convergence(title, samples, σ, "weibull_normal_3_sigma_convergence")
