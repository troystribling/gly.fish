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

def rejection_sample(f, x_samples, ymax, nsamples):
    y_samples = numpy.random.rand(nsamples)
    accepted_mask = (y_samples < f(x_samples) / ymax)
    return x_samples[accepted_mask], y_samples, accepted_mask

def plot_sampled_pdf(title, f, samples, plot_name):
    figure, axis = pyplot.subplots(figsize=(10, 6))
    axis.set_prop_cycle(config.distribution_sample_cycler)
    axis.set_title(title)
    axis.set_ylim([0, 2.0])
    axis.set_yticks([0.2, 0.6, 1.0, 1.4, 1.8])
    axis.set_xlim([0.0, 1.6])
    axis.hist(samples, 30, density=True, rwidth=0.8, label=f"Samples", zorder=5)
    x = numpy.linspace(0.0, 1.6, 500)
    axis.plot(x, f(x), label=f"Sampled Function", zorder=7)
    axis.legend(bbox_to_anchor=(0.38, 0.85))
    config.save_post_asset(figure, "rejection_sampling", plot_name)

def plot_sampling_functions(x_values, target_pdf, target_label, proposal_pdf, proposal_label):
    h = lambda x: target_pdf(x) / proposal_pdf(x)
    figure, axis = pyplot.subplots(figsize=(10, 6))
    axis.set_ylim([0, 2.0])
    axis.set_yticks([0.2, 0.6, 1.0, 1.4, 1.8])
    axis.set_xlim([0.0, 1.6])
    axis.set_title("Rejection Sampled Functions")
    axis.plot(x_values, target_pdf(x_values), label=target_label)
    axis.plot(x_values, proposal_pdf(x_values), label=proposal_label)
    axis.plot(x_values, h(x_values), label=r"$h(x)=f(x)/g(x)$")
    axis.legend()

def acceptance_plot(title, h, x_samples, nsamples, legend_loc, plot_name):
    x_values = numpy.linspace(0.0, xmax, 500)
    samples, y_samples, accepted_mask = rejection_sample(h, x_samples, ymax, nsamples)
    rejected_mask = numpy.logical_not(accepted_mask)
    efficiency = 100.0 * (len(samples) / nsamples)
    figure, axis = pyplot.subplots(figsize=(10, 6))
    axis.set_title(title + f", Efficiency={format(efficiency, '2.0f')}%")
    axis.set_ylim([0, 1.0])
    axis.set_yticks([0.2, 0.6, 0.8, 1.0])
    axis.set_xlim([0.0, 1.6])
    axis.plot(x_values, h(x_values) / ymax, zorder=5, lw=3, color="#5600C9")
    axis.scatter(x_samples[accepted_mask], y_samples[accepted_mask], label="Accepted Samples", alpha=0.5, zorder=5)
    axis.scatter(x_samples[rejected_mask], y_samples[rejected_mask], label="Rejected Samples", alpha=0.4, zorder=5)
    axis.legend(bbox_to_anchor=legend_loc, framealpha=0.9)
    config.save_post_asset(figure, "rejection_sampling", plot_name)

def mean_convergence(title, samples, μ, ylim, text_pos):
    nsamples = len(samples)
    time = range(nsamples)

    figure, axis = pyplot.subplots(figsize=(12, 6))
    axis.set_xlabel("Time")
    axis.set_ylabel("μ")
    axis.set_title(title)
    axis.set_xlim([1.0, nsamples])
    axis.set_ylim()
    cmean = stats.cummean(samples)
    μs = numpy.full(len(samples), μ)
    axis.semilogx(time, μs, label="Target μ")
    axis.semilogx(time, cmean, label="Sampled Distribution")
    axis.legend()

def sigma_convergense(title, samples, ylim, σ, text_pos):
    nsamples = len(samples)
    time = range(nsamples)

    figure, axis = pyplot.subplots(figsize=(12, 6))
    axis.set_xlabel("Time")
    axis.set_ylabel("σ")
    axis.set_title(title)
    axis.set_xlim([1.0, nsamples])
    axis.set_ylim([0.0, 0.5])
    csigma = stats.cumsigma(samples)
    σs = numpy.full(len(samples), σ)
    diff = numpy.mean(numpy.fabs(csigma - σs)[1000:])
    diff_text = r"$\Delta_{mean}=$"+r"${0:1.2E}$".format(diff)
    bbox = dict(boxstyle='square,pad=1', facecolor="#FFFFFF", edgecolor="lightgrey")
    axis.text(text_pos[0], text_pos[1], diff_text, fontsize=15, bbox=bbox)
    axis.semilogx(time, σs, label="Target σ", color="#000000")
    axis.semilogx(time, csigma, label=r"Sampled Distribution")
    axis.legend()

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
xmax = 1.6
ymax = 2.0
nsamples = 100000
x_samples = numpy.random.rand(nsamples) * xmax

samples, _, _ = rejection_sample(weibull_pdf, x_samples, ymax, nsamples)

title = f"Sampled Weibull Distribution k={k}, λ={λ}, Uniform Proposal"
plot_sampled_pdf(title, weibull_pdf, samples, "weibull_uniform_sampled_distribution")

# %%

xmax = 1.6
ymax = 2.0
nsamples = 10000
x_samples = numpy.random.rand(nsamples) * xmax
title = f"Sampled Weibull Density, Uniform Proposal"
acceptance_plot(title, weibull_pdf, x_samples, nsamples, (0.4, 0.75), "weibull_uniform_efficiency")

# %%

μ = stats.weibull_mean(k, λ)

title = r"Weibull Distribution, Uniform Proposal, Cumulative μ convergence"
mean_convergence(title, samples, μ, [2000.0, 1.045])

# %%

σ = stats.weibull_sigma(k, λ)
title = r"Weibull Distribution, Uniform Proposal, Cumulative σ convergence"
sigma_convergense(title, samples, σ, [2000.0, 0.37])

# Normal proposal density rejection sampled function
# %%

σ = 0.2
μ = 0.95
x_values = numpy.linspace(0.0, 2.0, 500)

plot_sampling_functions(x_values, [0.0, 2.0], [0.0, 5.0],
                        weibull_pdf, r"$f(x)=Webull(k={k}, λ={λ})$".format(k=k, λ=λ),
                        normal(μ, σ), r"$g(x)=Normal(μ={μ}, σ={σ})$".format(μ=μ, σ=σ))

# %%

xmax = 1.5
nsamples = 50000
x_samples = numpy.random.normal(μ, σ, nsamples)
h = lambda x: weibull_pdf(x) / normal(μ, σ)(x)
ymax = h(x_values).max()

samples, y_samples, accepted_mask = rejection_sample(h, x_samples, ymax, nsamples)

title = f"Weibull Density, Normal Proposal, k={k}, λ={λ}"
plot_sampled_pdf(title, weibull_pdf, samples, [0.0, xmax], [0.0, 2.0])

# %%

nsamples = 10000
x_samples = numpy.random.normal(μ, σ, nsamples)

acceptance_plot(title, h, x_samples, xmax, ymax, nsamples, (0.6, 0.7))

# %%

μ = stats.weibull_mean(k, λ)
title = r"Weibull Distribution, Normal Proposal, Cumulative μ convergence"
mean_convergence(title, samples, μ, [1900.0, 1.045])

# %%

σ = stats.weibull_sigma(k, λ)
title = r"Weibull Distribution, Normal Proposal, Cumulative σ convergence"
sigma_convergense(title, samples, σ, [1900.0, 0.37])


# %%

σ = 0.2
μ = 0.95
x_values = numpy.linspace(0.0, 2.0, 500)

xmax = 1.5
nsamples = 50000
x_samples = numpy.random.normal(μ, σ, nsamples)
h = lambda x: weibull_pdf(x) / normal(μ, σ)(x)
ymax = h(x_values).max()/3.0

samples, y_samples, accepted_mask = rejection_sample(h, x_samples, ymax, nsamples)

title = f"Weibull Density, Normal Proposal, k={k}, λ={λ}"
plot_sampled_pdf(title, weibull_pdf, samples, [0.0, xmax], [0.0, 2.0])

# %%

nsamples = 10000
x_samples = numpy.random.normal(μ, σ, nsamples)

acceptance_plot(title, h, x_samples, xmax, ymax, nsamples, (0.59, 0.5))

# %%

μ = stats.weibull_mean(k, λ)
title = r"Weibull Distribution, Normal Proposal, Cumulative μ convergence"
mean_convergence(title, samples, μ, [1900.0, 1.045])

# %%

σ = stats.weibull_sigma(k, λ)
title = r"Weibull Distribution, Normal Proposal, Cumulative σ convergence"
sigma_convergense(title, samples, σ, [1900.0, 0.37])

# Normal proposal density rejection sampled function
# %%

σ = 0.25
μ = 0.9
x_values = numpy.linspace(0.0, 2.0, 500)

plot_sampling_functions(x_values, [0.0, 2.0], [0.0, 2.0],
                        weibull_pdf, r"$f(x)=Webull(k={k}, λ={λ})$".format(k=k, λ=λ),
                        normal(μ, σ), r"$g(x)=Normal(μ={μ}, σ={σ})$".format(μ=μ, σ=σ))


# %%

xmax = 1.5
nsamples = 50000
x_samples = numpy.random.normal(μ, σ, nsamples)
h = lambda x: weibull_pdf(x) / normal(μ, σ)(x)
ymax = h(x_values).max()

samples, y_samples, accepted_mask = rejection_sample(h, x_samples, ymax, nsamples)

title = f"Weibull Density, Normal Proposal, k={k}, λ={λ}"
plot_sampled_pdf(title, weibull_pdf, samples, [0.0, xmax], [0.0, 2.0])

# %%

xmax = 1.5
nsamples = 10000
ymax = h(x_values).max()
x_samples = numpy.random.normal(μ, σ, nsamples)
title = f"Weibull Density, Normal Proposal, k={k}, λ={λ}"

acceptance_plot(title, h, x_samples, xmax, ymax, nsamples, (0.3, 0.7))

# %%

μ = stats.weibull_mean(k, λ)
title = r"Weibull Distribution, Normal Proposal, μ convergence"
mean_convergence(title, samples, μ, [2000.0, 1.045])

# %%

σ = stats.weibull_sigma(k, λ)
title = r"Weibull Distribution, Normal Proposal, σ convergence"
sigma_convergense(title, samples, σ, [2000.0, 0.37])
