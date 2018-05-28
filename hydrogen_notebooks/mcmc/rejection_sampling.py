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

def plot_sampled_pdf(title, f, samples, xlim, ylim):
    figure, axis = pyplot.subplots(figsize=(12, 5))
    axis.set_title(title)
    axis.set_xlim(xlim)
    axis.set_ylim(ylim)
    _, x_values, _ = axis.hist(samples, 50, density=True, alpha=0.6, label=f"Samples", edgecolor="#336699", zorder=6)
    x = numpy.linspace(0.0, xlim[-1], 500)
    axis.plot(x, f(x), label=f"Sampled Function", zorder=7)
    axis.legend()

def plot_sampling_functions(x_values, xlim, ylim, target_pdf, target_label, proposal_pdf, proposal_label):
    h = lambda x: target_pdf(x) / proposal_pdf(x)
    figure, axis = pyplot.subplots(figsize=(12, 5))
    axis.set_xlim(xlim)
    axis.set_ylim(ylim)
    axis.set_title("Rejection Sampled Functions")
    axis.plot(x_values, target_pdf(x_values), label=target_label)
    axis.plot(x_values, proposal_pdf(x_values), label=proposal_label)
    axis.plot(x_values, h(x_values), label=r"$h(x)=f(x)/g(x)$")
    axis.legend()

def acceptance_plot(title, h, x_samples, xmax, ymax, nsamples, legend_loc):
    x_values = numpy.linspace(0.0, xmax, 500)

    samples, y_samples, accepted_mask = rejection_sample(h, x_samples, ymax, nsamples)

    rejected_mask = numpy.logical_not(accepted_mask)
    efficiency = 100.0 * (len(samples) / nsamples)

    figure, axis = pyplot.subplots(figsize=(12, 5))
    axis.set_title(title + f", Efficiency={format(efficiency, '2.0f')}%")
    axis.grid(True, zorder=5)
    axis.set_xlim([0.0, xmax])
    axis.set_ylim([0.0, 1.0])
    axis.plot(x_values, h(x_values) / ymax, color="#A60628", zorder=5, lw=3)
    axis.scatter(x_samples[accepted_mask], y_samples[accepted_mask], label="Accepted Samples", color="#336699", alpha=0.5, zorder=5)
    axis.scatter(x_samples[rejected_mask], y_samples[rejected_mask], label="Rejected Samples", color="#00cc99", alpha=0.5, zorder=5)
    axis.legend(bbox_to_anchor=legend_loc)

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
xmax = 1.5
ymax = 2.0

x = numpy.linspace(0.001, xmax, 500)
cdf = numpy.cumsum(weibull_pdf(x)) * xmax/500.0

figure, axis = pyplot.subplots(figsize=(12, 5))
axis.set_xlim([0.0, xmax])
axis.set_ylim([0.0, ymax])
axis.set_title(f"Rejection Method Sampled Density, Weibull, k={k}, λ={λ}")
axis.plot(x, weibull_pdf(x), label="PDF")
axis.plot(x, cdf, label=f"CDF")
axis.legend()

# %%
xmax = 1.5
ymax = 2.0
nsamples = 50000
x_samples = numpy.random.rand(nsamples) * xmax

samples, _, _ = rejection_sample(weibull_pdf, x_samples, ymax, nsamples)

title = f"Weibull Density, Uniform Proposal, k={k}, λ={λ}"
plot_sampled_pdf(title, weibull_pdf, samples, [0.0, xmax], [0.0, ymax])

# %%

xmax = 1.5
ymax = 2.0
nsamples = 10000
x_samples = numpy.random.rand(nsamples) * xmax
title = f"Weibull Density, Uniform Proposal"
acceptance_plot(title, weibull_pdf, x_samples, xmax, ymax, nsamples, (0.3, 0.7))

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
ymax = h(x_values).max()
nsamples = 50000
x_samples = numpy.random.normal(μ, σ, nsamples)
h = lambda x: weibull_pdf(x) / normal(μ, σ)(x)

samples, y_samples, accepted_mask = rejection_sample(h, x_samples, ymax, nsamples)

title = f"Weibull Density, Normal Proposal, k={k}, λ={λ}"
plot_sampled_pdf(title, weibull_pdf, samples, [0.0, xmax], [0.0, 2.0])

# %%

nsamples = 10000
x_samples = numpy.random.normal(μ, σ, nsamples)

acceptance_plot(title, h, x_samples, xmax, ymax, nsamples, (0.6, 0.7))

# %%

xmax = 1.5
ymax = 1.5
nsamples = 50000
x_samples = numpy.random.normal(μ, σ, nsamples)
h = lambda x: weibull_pdf(x) / normal(μ, σ)(x)

samples, y_samples, accepted_mask = rejection_sample(h, x_samples, ymax, nsamples)

title = f"Weibull Density, Normal Proposal, k={k}, λ={λ}"
plot_sampled_pdf(title, weibull_pdf, samples, [0.0, xmax], [0.0, 2.0])

# %%

nsamples = 10000
x_samples = numpy.random.normal(μ, σ, nsamples)

acceptance_plot(title, h, x_samples, xmax, ymax, nsamples, (0.59, 0.5))

# Normal proposal density rejection sampled function
# %%

σ = 0.25
μ = 0.9
x_values = numpy.linspace(0.0, 2.0, 500)

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
ymax = h(x_values).max()
nsamples = 50000
x_samples = numpy.random.normal(μ, σ, nsamples)
h = lambda x: weibull_pdf(x) / normal(μ, σ)(x)

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
