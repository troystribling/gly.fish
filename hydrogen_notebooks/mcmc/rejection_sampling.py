# %%
%load_ext autoreload
%autoreload 2

import numpy
import sympy

from matplotlib import pyplot
from scipy import stats
from glyfish import config

%matplotlib inline

pyplot.style.use(config.glyfish_style)

# Uniform rejection method sampled function
# %%

f = lambda v: numpy.exp(-(v-1.0)**2 / (2.0 * v)) * (v + 1) / 12.0
x = numpy.linspace(0.001, 10, 100)
pdf = f(x)
cdf = numpy.cumsum(pdf) * 0.1

figure, axis = pyplot.subplots(figsize=(12, 5))
axis.set_xlabel("Value")
axis.set_ylabel("Sampled")
axis.set_title("Rejection Method Sampled Functions")
axis.grid(True, zorder=5)
axis.plot(x, pdf, color="#A60628", label=f"Sampled PDF", lw="3", zorder=10)
axis.plot(x, cdf, color="#348ABD", label=f"Sampled CDF", lw="3", zorder=10)
axis.legend()

# %%

M = 0.3
nsamples = 10000

x_samples = numpy.random.rand(nsamples) * 10
y_samples = numpy.random.rand(nsamples)
accepted_mask = (y_samples < f(x_samples) / M)
accepted_samples = x_samples[accepted_mask]

figure, axis = pyplot.subplots(figsize=(12, 5))
axis.set_xlabel("Value")
axis.set_ylabel("PDF")
axis.set_title("Rejection Sampling")
axis.grid(True, zorder=5)
_, x_values, _ = axis.hist(accepted_samples, 50, density=True, color="#336699", alpha=0.6, label=f"Sampled Density", edgecolor="#336699", lw="3", zorder=10)
axis.plot(x_values, f(x_values), color="#A60628", label=f"Sampled Function", lw="3", zorder=10)
axis.legend()

# %%

rejected_mask = numpy.logical_not(accepted_mask)
efficiency = 100.0 * (len(x_samples[accepted_mask]) / nsamples)

figure, axis = pyplot.subplots(figsize=(12, 5))
axis.set_xlabel("Value")
axis.set_ylabel("PDF")
axis.set_title(f"Rejection Sampling, Efficiency={format(efficiency, '2.0f')}%")
axis.grid(True, zorder=5)
axis.set_xlim([0.0, 10.0])
axis.set_ylim([0.0, 1.0])
axis.plot(x_values, f(x_values) / M, color="#A60628", lw="3", zorder=10)
axis.scatter(x_samples[accepted_mask], y_samples[accepted_mask], label="Accepted Samples", color="#336699", alpha=0.5)
axis.scatter(x_samples[rejected_mask], y_samples[rejected_mask], label="Rejected Samples", color="#00cc99", alpha=0.5)
axis.legend(bbox_to_anchor=(0.7, 0.7))

# Instead of using a uniform sample of values use a distribution that is similar to target to increase efficiency
# %%

chi = stats.chi2(4)
h = lambda x: f(x) / chi.pdf(x)
figure, axis = pyplot.subplots(figsize=(12, 5))
axis.set_xlabel("Value")
axis.set_ylabel("PDF")
axis.set_title(f"Rejection Sampled Functions")
axis.grid(True, zorder=5)
axis.plot(x_values, f(x_values), color="#A60628", lw="3", label="$f(x)$", zorder=10)
axis.plot(x_values, chi.pdf(x_values), color="#348ABD", lw="3", label="$g(x)$", zorder=10)
axis.legend()


# %%
figure, axis = pyplot.subplots(figsize=(12, 5))
axis.set_xlabel("Value")
axis.set_ylabel("PDF")
axis.set_title(f"Rejection Sampled Functions")
axis.grid(True, zorder=5)
axis.plot(x_values, h(x_values), color="#A60628", lw="3", label="$h(x)$", zorder=10)
axis.legend()

# %%
hmax = h(x_values).max()
x_samples = chi.rvs(nsamples)
y_samples = numpy.random.rand(nsamples)
accepted_mask = (y_samples <= h(x_samples) / hmax)
efficiency = 100.0 * (len(x_samples[accepted_mask]) / nsamples)

figure, axis = pyplot.subplots(figsize=(12, 5))
axis.set_xlabel("Value")
axis.set_ylabel("PDF")
axis.set_title(f"Rejection Sampling, Efficiency={efficiency}%")
axis.grid(True, zorder=5)
_, x_values, _ = axis.hist(x_samples[accepted_mask], 50, density=True, color="#348ABD", alpha=0.6, label=f"Sampled Density", edgecolor="#348ABD", lw="3", zorder=10)
axis.plot(x_values, f(x_values), color="#A60628", label=f"Sampled Function", lw="3", zorder=10)
axis.legend()

# %%

rejected_mask = numpy.logical_not(accepted_mask)
efficiency = 100.0 * (len(x_samples[accepted_mask]) / nsamples)

figure, axis = pyplot.subplots(figsize=(12, 5))
axis.set_xlabel("Value")
axis.set_ylabel("PDF")
axis.set_title(f"Rejection Sampling, Efficiency={efficiency}%")
axis.grid(True, zorder=5)
axis.plot(x_values, h(x_values) / hmax, color="#A60628", lw="3", zorder=10)
axis.scatter(x_samples[accepted_mask], y_samples[accepted_mask], label="Accepted Samples", color="#348ABD", alpha=0.5)
axis.scatter(x_samples[rejected_mask], y_samples[rejected_mask], label="Rejected Samples", color="#0BAA54", alpha=0.5)
