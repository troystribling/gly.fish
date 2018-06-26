# %%
%load_ext autoreload
%autoreload 2

import numpy

from matplotlib import pyplot
from glyfish import config
from glyfish import stats

%matplotlib inline

pyplot.style.use(config.glyfish_style)

# %%
# Inverse CDF Descrete random variable

df = numpy.array([1/12, 1/12, 1/6, 1/6, 1/12, 5/12])
cdf = numpy.cumsum(df)
x = numpy.array(range(len(df)))

nsamples = 100000
cdf_samples = numpy.random.rand(nsamples)
df_samples = [numpy.flatnonzero(cdf >= cdf_samples[i])[0] for i in range(nsamples)]
multinomial_samples = numpy.random.multinomial(nsamples, df, size=1)/nsamples

# %%

figure, axis = pyplot.subplots(figsize=(10, 6))
axis.set_xlabel("Value")
axis.set_ylabel("Probability")
axis.set_ylim([0, 1.1])
axis.set_title("Inverse CDF Sampled Discrete Distribution")
axis.bar(x - 0.2, df, 0.4, label=f"Distribution", zorder=5)
axis.bar(x + 0.2, cdf, 0.4, label=f"CDF", zorder=5)
axis.legend(bbox_to_anchor=(0.3, 0.85))

config.save_post_asset(figure, "inverse_cdf_sampling", "discrete_cdf")

# %%

bins = numpy.array([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
hist, _ = numpy.histogram(df_samples, bins)
p = hist/numpy.sum(hist)

figure, axis = pyplot.subplots(figsize=(10, 6))
axis.set_xlabel("Value")
axis.set_ylabel("Probability")
axis.set_ylim([0, 0.5])
axis.set_title("Inverse CDF Sampled Discrete Distribution")
axis.bar(x - 0.2, df, 0.4, label=f"Distribution", zorder=5)
axis.bar(x + 0.2, p, 0.4, label=f"Samples", zorder=5)
axis.set_prop_cycle(config.bar_chart_cycler)
axis.legend(bbox_to_anchor=(0.3, 0.85))

# %%
# Inverse CDF sampling for exponential

nsamples = 10000
cdf_inv = lambda v: numpy.log(1.0 / (1.0 - v))
pdf = lambda v: numpy.exp(-v)

samples = [cdf_inv(u) for u in numpy.random.rand(nsamples)]
x = numpy.linspace(0.0, 6.0, 500)
dx = 6.0/499.0

# %%

figure, axis = pyplot.subplots(figsize=(10, 6))
axis.set_xlabel("n")
axis.set_ylim([0, 1.1])
axis.set_xlim([0.0, 6.0])
axis.set_ylabel("Probability")
axis.set_title("Exponential Distribution")
axis.plot(x, pdf(x), label="PDF", zorder=5)
axis.plot(x, dx*numpy.cumsum(pdf(x)), label="CDF", zorder=5)
axis.legend(bbox_to_anchor=(0.9, 0.6))

# %%

figure, axis = pyplot.subplots(figsize=(12, 5))
axis.set_xlabel("Sample")
axis.set_ylabel("PDF")
axis.set_xlim([0.0, 6.0])
axis.set_title("Inverse CDF Sampled Exponential Distribution")
axis.set_prop_cycle(config.distribution_sample_cycler)
axis.hist(samples, 50, density=True, rwidth=0.8, label=f"Samples", zorder=5, color="#329EFF")
axis.plot(x, pdf(x), label=f"Sampled PDF", zorder=6, color="#003B6F")
axis.legend()

# %%
# Inverse CDF sampling from weibull distribution

k = 5.0
λ = 1.0
nsample = 10000
pdf = stats.weibull(k, λ)
cdf_inv = lambda u: λ * (numpy.log(1.0/(1.0 - u)))**(1.0/k)
x = numpy.linspace(0.001, 1.6, 100)
dx = 1.6/99.0

samples = [cdf_inv(u) for u in numpy.random.rand(nsample)]

# %%

figure, axis = pyplot.subplots(figsize=(10, 6))
axis.set_xlabel("n")
axis.set_ylim([0, 2.0])
axis.set_xlim([0.0, 1.6])
axis.set_ylabel("Probability")
axis.set_title("Weibull Distribution")
pdf_values = [pdf(v) for v in x]
axis.plot(x, pdf_values, label="PDF", zorder=5)
axis.plot(x, dx*numpy.cumsum(pdf_values), label="CDF", zorder=5)
axis.legend()

# %%

figure, axis = pyplot.subplots(figsize=(12, 5))
axis.set_xlabel("Sample")
axis.set_ylabel("PDF")
axis.set_ylim([0, 2.0])
axis.set_xlim([0.0, 1.6])
axis.set_title("Inverse CDF Sampled Weibill Distribution")
axis.plot(x, pdf_values, label=f"Sampled Function", zorder=6)
axis.hist(samples, 40, density=True, rwidth=0.8 , label=f"Sampled Density", zorder=5)
pdf_values = [pdf(u) for u in x]
axis.legend()

# %%

μ = stats.weibull_mean(k, λ)
title = r"Weibull Distribution, Inverse CDF Sampled, μ convergence"
time = range(nsamples)

figure, axis = pyplot.subplots(figsize=(12, 6))
axis.set_xlabel("Time")
axis.set_ylabel("μ")
axis.set_title(title)
axis.set_xlim([1.0, nsamples])
axis.set_ylim([0.0, 2.0])
axis.semilogx(time, numpy.full(nsamples, μ), label="Target μ", color="#000000")
axis.semilogx(time, stats.cummean(samples), label="Sampled Distribution")
axis.legend()

# %%

σ = stats.weibull_sigma(k, λ)
title = r"Weibull Distribution, Inverse CDF Sampled, σ convergence"
time = range(nsamples)

figure, axis = pyplot.subplots(figsize=(12, 6))
axis.set_xlabel("Time")
axis.set_ylabel("σ")
axis.set_title(title)
axis.set_xlim([1.0, nsamples])
axis.set_ylim([0.0, 0.6])
axis.semilogx(time, numpy.full(nsamples, σ), label="Target σ", color="#000000")
axis.semilogx(time, stats.cumsigma(samples), label=r"Sampled Distribution")
axis.legend()
