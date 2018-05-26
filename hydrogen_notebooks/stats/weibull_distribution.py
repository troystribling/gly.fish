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

k = 5.0
λ = 1.0

nsample = 50000
samples = numpy.random.weibull(k, nsample)

# %%

target_pdf = stats.weibull(k, λ)

x = numpy.linspace(0.001, 2.0, 500)
figure, axis = pyplot.subplots(figsize=(12, 5))
axis.set_xlabel("X")
axis.set_ylabel("PDF")
axis.set_xlim([0.0, x[-1]])
axis.set_title(f"Weibull Distribution, k={k}, λ={λ}")
axis.plot(x, [target_pdf(j) for j in x])

# %%

figure, axis = pyplot.subplots(figsize=(12, 5))
axis.set_xlabel("Sample")
axis.set_ylabel("PDF")
axis.set_title(f"Weibull Distribution, k={k}, λ={λ}")
_, bins, _ = axis.hist(samples, 50, density=True, color="#348ABD", alpha=0.6, label=f"Sampled Distribution", edgecolor="#348ABD", zorder=5)
delta = (bins[-1] - bins[0]) / 200.0
sample_distribution = [target_pdf(val) for val in numpy.arange(bins[0], bins[-1], delta)]
axis.plot(numpy.arange(bins[0], bins[-1], delta), sample_distribution, color="#A60628", label=f"Sampled Function", zorder=6)
axis.legend()


# %%

npts = 50
figure, axis = pyplot.subplots(figsize=(12, 9))
axis.set_title("Weibull Distribution")
axis.set_xlabel("Time Lag")
axis.set_xlim([0, npts])
ac = stats.autocorrelate(samples)
axis.plot(range(npts), numpy.real(ac[:npts]))
 
