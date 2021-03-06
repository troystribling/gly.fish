# %%
%load_ext autoreload
%autoreload 2

import numpy
from matplotlib import pyplot
from glyfish import config
from glyfish import stats
from scipy import fftpack

%matplotlib inline

pyplot.style.use(config.glyfish_style)

# %%

def autocorrelate_sum(x):
    n = len(x)
    x_shifted = x - x.mean()
    ac = numpy.zeros(n)
    for t in range(n):
        for k in range(0, n - t):
            ac[t] += x_shifted[k] * x_shifted[k + t]
    return ac/ac[0]

def ar_1_series(α, σ, x0=0.0, nsamples=100):
    samples = numpy.zeros(nsamples)
    ε = numpy.random.normal(0.0, σ, nsamples)
    samples[0] = x0
    for i in range(1, nsamples):
        samples[i] = α * samples[i-1] + ε[i]
    return samples

# %%
# Check autocorrleation

f = numpy.array([8, 4, 8, 0])
stats.autocorrelate(f)
autocorrelate_sum(f)

# %%
#  Example AR(1) Time Series
σ = 1.0
x0 = 1.0
αs = [0.1, 0.6, 0.9]
nsample = 1000

figure, axis = pyplot.subplots(3, sharex=True, figsize=(10, 7))
axis[0].set_title(f"AR(1) Time Series: σ={format(σ, '2.2f')}")
axis[2].set_xlabel("Time")

for i in range(0, len(αs)):
    α = αs[i]
    samples = ar_1_series(α, σ, x0, nsample)
    axis[i].set_xlim([0, 1000])
    axis[i].set_ylim([-7.0, 7.0])
    axis[i].text(50, 5.2, f"α={α}", fontsize=16)
    axis[i].plot(range(0, len(samples)), samples, lw="1")
config.save_post_asset(figure, "discrete_cross_correlation_theorem", "ar1_alpha_sample_comparison")

# %%
# compute autocorrlation

σ = 1.0
x0 = 1.0
αs = [0.1, 0.6, 0.9]
nsamples = 10000
nplot = 75

figure, axis = pyplot.subplots(figsize=(10, 7))
axis.set_title(f"AR(1) Time Series Autocorrelation: σ={format(σ, '2.2f')}")
axis.set_xlabel("Time Lag")
axis.set_xlim([0, nplot])

for i in range(0, len(αs)):
    α = αs[i]
    samples = ar_1_series(α, σ, x0, nsamples)
    ac = stats.autocorrelate(samples)
    axis.plot(range(nplot), numpy.real(ac[:nplot]), label=f"α={α}")
axis.legend(bbox_to_anchor=(0.9, 0.95), fontsize=16)
config.save_post_asset(figure, "discrete_cross_correlation_theorem", "ar1_alpha_autocorrelation_comparison")

# %%
# compute autocorrlation relaxation

σ = 1.0
x0 = 1.0
α = 0.7
nsamples = 100000
max_samples = [5000, 10000, 15000, 100000]
max_lag = 40

samples = ar_1_series(α, σ, x0, nsamples)

figure, axis = pyplot.subplots(figsize=(10, 7))
axis.set_title(f"AR(1) Time Series Autocorrelation Equilibrium Relaxation: σ={format(σ, '2.2f')}")
axis.set_xlabel("Time Lag")
axis.set_xlim([-0.5, max_lag])

for i in range(len(max_samples)):
    ac = stats.autocorrelate(samples[1:max_samples[i]])
    axis.plot(range(0, max_lag), numpy.real(ac[0:max_lag]), label=f"t={max_samples[i]}")

axis.legend(bbox_to_anchor=(0.9, 0.95), fontsize=16)

# %%
# compare autocorrelation to equilibrium value
σ = 1.0
x0 = 1.0
αs = [0.1, 0.6, 0.9]
nsamples = 10000
nplot = 50

figure, axis = pyplot.subplots(1, 3, sharey=True, figsize=(10, 7))
axis[1].set_title(f"AR(1) Time Series Autocorrelation Coefficient: σ={format(σ, '2.2f')}")

for i in range(0, len(αs)):
    α = αs[i]
    samples = ar_1_series(α, σ, x0, nsamples)
    ac = stats.autocorrelate(samples)
    ac_eq = [α**n for n in range(nplot)]
    if i==1 :
        axis[i].set_xlabel(r"Time Lag $(\tau)$")
        
    axis[i].set_xlim([-5.0, nplot])
    axis[i].set_ylim([-0.1, 1.1])
    axis[i].text(20.0, 1.0, f"α={α}", fontsize=16)
    axis[i].plot(range(nplot), numpy.real(ac[:nplot]), marker='o', markersize=10.0, linestyle="None", markeredgewidth=1.0, alpha=0.75, label="simulation", zorder=6)
    axis[i].plot(range(nplot), ac_eq, lw="2", label=r"$γ^E_{\tau}$", zorder=5)
    if i == 0:
        axis[i].legend(bbox_to_anchor=(0.2, 0.8), fontsize=16)
config.save_post_asset(figure, "discrete_cross_correlation_theorem", "ar1_alpha_equilibrium_autocorrelation_comparison")
