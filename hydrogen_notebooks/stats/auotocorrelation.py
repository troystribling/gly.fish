# %%
%load_ext autoreload
%autoreload 2

import numpy
from matplotlib import pyplot
from glyfish import config
from scipy import fftpack

pyplot.style.use(config.glyfish_style)

# %%

def autocorrelate(x):
    n = len(x)
    x_shifted = x - x.mean()
    x_padded = numpy.concatenate((x_shifted, numpy.zeros(n-1)))
    x_fft = numpy.fft.fft(x_padded)
    h_fft = numpy.conj(x_fft) * x_fft
    ac = numpy.fft.ifft(h_fft)
    return ac[0:n]/ac[0]

def autocorrelate_sum(x):
    n = len(x)
    x_shifted = x - x.mean()
    ac = numpy.zeros(n)
    for t in range(n):
        for k in range(0, n - t):
            ac[t] += x[k] * x[k + t]
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

autocorrelate(f)
autocorrelate_sum(f)

# %%
#  Example AR(1) Time Series
σ = 1.0
x0 = 1.0
αs = [0.1, 0.6, 0.9]
nsamples = 1000

figure, axis = pyplot.subplots(3, sharex=True, figsize=(12, 9))
axis[0].set_title(f"AR(1) Time Series")
axis[2].set_xlabel("Time")

for i in range(0, len(αs)):
    α = αs[i]
    samples = ar_1_series(α, σ, x0, nsamples)
    axis[i].set_xlim([0, nsamples])
    axis[i].set_ylim([-7.0, 7.0])
    axis[i].text(50, 5.2, f"α={α}", fontsize=14)
    axis[i].plot(range(0, len(samples)), samples, lw="1")

# %%
# compute autocorrlation

σ = 1.0
x0 = 1.0
αs = [0.1, 0.6, 0.9]
nsamples = 10000
nplot = 50

figure, axis = pyplot.subplots(figsize=(12, 9))
axis.set_title(f"AR(1) Time Series Autocorrelation")
axis.set_xlabel("Time")
axis.set_xlim([0, nplot])

for i in range(0, len(αs)):
    α = αs[i]
    samples = ar_1_series(α, σ, x0, nsamples)
    ac = autocorrelate(samples)
    axis.plot(range(nplot), numpy.real(ac[:nplot]), label=f"α={α}")
axis.legend()


# %%
# compare autocorrelation to equilibrium value
σ = 1.0
x0 = 1.0
αs = [0.1, 0.6, 0.9]
nsamples = 10000
nplot = 50

figure, axis = pyplot.subplots(3, sharex=True, figsize=(12, 9))
axis[0].set_title(f"AR(1) Time Series")
axis[2].set_xlabel("Time")

for i in range(0, len(αs)):
    α = αs[i]
    samples = ar_1_series(α, σ, x0, nsamples)
    ac = autocorrelate(samples)
    ac_eq = [α**n for n in range(nplot)]
    axis[i].set_xlim([0, nplot])
    axis[i].set_ylim([-0.1, 1.1])
    axis[i].plot(range(nplot), numpy.real(ac[:nplot]), marker='o', markersize=10.0, linestyle="None", markeredgewidth=1.0, alpha=0.5, label="samples")
    axis[i].plot(range(nplot), ac_eq, lw="2", label=r"$γ_E$")
    axis[i].legend()
