# %%
%load_ext autoreload
%autoreload 2

import numpy
import scipy

from matplotlib import pyplot
from glyfish import config

%matplotlib inline

pyplot.style.use(config.glyfish_style)

# %%

def ar_1_series(α, σ, x0, nsamples=100):
    samples = numpy.zeros(nsamples)
    ε = numpy.random.normal(0.0, σ, nsamples)
    samples[0] = x0
    for i in range(1, nsamples):
        samples[i] = α * samples[i-1] + ε[i]
    return samples

def ar_1_kernel(α, σ, x, y):
    p = numpy.zeros(len(y))
    for i in range(0, len(y)):
        ε  = ((y[i] -  α * x)**2) / (2.0 * σ**2)
        p[i] = numpy.exp(-ε) / numpy.sqrt(2 * numpy.pi * σ**2)
    return p


def ar_1_equilibrium_distributions(α, σ, x0, y, nsample=100):
    py = [ar_1_kernel(α, σ, x, y) for x in ar_1_series(α, σ, x0, nsample)]
    pavg = []
    for i in range(0, len(py)):
        pavg_next = py[i] if i == 0 else (py[i] + i * pavg[i-1]) / (i + 1)
        pavg.append(pavg_next)
    return pavg

def alpha_steps(nplots):
    alpha_min = 0.5
    alpha_max = 0.8
    dalpha = (alpha_max - alpha_min) / (nplots - 1)
    return [alpha_min + dalpha * i for i in range(0, nplots)]


def y_steps(α, σ, npts):
    γ = equilibrium_standard_deviation(α, σ)
    ymax = 5.0 * γ
    dy = 2.0 * ymax / (npts - 1)
    return [-ymax + dy * i for i in range(0, npts)]


def equilibrium_standard_deviation(α, σ):
    return numpy.sqrt(σ**2/(1.0 - α**2))


def cummean(α, σ, x0, nsample=100):
    samples = ar_1_series(α, σ, x0, nsample)
    mean = numpy.zeros(nsample)
    mean[0] = samples[0]
    for i in range(1, len(samples)):
        mean[i] = (float(i) * mean[i - 1] + samples[i])/float(i + 1)
    return mean

def cumsigma(α, σ, x0, nsample=100):
    samples = ar_1_series(α, σ, 5.0, nsample)
    var = numpy.zeros(nsample)
    var[0] = samples[0]**2
    for i in range(1, len(samples)):
        var[i] = (float(i) * var[i - 1] + samples[i]**2)/float(i + 1)
    return numpy.sqrt(var)

def ar_1_equilibrium_distribution(α, σ, y):
    σ = equilibrium_standard_deviation(α, σ)
    p = numpy.zeros(len(y))
    for i in range(0, len(y)):
        ε  = y[i]**2 / ( 2.0 * σ**2)
        p[i] = numpy.exp(-ε) / numpy.sqrt(2 * numpy.pi * σ**2)
    return  p

# %%

σ = 1.0
x0 = 1.0
αs = [0.1, 0.6, 0.9]

figure, axis = pyplot.subplots(3, sharex=True, figsize=(10, 7))
axis[0].set_title(f"AR(1) Time Series")
axis[2].set_xlabel("Time")

for i in range(0, len(αs)):
    α = αs[i]
    samples = ar_1_series(α, σ, x0, 1000)
    axis[i].set_xlim([0, 1000])
    axis[i].set_ylim([-7.0, 7.0])
    axis[i].text(50, 5.2, f"α={α}", fontsize=16)
    axis[i].plot(range(0, len(samples)), samples, lw="1")
config.save_post_asset(figure, "continuous_state_markov_chain_equilibrium", "ar1_alpha_sample_comparison")

# %%

σ = 10.0
x0 = 1.0
α = 1.002

samples = ar_1_series(α, σ, x0, 1000)

figure, axis = pyplot.subplots(figsize=(10, 6))
axis.set_xlabel("Time")
axis.set_xlim([0, 1000])
axis.set_title("AR(1) Time Series")
axis.plot(range(0, len(samples)), samples)
axis.text(50, 500, f"α={α}", fontsize=16)
config.save_post_asset(figure, "continuous_state_markov_chain_equilibrium", "ar1_alpha_larger_than_1")


# %%

σ = 1.0
x0 = 5.0
αs = [0.1, 0.6, 0.9]
nsample = 10000
time = range(nsample)

figure, axis = pyplot.subplots(figsize=(10, 6))
axis.set_xlabel("Time")
axis.set_ylabel(r"$μ$")
axis.set_title(r"AR(1) Convergence to $μ_E$")
axis.set_xlim([1.0, nsample])

for i in range(0, len(αs)):
    α = αs[i]
    mean = cummean(α, σ, x0, nsample)
    axis.semilogx(time, mean, label=f"α={α}")

axis.semilogx(time, numpy.full((len(time)), 0.0), label=r"$μ_E$")

axis.set_prop_cycle(None)

for i in range(0, len(αs)):
    α = αs[i]
    mean = cummean(α, σ, -x0, nsample)
    axis.semilogx(time, mean)

axis.legend(bbox_to_anchor=(0.95, 0.95), fontsize=16)
config.save_post_asset(figure, "continuous_state_markov_chain_equilibrium", "mean_convergence")


# %%

σ = 1.0
x0 = 5.0
αs = [0.1, 0.6, 0.9]
nsample = 10000

figure, axis = pyplot.subplots(figsize=(10, 6))
axis.set_xlabel("Time")
axis.set_ylabel(r"$σ$")
axis.set_title(r"AR(1) Convergence $σ_E$")
axis.set_xlim([1.0, nsample])
axis.set_ylim([0.0, 6.0])
axis.set_yticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])

for i in range(len(αs)):
    α = αs[i]
    sigma = cumsigma(α, σ, x0, nsample)
    time = range(len(sigma))
    axis.semilogx(time, sigma, label=f"α={α}")

axis.set_prop_cycle(None)

for i in range(0, len(αs)):
    α = αs[i]
    sigma = cumsigma(α, σ, x0, nsample)
    time = range(len(sigma))
    γ = equilibrium_standard_deviation(α, σ)
    axis.semilogx(time, numpy.full((nsample), γ))

axis.legend(bbox_to_anchor=(0.95, 0.95), fontsize=16)
config.save_post_asset(figure, "continuous_state_markov_chain_equilibrium", "sigma_convergence")

# %%

σ = 1.0
α = 0.5
nsamples = 500
x0 = 5.0

steps = [[0, 1, 2, 3, 5], [10, 15, 20, 25, 30], [40, 50, 60, 70, 80], [100, 200, 300, 400]]
colors = config.bar_plot_colors
alpha = alpha_steps(len(colors))
y = y_steps(α, σ, 200)

kernel_mean = ar_1_equilibrium_distributions(α, σ, x0, y, nsamples)
title = r"AR(1) Relaxation to Equilibrium: $\alpha=$"+f"{format(α, '2.2f')}" + r"$, \sigma=$"+f"{format(σ, '2.2f')}" + r"$, x_0=$"+f"{format(x0, '2.2f')}"

figure, axis = pyplot.subplots(figsize=(10, 6))
axis.set_xlabel("y")
axis.set_ylabel(r'$\pi$(y)')
axis.set_title(title)
axis.set_ylim([0, 0.45])
axis.set_xlim([-5.0, 5.0])
axis.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
axis.set_xticks([-4.0, -2, 0.0, 2.0, 4.0])

for i in range(0, len(steps)):
    sub_steps = steps[i]
    axis.plot(y, kernel_mean[sub_steps[0]], color=colors[i], lw="2", alpha=alpha[i], label=f"t={sub_steps[0]}-{sub_steps[-1]}", zorder=6)
    for j in range(1, len(sub_steps)):
        axis.plot(y, kernel_mean[sub_steps[j]], color=colors[i], lw="2", zorder=6, alpha=alpha[i])
axis.plot(y, kernel_mean[-1], color="#000000", lw="4", label=f"Equlibrium", zorder=10, alpha=alpha[i])
axis.legend(bbox_to_anchor=(0.3, 0.95))
config.save_post_asset(figure, "continuous_state_markov_chain_equilibrium", "ar1_relaxation_to_equilibrium_1")

# %%

x0 = -5.0
title = r"AR(1) Relaxation to Equilibrium: $\alpha=$"+f"{format(α, '2.2f')}" + r"$, \sigma=$"+f"{format(σ, '2.2f')}" + r"$, x_0=$"+f"{format(x0, '2.2f')}"

kernel_mean = ar_1_equilibrium_distributions(α, σ, x0, y, nsamples)

figure, axis = pyplot.subplots(figsize=(10, 6))
axis.set_xlabel("y")
axis.set_ylabel(r'$\pi$(y)')
axis.set_title(title)
axis.set_ylim([0, 0.45])
axis.set_xlim([-6.5, 6.6])
axis.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
axis.set_xticks([-6.0,-4.0, -2, 0.0, 2.0, 4.0, 6.0])

for i in range(0, len(steps)):
    sub_steps = steps[i]
    axis.plot(y, kernel_mean[sub_steps[0]], color=colors[i], lw="2", alpha=alpha[i], label=f"t={sub_steps[0]}-{sub_steps[-1]}", zorder=6)
    for j in range(1, len(sub_steps)):
        axis.plot(y, kernel_mean[sub_steps[j]], color=colors[i], lw="2", zorder=6, alpha=alpha[i])
axis.plot(y, kernel_mean[-1], color="#000000", lw="4", label=f"Equlibrium", zorder=10, alpha=alpha[i])
axis.legend(bbox_to_anchor=(0.915, 0.95), fontsize=14)
config.save_post_asset(figure, "continuous_state_markov_chain_equilibrium", "ar1_relaxation_to_equilibrium_2")

# %%

α = 0.5
nsteps = 50
kernel_mean = ar_1_equilibrium_distributions(α, σ, 5.0, y, nsteps)
π_eq = ar_1_equilibrium_distribution(α, σ, y)
title = f"Equilbrium PDF Comparison: time steps={nsteps}, " +  r"$\alpha=$"+f"{format(α, '2.2f')}" + r"$, \sigma=$"+f"{format(σ, '2.2f')}"

figure, axis = pyplot.subplots(figsize=(10, 7))
axis.set_xlabel("y", fontsize=14)
axis.set_ylabel(r'$\pi$(y)', fontsize=14)
axis.set_title(title)
axis.set_ylim([0.0, 0.4])
axis.set_xlim([-5.0, 5.0])
axis.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
axis.plot(y, π_eq, label=r"$π_E$", zorder=5)
axis.plot(y, kernel_mean[-1], label=f"Kernel Mean", zorder=5)
axis.legend(bbox_to_anchor=(0.95, 0.95))
config.save_post_asset(figure, "continuous_state_markov_chain_equilibrium", "equilibrium_pdf_comparison")

# %%

α = 0.5
nsteps = 500
nsamples = 1000000
kernel_mean = ar_1_equilibrium_distributions(α, σ, 5.0, y, nsteps)
samples = ar_1_series(α, σ, 5.0, nsamples)
title = r"Equilbrium PDF Comparison: $\alpha=$"+f"{format(α, '2.2f')}" + r"$, \sigma=$"+f"{format(σ, '2.2f')}" + r"$, x_0=$"+f"{format(x0, '2.2f')}"

figure, axis = pyplot.subplots(figsize=(10, 7))
axis.set_xlabel("y")
axis.set_ylabel(r'$\pi$(y)')
axis.set_ylim([0.0, 0.4])
axis.set_xlim([-5.0, 5.0])
axis.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
axis.set_title(title)
axis.set_prop_cycle(config.distribution_sample_cycler)
_, x_values, _ = axis.hist(samples, 50, density=True, rwidth=0.8, label=f"Sampled Density", zorder=5)
axis.plot(y, kernel_mean[-1], label=f"Kernel Mean", zorder=5)
axis.legend(bbox_to_anchor=(0.95, 0.9))
config.save_post_asset(figure, "continuous_state_markov_chain_equilibrium", "equilibrium_pdf_comparison_samples")
