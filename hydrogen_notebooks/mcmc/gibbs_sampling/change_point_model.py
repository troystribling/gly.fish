# %%

%load_ext autoreload
%autoreload 2

import numpy

from matplotlib import pyplot
import scipy
from glyfish import config
from glyfish import stats

%matplotlib inline

pyplot.style.use(config.glyfish_style)

# %%

def generate_counts_time_series(ncounts, α, β):
    n = scipy.stats.randint.rvs(0, ncounts)
    λ1 = scipy.stats.gamma.rvs(α, scale=1.0/β)
    λ2 = scipy.stats.gamma.rvs(α, scale=1.0/β)
    return n, λ1, λ2, generate_counts_time_series_from_params(ncounts, λ1, λ2, n)

def generate_counts_time_series_from_params(ncounts, λ1, λ2, n):
    counts = numpy.zeros(ncounts)
    for i in range(ncounts):
        λ = λ1 if i < n+1 else λ2
        counts[i] = scipy.stats.poisson.rvs(λ)
    return counts

def change_point_df_cdf(counts, λ1, λ2):
    ncounts = len(counts)
    df = numpy.zeros(ncounts)
    for n in range(ncounts):
        counts_sum_lower = numpy.sum(counts[:n+1])
        counts_sum_upper = numpy.sum(counts[n+1:])
        df[n] = numpy.log(λ1)*counts_sum_lower+numpy.log(λ2)*counts_sum_upper + n*(λ2-λ1) - ncounts*λ2
    df = df - numpy.max(df)
    df = numpy.exp(df)
    df = df / numpy.sum(df)
    cdf = numpy.cumsum(df)
    return df, cdf

def change_point_inverse_cdf_sample(counts, λ1, λ2):
    ndf, ncdf = change_point_df_cdf(counts, λ1, λ2)
    cdf_value = numpy.random.rand()
    return numpy.flatnonzero(ncdf >= cdf_value)[0]

def change_point_multinomial_sample(counts, λ1, λ2):
    ndf, ncdf = change_point_df_cdf(counts, λ1, λ2)
    return numpy.where(numpy.random.multinomial(1, ndf, size=1)==1)[1][0]

def lower_λ_pdf(λ, counts, n, α, β):
    α1 = numpy.sum(counts[:n+1]) + α
    β1 = n + β
    return scipy.stats.gamma.pdf(λ, α1, scale=1.0/β1)

def lower_λ_mean(counts, n, α, β):
    α1 = numpy.sum(counts[:n+1]) + α
    β1 = n + β
    return scipy.stats.gamma.mean(α1, scale=1.0/β1)

def lower_λ_std(counts, n, α, β):
    α1 = numpy.sum(counts[:n+1]) + α
    β1 = n + β
    return scipy.stats.gamma.std(α1, scale=1.0/β1)

def lower_λ_sample(counts, n, α, β):
    α1 = numpy.sum(counts[:n+1]) + α
    β1 = n + β
    return scipy.stats.gamma.rvs(α1, scale=1.0/β1)

def upper_λ_pdf(λ, counts, n, α, β):
    ncount = len(counts)
    α2 = numpy.sum(counts[n+1:]) + α
    β2 = ncount - n + β
    return scipy.stats.gamma.pdf(λ, α2, scale=1.0/β2)

def upper_λ_mean(counts, n, α, β):
    ncount = len(counts)
    α2 = numpy.sum(counts[n+1:]) + α
    β2 = ncount - n + β
    return scipy.stats.gamma.mean(α2, scale=1.0/β2)

def upper_λ_std(counts, n, α, β):
    ncount = len(counts)
    α2 = numpy.sum(counts[n+1:]) + α
    β2 = ncount - n + β
    return scipy.stats.gamma.std(α2, scale=1.0/β2)

def upper_λ_sample(counts, n, α, β):
    ncount = len(counts)
    α2 = numpy.sum(counts[n+1:]) + α
    β2 = ncount - n + β
    return scipy.stats.gamma.rvs(α2, scale=1.0/β2)

def mean(x, p, dx):
    return dx*numpy.sum(p*x[1:])

def std(x, p, dx):
    return numpy.sqrt(dx*numpy.sum(p*x[1:]**2) - mean(x, p, dx)**2)

def gibbs_sample(counts, n0, λ10, λ20, α, β, nsample):
    ncount = len(counts)
    n = numpy.zeros(nsample, dtype=int)
    λ1 = numpy.zeros(nsample)
    λ2 = numpy.zeros(nsample)
    n[0] = n0
    λ1[0] = λ10
    λ2[0] = λ20
    for i in range(1, nsample):
        λ1[i] = lower_λ_sample(counts, n[i-1], α, β)
        λ2[i] = upper_λ_sample(counts, n[i-1], α, β)
        n[i] = change_point_inverse_cdf_sample(counts, λ1[i], λ2[i])
    return n, λ1, λ2

# %%
# Global Parameters

ncounts = 101
α = 2
β = 1

# %%
## gamma distribution

λ = numpy.linspace(0.001, 10.0, 200)

figure, axis = pyplot.subplots(figsize=(10, 7))
axis.set_xlabel("λ")
axis.set_ylabel("PDF")
axis.set_title(f"Gamma Distribution α={α}, β={β}")
axis.set_ylim([0.0, 0.4])
axis.set_xlim([0.0, 10.0])
pdf = stats.gamma(α, 1.0/β)
axis.plot(λ, [pdf(x) for x in λ])
config.save_post_asset(figure, "gibbs_sampling", "change_point_count_mean_distribution")

# %%
# Change point probability

nplot = 5

n = 50
λ1 = [1.0, 1.0, 1.0, 1.0, 1.0]
λ2 = [2.0, 3.0, 4.0, 3.0, 4.0]

figure, axis = pyplot.subplots(figsize=(10, 7))
axis.set_xlabel("n")
axis.set_xlim([35, 65])
axis.set_ylabel("Probabilty")
axis.set_title(r"Change Point Model, $n_{cp}=$" + f"{n}")

for i in range(nplot):
    counts = generate_counts_time_series_from_params(ncounts, λ1[i], λ2[i], n)
    ndf, ncdf = change_point_df_cdf(counts, λ1[i], λ2[i])
    label = r"$λ_1=$"+f"{format(λ1[i], '1.0f')}" + r"$,  λ_2=$"+f"{format(λ2[i], '1.0f')}"
    axis.plot(range(ncounts), ndf, label=label)

axis.set_ylim([-0.1, 1.0])
axis.legend()
config.save_post_asset(figure, "gibbs_sampling", "change_point_distribution_comparison_1")

# %%
# Change point probability

nplot = 3

n = 50
λ1 = [1.0, 1.5, 1.5]
λ2 = [1.0, 1.0, 1.0]

figure, axis = pyplot.subplots(figsize=(10, 7))
axis.set_xlabel("n")
axis.set_xlim([0, ncounts-1])
axis.set_ylabel("Probabilty")
axis.set_title(r"Change Point Model, $n_{cp}=$" + f"{n}")

for i in range(nplot):
    counts = generate_counts_time_series_from_params(ncounts, λ1[i], λ2[i], n)
    ndf, ncdf = change_point_df_cdf(counts, λ1[i], λ2[i])
    label = r"$λ_1=$"+f"{format(λ1[i], '1.1f')}" + r"$, λ_2=$"+f"{format(λ2[i], '1.1f')}"
    axis.plot(range(ncounts), ndf, label=label)

axis.set_ylim([-0.01, 0.1])
axis.legend()
config.save_post_asset(figure, "gibbs_sampling", "change_point_distribution_comparison_2")

# %%
# Change point probaility simulation parameters

nsample = 10000
n = 50
λ1 = 1.0
λ2 = 3.0
counts = generate_counts_time_series_from_params(ncounts, λ1, λ2, n)
ndf, ncdf = change_point_df_cdf(counts, λ1, λ2)

# %%

title = f"Change Point Model"+r", $λ_1=$"+f"{format(λ1, '2.0f')}"+r", $λ_2=$"+f"{format(λ2, '2.0f')}, "+ r"$n_{cp}=$"+f"{n}"

figure, axis = pyplot.subplots(figsize=(10, 7))
axis.set_xlabel("n")
axis.set_xlim([0, ncounts-1])
axis.set_ylim([-0.1, 1.1])
axis.set_title(title)
axis.plot(range(ncounts), ndf, label="Distribution")
axis.plot(range(ncounts), ncdf, label="CDF")
axis.legend(bbox_to_anchor=(0.35, 0.85))
config.save_post_asset(figure, "gibbs_sampling", "change_point_distribution_solution_1")

# %%

samples = [change_point_inverse_cdf_sample(counts, λ1, λ2) for _ in range(nsample)]
title = f"Inverse CDF Sampled Change Point"+r", $λ_1=$"+f"{format(λ1, '2.0f')}"+r", $λ_2=$"+f"{format(λ2, '2.0f')}, "+ r"$n_{cp}=$"+f"{n}"

figure, axis = pyplot.subplots(figsize=(10, 7))
axis.set_xlabel("n")
axis.set_xlim([0, ncounts-1])
axis.set_ylabel("Probability")
axis.set_title(title)
axis.set_prop_cycle(config.distribution_sample_cycler)

bins = numpy.linspace(-0.5, 100.5, ncounts)
hist, _ = numpy.histogram(samples, bins)
p = hist/numpy.sum(hist)

axis.bar(range(ncounts-1), p, label=f"Samples", zorder=5, width=0.75)
axis.plot(range(ncounts), ndf, label="Distribution", zorder=6, color="#320075")

bbox = dict(boxstyle='square,pad=1', facecolor="#FFFFFF", edgecolor="white")
axis.text(15, 0.2, f"mode={numpy.argmax(p)}", fontsize=14, bbox=bbox)
axis.legend(bbox_to_anchor=(0.9, 0.8))
config.save_post_asset(figure, "gibbs_sampling", "change_point_distribution_historgram_1")

# %%

nbins = 50
nx = 200
μ = lower_λ_mean(counts, n, α, β)
σ = lower_λ_std(counts, n, α, β)
xlim = [μ - 5.0*σ, μ + 5.0*σ]
x = numpy.linspace(xlim[0], xlim[1], nx)
bins = numpy.linspace(xlim[0], xlim[1], nbins)
title = r"$λ_1$ Distribution, $λ_1=$"+f"{format(λ1, '2.0f')}"+r", $λ_2=$"+f"{format(λ2, '2.0f')}, n={n}"

samples = [lower_λ_sample(counts, n, α, β) for _ in range(nsample)]

figure, axis = pyplot.subplots(figsize=(10, 7))
axis.set_xlabel(r"$λ_1$")
axis.set_ylabel("PDF")
axis.set_xlim(xlim)
axis.set_title(title)
axis.set_prop_cycle(config.distribution_sample_cycler)
pdf, _, _ = axis.hist(samples, bins, density=True, rwidth=0.8, label=f"Samples", zorder=5)
axis.plot(x, lower_λ_pdf(x, counts, n, α, β), label=f"Sampled Density", zorder=6)

ylim = axis.get_ylim()
xbox = xlim[0]+0.075*(xlim[1]-xlim[0])
ybox = ylim[0]+ 0.75*(ylim[1]-ylim[0])
bbox = dict(boxstyle='square,pad=1', facecolor="#FFFFFF", edgecolor="white")
axis.text(xbox, ybox, f"mode={format(bins[numpy.argmax(pdf)], '2.2f')}", fontsize=14, bbox=bbox)
axis.legend(bbox_to_anchor=(1.0, 0.8))
config.save_post_asset(figure, "gibbs_sampling", "change_point_count_lambda_1_distribution_historgram_1")

# %%

nbins = 50
nx = 200
μ = upper_λ_mean(counts, n, α, β)
σ = upper_λ_std(counts, n, α, β)
xlim = [μ - 5.0*σ, μ + 5.0*σ]
x = numpy.linspace(xlim[0], xlim[1], nx)
bins = numpy.linspace(xlim[0], xlim[1], nbins)

samples = [upper_λ_sample(counts, n, α, β) for _ in range(nsample)]
title = r"$λ_2$ Distribution, $λ_2=$"+f"{format(λ1, '2.0f')}"+r", $λ_2=$"+f"{format(λ2, '2.0f')}, n={n}"

figure, axis = pyplot.subplots(figsize=(10, 7))
axis.set_xlabel(r"$λ_2$")
axis.set_ylabel("PDF")
axis.set_xlim(xlim)
axis.set_title(title)
axis.set_prop_cycle(config.distribution_sample_cycler)
pdf, _, _ = axis.hist(samples, bins, density=True, rwidth=0.8, label=f"Samples", zorder=5)
axis.plot(x, upper_λ_pdf(x, counts, n, α, β), label=f"Sampled Density", zorder=6)

ylim = axis.get_ylim()
xbox = xlim[0] + 0.075*(xlim[1]-xlim[0])
ybox = ylim[0] + 0.75*(ylim[1]-ylim[0])
bbox = dict(boxstyle='square,pad=1', facecolor="#FFFFFF", edgecolor="white")
axis.text(xbox, ybox, f"mode={format(bins[numpy.argmax(pdf)], '2.2f')}", fontsize=14, bbox=bbox)
axis.legend(bbox_to_anchor=(1.0, 0.8))
config.save_post_asset(figure, "gibbs_sampling", "change_point_count_lambda_2_distribution_historgram_1")

# %%
# Generate Simulation Parameters

n, λ1, λ2, counts = generate_counts_time_series(ncounts, α, β)

# %%

figure, axis = pyplot.subplots(figsize=(10, 7))
title = f"Change Point Counts"+r", $λ_1=$"+f"{format(λ1, '2.2f')}"+r", $λ_2=$"+f"{format(λ2, '2.2f')}, n={n}"
axis.set_xlabel("Time")
axis.set_ylabel("Count")
axis.set_title(title)
axis.set_xlim([0, len(counts)])
axis.bar(numpy.arange(len(counts)), counts, zorder=6)
config.save_post_asset(figure, "gibbs_sampling", "change_point_counts_1")

# %%

x = numpy.arange(7)
figure, axis = pyplot.subplots(figsize=(10, 7))
axis.bar(x - 0.2, scipy.stats.poisson.pmf(x, λ1), 0.4, label=f"λ = {format(λ1, '2.2f')}", zorder=5)
axis.bar(x + 0.2, scipy.stats.poisson.pmf(x, λ2), 0.4, label=f"λ = {format(λ2, '2.2f')}", zorder=5)
axis.set_xlabel("Count")
axis.set_xticks(x)
axis.set_ylabel("Probability")
axis.set_title(f"Change Point Count Distribution")
axis.legend(bbox_to_anchor=(0.9, 0.85))
config.save_post_asset(figure, "gibbs_sampling", "change_point_change_count_distribution_1")

# %%
# Gibbs Sampling

nsample = 20000
n0 = 50
λ10 = 1.0
λ20 = 2.0
n_samples, λ1_samples, λ2_samples = gibbs_sample(counts, n0, λ10, λ20, α, β, nsample)

# %%

ndf, ncdf = change_point_df_cdf(counts, λ1, λ2)
title = f"Change Point Distribution"+r", $λ_1=$"+f"{format(λ1, '2.2f')}"+r", $λ_2=$"+f"{format(λ2, '2.2f')}, n={n}"

figure, axis = pyplot.subplots(figsize=(10, 7))
axis.set_xlabel("n")
axis.set_xlim([0, ncounts-1])
axis.set_ylabel("Probability")
axis.set_title(title)
bins = numpy.linspace(-0.5, 100.5, ncounts)
hist, _ = numpy.histogram(n_samples, bins)
p = hist/numpy.sum(hist)
axis.set_prop_cycle(config.distribution_sample_cycler)
axis.bar(range(ncounts-1), p, label=f"Samples", zorder=5, width=0.75)
axis.plot(range(ncounts), ndf, label="Distribution", zorder=6, color="#320075")
axis.legend(bbox_to_anchor=(0.9, 0.85))
config.save_post_asset(figure, "gibbs_sampling", "change_point_distribution_3")

# %%

nbins = 50
nx = 200
μ = lower_λ_mean(counts, n, α, β)
σ = lower_λ_std(counts, n, α, β)
xlim = [μ - 5.0*σ, μ + 5.0*σ]
x = numpy.linspace(xlim[0], xlim[1], nx)
bins = numpy.linspace(xlim[0], xlim[1], nbins)
title = r"$λ_1$ Distribution, $λ_1=$"+f"{format(λ1, '2.2f')}"+r", $λ_2=$"+f"{format(λ2, '2.2f')}, n={n}"

figure, axis = pyplot.subplots(figsize=(10, 7))
axis.set_xlabel(r"$λ_1$", fontsize=15)
axis.set_ylabel("PDF")
axis.set_prop_cycle(config.distribution_sample_cycler)
axis.set_title(title)
axis.set_xlim(xlim)
pdf, _, _ = axis.hist(λ1_samples, bins, density=True, rwidth=0.8, label=f"Samples", zorder=5)
axis.plot(x, lower_λ_pdf(x, counts, n, α, β), label=f"PDF", zorder=6)

dx = (xlim[1] - xlim[0]) / (nbins - 1)
ylim = axis.get_ylim()
xbox = xlim[0]+0.075*(xlim[1]-xlim[0])
ybox = ylim[0]+0.6*(ylim[1]-ylim[0])
stats_box = "$λ_{{mode}}={{{}}}$\n$λ_μ={{{}}}$\n$λ_σ={{{}}}$".format(format(bins[numpy.argmax(pdf)], '2.2f'), format(mean(bins, pdf, dx), '2.2f'), format(std(bins, pdf, dx), '2.2f'))
bbox = dict(boxstyle='square,pad=1', facecolor="#FFFFFF", edgecolor="white")
axis.text(xbox, ybox, stats_box, fontsize=16, bbox=bbox)

axis.legend(bbox_to_anchor=(0.9, 0.8))
config.save_post_asset(figure, "gibbs_sampling", "change_point_count_lambda_1_distribution_historgram_2")

# %%

nbins = 50
nx = 200
μ = upper_λ_mean(counts, n, α, β)
σ = upper_λ_std(counts, n, α, β)
xlim = [μ - 5.0*σ, μ + 5.0*σ]
x = numpy.linspace(xlim[0], xlim[1], 200)
bins = numpy.linspace(xlim[0], xlim[1], 50)
title = r"$λ_2$ Distribution, $λ_1=$"+f"{format(λ1, '2.2f')}"+r", $λ_2=$"+f"{format(λ2, '2.2f')}, n={n}"

figure, axis = pyplot.subplots(figsize=(10, 7))
axis.set_xlabel(r"$λ_2$")
axis.set_ylabel("PDF")
axis.set_title(title)
axis.set_prop_cycle(config.distribution_sample_cycler)
pdf, _, _ = axis.hist(λ2_samples, bins, density=True, rwidth=0.8, label=f"Samples", zorder=5)
axis.plot(x, upper_λ_pdf(x, counts, n, α, β), label=f"PDF", zorder=6)

dx = (xlim[1] - xlim[0]) / (nbins - 1)
ylim = axis.get_ylim()
xbox = xlim[0]+0.075*(xlim[1]-xlim[0])
ybox = ylim[0]+0.6*(ylim[1]-ylim[0])

stats_box = "$λ_{{mode}}={{{}}}$\n$λ_μ={{{}}}$\n$λ_σ={{{}}}$".format(format(bins[numpy.argmax(pdf)], '2.2f'), format(mean(bins, pdf, dx), '2.2f'), format(std(bins, pdf, dx), '2.2f'))
axis.text(xbox, ybox, stats_box, fontsize=16, bbox=bbox)

axis.legend(bbox_to_anchor=(0.9, 0.8))
config.save_post_asset(figure, "gibbs_sampling", "change_point_count_lambda_2_distribution_historgram_2")
