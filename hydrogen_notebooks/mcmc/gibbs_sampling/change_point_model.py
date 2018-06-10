# %%
%load_ext autoreload
%autoreload 2

import numpy

from matplotlib import pyplot
from scipy import stats
from glyfish import config

%matplotlib inline

pyplot.style.use(config.glyfish_style)

# %%

def generate_counts_time_series(ncounts, α, β):
    counts = numpy.zeros(ncounts)
    n = stats.randint.rvs(1, ncounts + 1)
    λ1 = stats.gamma.rvs(α, scale=1.0/β)
    λ2 = stats.gamma.rvs(α, scale=1.0/β)
    for i in range(ncounts):
        λ = λ1 if i < n + 1 else λ2
        counts[i] = stats.poisson.rvs(λ)

    return n, λ1, λ2, counts

# %%

n, λ1, λ2, counts = generate_counts_time_series(100, 2, 1)

# %%

figure, axis = pyplot.subplots(figsize=(12, 5))
title = f"Change Point Model"+r", $λ_1=$"+f"{format(λ1, '2.2f')}"+r", $λ_2=$"+f"{format(λ2, '2.2f')}, n={n}"
axis.set_xlabel("Time")
axis.set_ylabel("Count")
axis.set_title(title)
axis.set_xlim([0, len(counts)])
axis.bar(numpy.arange(len(counts)), counts, zorder=6)

# %%

x = numpy.arange(10)
figure, axis = pyplot.subplots(figsize=(12, 5))
axis.bar(x, stats.poisson.pmf(x, λ1), label=f"λ = {format(λ1, '2.2f')}", alpha=0.6, zorder=10)
axis.bar(x, stats.poisson.pmf(x, λ2), label=f"λ = {format(λ2, '2.2f')}", alpha=0.6, zorder=10)
axis.set_xlabel("Count")
axis.set_xticks(x)
axis.set_ylabel("Probability of $k$")
axis.set_title(f"Poisson Distribution")
axis.legend()
