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

def change_point_df_cdf(counts, λ1, λ2):
    ncounts = len(counts)
    df = numpy.zeros(ncounts)
    for n in range(ncounts):
        counts_sum_lower = numpy.sum(counts[:n])
        counts_sum_upper = numpy.sum(counts[n:])
        df[n] = numpy.log(λ1)*counts_sum_lower+numpy.log(λ2)*counts_sum_upper + n*(λ2-λ1)

    df = df - numpy.max(df)
    df = numpy.exp(df)
    df = df / numpy.sum(df)
    cdf = numpy.cumsum(df)
    return df, cdf

def change_point_samples_example(nsamples, counts, λ1, λ2):
	for i in range(N):
		mult_n[i]=sum(counts[0:i])*log(λ1)-i*λ1+sum(counts[i:N])*log(λ2)-(N-i)*λ2
	mult_n=exp(mult_n-max(mult_n))
	return numpy.where(multinomial(1,mult_n/sum(mult_n),size=1)==1)[1][0]

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
axis.bar(x - 0.2, stats.poisson.pmf(x, λ1), 0.4, label=f"λ = {format(λ1, '2.2f')}", zorder=5)
axis.bar(x + 0.2, stats.poisson.pmf(x, λ2), 0.4, label=f"λ = {format(λ2, '2.2f')}", zorder=5)
axis.set_xlabel("Count")
axis.set_xticks(x)
axis.set_ylabel("Probability")
axis.set_title(f"Poisson Distribution")
axis.legend()


# %%

ndf, ncdf = change_point_df_cdf(counts, λ1, λ2)
