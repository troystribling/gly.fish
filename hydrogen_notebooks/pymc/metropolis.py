# %%
import numpy
from matplotlib import pyplot
from scipy import stats

%matplotlib inline

# %%


def parabola(x):
    if x < 0.0 or x > 1.0:
        return None
    return 3.0 * numpy.power(x, 2)


def linear(x):
    if x < 0.0 or x > 1.0:
        return None
    return 2.0 * x


def gauss(x):
    return stats.norm.pdf(x, 2.0, 2.0)


def exp(x):
    return stats.expon.pdf(x, 0.0, 2.0)


def cauchy(x):
    return stats.cauchy.pdf(x)


def pareto(x):
    if x < 1.0:
        return None
    return stats.pareto.pdf(x, 1.0)


def metropolis(p, nsample=10000, x0=0.0):
    x = x0
    i = 0
    samples = numpy.zeros(nsample)

    while i < nsample:
        x_star = x + numpy.random.normal()
        reject = numpy.random.rand()
        px_star = p(x_star)
        px = p(x)
        if px is not None and px_star is not None:
            if reject < px_star / px:
                x = x_star
        samples[i] = x
        i += 1

    return samples


def sample_plot(samples, sampled_function):
    figure, axis = pyplot.subplots(figsize=(12, 5))
    axis.set_xlabel("Sample")
    axis.set_ylabel("Value")
    axis.set_title("Metropolis Sampling")
    axis.grid(True, zorder=5)
    _, bins, _ = axis.hist(samples, 50, density=True, color="#348ABD", alpha=0.6, label=f"Sampled Distribution", edgecolor="#348ABD", lw="3", zorder=10)
    sample_values = [sampled_function(val) for val in bins]
    axis.plot(bins, sample_values, color="#A60628", label=f"Sampled Function", lw="3", zorder=10)
    axis.legend()


# %%

samples = metropolis(gauss, nsample=10000)
sample_plot(samples, gauss)

# %%

samples = metropolis(exp, nsample=10000)
sample_plot(samples, exp)


# %%

samples = metropolis(cauchy, nsample=10000)
sample_plot(samples, cauchy)


# %%

samples = metropolis(pareto, nsample=10000, x0=10.0)
sample_plot(samples, pareto)

# %%

samples = metropolis(linear, nsample=100000, x0=0.5)
sample_plot(samples, linear)

# %%

samples = metropolis(parabola, nsample=100000, x0=0.5)
sample_plot(samples, parabola)
