# %%
%load_ext autoreload
%autoreload 2

%aimport numpy
%aimport sympy

from matplotlib import pyplot
from scipy import stats

%matplotlib inline

# %%
# Inverse CDF Descrete random variables


x = sympy.symbols('x')
sympy.Heaviside(0, 1)
steps = [sympy.Heaviside(x - i + 1, 1) for i in range(1, 7)]

steps[0]
steps[0].subs(x, 2)

cdf = sum(steps) / 6
cdf.subs(x, 3)

sympy.plot(cdf, (x, 0, 6), ylabel="CDF(x)", xlabel='x', ylim=(0, 1))

# %%
# The inverse of the heavyside distribution is given by
x = sympy.symbols('x')
intervals = [(1, sympy.Interval(0, 1 / 6, False, True).contains(x)), (2, sympy.Interval(1 / 6, 2 / 6, False, True).contains(x)), (3, sympy.Interval(2 / 6, 3 / 6, False, True).contains(x)), (4, sympy.Interval(3 / 6, 4 / 6, False, True).contains(x)), (5, sympy.Interval(4 / 6, 5 / 6, False, True).contains(x)), (6, sympy.Interval(5 / 6, 1, False, False).contains(x))]
inv_cdf = sympy.Piecewise(*intervals)
samples = [int(inv_cdf.subs(x, i)) for i in numpy.random.rand(10000)]
n, bins, _ = pyplot.hist(samples, bins=[1, 2, 3, 4, 5, 6, 7], density=True, align='left', rwidth=0.9)
pyplot.title("Simulated PDF")

# %%
sampled_cdf = numpy.cumsum(n)
cdf_values = [float(cdf.subs(x, i)) for i in range(0, 6)]

figure, axis = pyplot.subplots(figsize=(12, 5))
axis.set_xlabel("Sample")
axis.set_ylabel("Value")
axis.set_title("Simulated CDF")
axis.grid(True, zorder=5)
random_variable_values = [i + 0.2 for i in range(1, 7)]
axis.bar(random_variable_values, sampled_cdf, 0.4, color="#A60628", label=f"Sampled CDF Estimate", alpha=0.6, lw="3", edgecolor="#A60628", zorder=10)
random_variable_values = [i - 0.2 for i in range(1, 7)]
axis.bar(random_variable_values, cdf_values, 0.4, color="#348ABD", label=f"CDF", alpha=0.6, lw="3", edgecolor="#348ABD", zorder=10)
axis.legend()

# %%
# For the continuous distribution exp(x)

f_inv = lambda v: numpy.log(1.0 / (1.0 - v))
samples = [f_inv(v) for v in numpy.random.rand(10000)]

figure, axis = pyplot.subplots(figsize=(12, 5))
axis.set_xlabel("Sample")
axis.set_ylabel("PDF")
axis.set_title("Inverse Sampling")
axis.grid(True, zorder=5)
_, bins, _ = axis.hist(samples, 50, density=True, color="#348ABD", alpha=0.6, label=f"Sampled Density", edgecolor="#348ABD", lw="3", zorder=10)
sample_values = [numpy.exp(-v) for v in bins]
axis.plot(bins, sample_values, color="#A60628", label=f"Sampled Function", lw="3", zorder=10)
axis.legend()


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
_, x_values, _ = axis.hist(accepted_samples, 50, density=True, color="#348ABD", alpha=0.6, label=f"Sampled Density", edgecolor="#348ABD", lw="3", zorder=10)
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
axis.plot(x_values, f(x_values) / M, color="#A60628", lw="3", zorder=10)
axis.scatter(x_samples[accepted_mask], y_samples[accepted_mask], label="Accepted Samples", color="#348ABD", alpha=0.5)
axis.scatter(x_samples[rejected_mask], y_samples[rejected_mask], label="Rejected Samples", color="#0BAA54", alpha=0.5)
axis.legend()

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
