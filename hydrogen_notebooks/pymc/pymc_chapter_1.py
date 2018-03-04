# %%
import pymc3 as pymc
import theano
import theano.tensor as tensor
import numpy
from matplotlib import pyplot
from scipy import stats

%matplotlib inline

# %%
# Poisson random variable
a = numpy.arange(16)
dist = stats.poisson
λ = [1.5, 4.25]
colors = ["#348ABD", "#A60628"]
figure, axis = pyplot.subplots(figsize=(12, 5))
for i in range(0, 2):
    axis.bar(a, dist.pmf(a, λ[i]), color=colors[i], label=f"λ = {λ[i]}", alpha=0.6, edgecolor=colors[i], lw="3", zorder=10)
axis.set_xlabel("$k$")
axis.set_xticks(a)
axis.set_ylabel("Probability of $k$")
axis.set_title(f"Probability mass function for Poisson randowm variable with differing λ")
axis.grid(True, zorder=5)
axis.legend()


# %%
# Exponential random variable
a = numpy.arange(0.0, 4.0, 0.1)
dist = stats.expon
λ = [0.5, 1.0]
colors = ["#348ABD", "#A60628"]
figure, axis = pyplot.subplots(figsize=(12, 5))
for i in range(0, 2):
    axis.plot(a, dist.pdf(a, 0.0, λ[i]), color=colors[i], label=f"λ = {λ[i]}", alpha=0.6, lw="3", zorder=10)
    axis.fill_between(a, dist.pdf(a, 0.0, λ[i]), numpy.zeros(len(a)), color=colors[i], alpha=0.6, zorder=10)
axis.set_xlabel("$k$")
axis.set_xlim(0.0, 4.0)
axis.set_ylim(0.0, 2.0)
axis.grid(True, zorder=5)
axis.set_ylabel("Probability of $k$")
axis.set_title(f"Probability mass function for Poisson randowm variable with differing λ")
axis.legend()

# %%
# Received text message data from chapter 1
count_data = numpy.loadtxt("hydrogen_notebooks/pymc/data/texts.csv")
figure, axis = pyplot.subplots(figsize=(12, 5))
axis.bar(numpy.arange(len(count_data)), count_data, color="#348ABD", zorder=10)
axis.set_xlabel("Time (days)")
axis.set_ylabel("Text messages received")
axis.set_title("Did the users texting habits change over time?")
axis.grid(True, zorder=5)
axis.set_xlim([0, len(count_data)])

# %%
# Assume data set can be modeled by 2 Poisson random variables with parameters λ1 and λ2 where the
# switch occurs at t = τ. Model the λ's as Exponential distributions with parameter α and τ ascii
# a uniform over the given time interval.
#
# Since λ is exponentially distributed with parameter α we can estimate α with α = λ / count_data.mean()


@theano.compile.ops.as_op(itypes=[tensor.lscalar, tensor.dscalar, tensor.dscalar], otypes=[tensor.dvector])
def mean(τ, λ1, λ2):
    out = numpy.empty(len(count_data))
    out[:τ] = λ1
    out[τ:] = λ2
    return out


def run_model(steps=10000):
    model = pymc.Model()
    with model:
        α = 1 / count_data.mean()
        λ1 = pymc.Exponential("λ1", α)
        λ2 = pymc.Exponential("λ2", α)
        τ = pymc.DiscreteUniform("τ", lower=0.0, upper=len(count_data))
        process_mean = mean(τ, λ1, λ2)
        observation = pymc.Poisson("observation", process_mean, observed=count_data)
        start = {"λ1": 10.0, "λ2": 30.0}
        step1 = pymc.Slice([λ1, λ2])
        step2 = pymc.Metropolis([τ])
        trace = pymc.sample(steps, tune=500, start=start, step=[step1, step2], cores=2)
    return pymc.trace_to_dataframe(trace)


result = run_model()
