# %%
%load_ext autoreload
%autoreload 2

import numpy
from matplotlib import pyplot
from IPython.display import Image
from glyfish import config

%matplotlib inline

pyplot.style.use(config.glyfish_style)

# %%

g1 = """digraph markov_chain {
   size="4.5,6";
   ratio=fill;
   node[fontsize=24, fontname=Helvetica];
   edge[fontsize=24, fontname=Helvetica];
   graph[fontsize=24, fontname=Helvetica];
   labelloc="t";
   label="Markov Transition Matrix";
   0 -> 1 [label=" 0.9"];
   0 -> 2 [label=" 0.1"];
   1 -> 0 [label=" 0.8"];
   1 -> 1 [label=" 0.1"];
   1 -> 3 [label=" 0.1"];
   2 -> 1 [label=" 0.5"];
   2 -> 2 [label=" 0.3"];
   2 -> 3 [label=" 0.2"];
   3 -> 0 [label=" 0.1"];
   3 -> 3 [label=" 0.9"];
}"""
config.draw(g1, 'discrete_state_markov_chain_equilibrium', 'transition_diagram')

# %%

t = [[0.0, 0.9, 0.1, 0.0],
     [0.8, 0.1, 0.0, 0.1],
     [0.0, 0.5, 0.3, 0.2],
     [0.1, 0.0, 0.0, 0.9]]
p = numpy.matrix(t)

# %%

def sample_chain(t, x0, nsample):
    xt = numpy.zeros(nsample, dtype=int)
    xt[0] = x0
    up = numpy.random.rand(nsample)
    cdf = [numpy.cumsum(t[i]) for i in range(4)]
    for t in range(nsample - 1):
        xt[t] = numpy.flatnonzero(cdf[xt[t-1]] >= up[t])[0]
    return xt

def eq_dist(π, p, nsteps):
    πt = π.T
    result = [πt]
    for i in range(0, nsteps):
        πt = πt * p
        result.append(πt)
    return result

# %%

nsamples = 10000
x0 = 1
chain_samples = sample_chain(t, x0, nsamples)

figure, axis = pyplot.subplots(figsize=(10, 6))
axis.set_xlabel("State")
axis.set_ylabel("Probability")
axis.set_title(f"Markov Chain Distribution {nsamples} Samples")
axis.set_prop_cycle(config.bar_plot_cycler)
axis.set_xlim([-0.5, 3.5])
axis.set_ylim([0.0, 0.55])
axis.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
axis.grid(True, zorder=5)
axis.set_xticks([0, 1, 2, 3])
_ = axis.hist(chain_samples - 0.5, [-0.5, 0.5, 1.5, 2.5, 3.5], rwidth=0.8, density=True, label=f"Sampled Density", zorder=5)


# %%
nsteps = 50
c = [[0.1],
     [0.5],
     [0.35],
     [0.05]]
π = numpy.matrix(c)
πt = eq_dist(π, p, nsteps)


def relaxation_plot(πt, nsteps, plot_name):
    steps = [i for i in range(0, nsteps + 1)]
    figure, axis = pyplot.subplots(figsize=(10, 6))
    axis.set_xlabel("Time")
    axis.set_ylabel("Probability")
    axis.set_title("Relaxation to Equilibrium Distribution")
    axis.set_ylim([0, 0.55])
    axis.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
    axis.set_xlim([0, nsteps])
    axis.plot(steps, [πt[i][0, 0] for i in steps], label=f"0", lw="3", zorder=10)
    axis.plot(steps, [πt[i][0, 1] for i in steps], label=f"1", lw="3", zorder=10)
    axis.plot(steps, [πt[i][0, 2] for i in steps], label=f"2", lw="3", zorder=10)
    axis.plot(steps, [πt[i][0, 3] for i in steps], label=f"3", lw="3", zorder=10)
    axis.legend(bbox_to_anchor=(0.8, 0.15))
    config.save_post_asset(figure, "discrete_state_markov_chain_equilibrium", plot_name)


relaxation_plot(πt, nsteps, "distribution_relaxation_1")

# %%

nsteps = 50
c = [[0.25],
     [0.25],
     [0.25],
     [0.25]]
π = numpy.matrix(c)
πt = eq_dist(π, p, nsteps)
relaxation_plot(πt, nsteps, "distribution_relaxation_2")


# %%

πsamples = 1000
nsamples = 10000
c = [[0.25],
     [0.25],
     [0.25],
     [0.25]]

π = numpy.matrix(c)
πcdf = numpy.cumsum(c)
π_samples = [numpy.flatnonzero(πcdf >= u)[0] for u in numpy.random.rand(πsamples)]

chain_samples = numpy.array([])
for x0 in π_samples:
    chain_samples = numpy.append(chain_samples, sample_chain(t, x0, nsamples))

# %%

figure, axis = pyplot.subplots(figsize=(10, 6))
axis.set_xlabel("State")
axis.set_ylabel("Probability")
axis.set_ylim([0.0, 0.55])
axis.set_prop_cycle(config.bar_plot_cycler)
axis.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
axis.set_title(f"Markov Chain Equilbrium PDF")
axis.set_xlim([-0.5, 3.5])
axis.set_xticks([0, 1, 2, 3])
shifted_chain_samples = chain_samples - 0.5
simpulated_pdf, _, _  = axis.hist(shifted_chain_samples, [-0.5, 0.5, 1.5, 2.5, 3.5], rwidth=0.8, density=True, label=f"Sampled Density", zorder=5)

# %%

s = numpy.concatenate((p.T - numpy.eye(4), [numpy.ones(4)]))
πe, _, _, _ = numpy.linalg.lstsq(s, numpy.array([0.0, 0.0, 0.0, 0.0, 1.0]), rcond=None)

# %%

nsteps = 50
πt = eq_dist(π, p, nsteps)
_, nπ = πt[nsteps].shape
computed_pdf = [πt[50][0, i] for i in range(0, nπ)]
states = numpy.array([0, 1, 2, 3])

figure, axis = pyplot.subplots(figsize=(10, 6))
axis.set_xlabel("State")
axis.set_ylabel("Probability")
axis.set_ylim([0.0, 0.55])
axis.set_prop_cycle(config.bar_plot_cycler)
axis.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
axis.set_title("Equilibrium Distribution Comparison")
axis.set_prop_cycle(config.bar_plot_cycler)
axis.set_xticks([0, 1, 2, 3])
axis.bar(states - 0.2, computed_pdf, 0.2, label=r'$\pi_t^T$', zorder=5)
axis.bar(states, πe, 0.2, label=r'$\pi_E^T$', lw="3", zorder=10)
axis.bar(states + 0.2, simpulated_pdf, 0.2, label="Simulation", lw="3", zorder=10)
axis.legend(bbox_to_anchor=(0.8, 0.95), fontsize=16)
config.save_post_asset(figure, "discrete_state_markov_chain_equilibrium", "distribution_comparison")
