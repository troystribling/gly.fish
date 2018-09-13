# %%
%load_ext autoreload
%autoreload 2

import numpy
from matplotlib import pyplot
from glyfish import metropolis_hastings as mh
from glyfish import config
from glyfish import gplot
from glyfish import stats

%matplotlib inline

pyplot.style.use(config.glyfish_style)

# %%

k = 5.0
λ = 1.0
x0 = 1.0
target_pdf = stats.weibull(k, λ)

nsample = 100000
burn_in = 10000
stepsize = 0.12
nsimulations = 6
nlag = 100

# %%
# perform mimulations that scan the step size

all_samples = []
all_accepted = []
for i in range(nsimulations):
    samples, accepted = mh.metropolis_hastings(target_pdf, mh.normal_proposal, mh.normal_generator, stepsize, nsample=nsample, x0=x0)
    all_samples.append(samples)
    all_accepted.append(accepted)

all_samples = numpy.array(all_samples)
all_accepted = numpy.array(all_accepted)

# %%

title = title = r"Weibull Target, Normal Proposal, Autocorrelation: " + f"stepsize={format(stepsize, '2.2f')}, " + r"$X_0$="+f"{format(x0, '2.1f')}"
nplot = len(all_samples)

figure, axis = pyplot.subplots(figsize=(10, 7))
axis.set_title(title)
axis.set_xlabel("Time Lag")
axis.set_xlim([0.0, nlag])
for i in range(nplot):
    auto_core_range = range(burn_in, nsample)
    ac = stats.autocorrelate(all_samples[i][auto_core_range])
    axis.plot(range(nlag), numpy.real(ac[:nlag]))
config.save_post_asset(figure, "metropolis_hastings_sampling", "normal_proposal_thinning-autocorrelation")

# %%

title = title = r"Weibull Target, Normal Proposal, Autocorrelation: " + f"stepsize={format(stepsize, '2.2f')}, " + r"$X_0$="+f"{format(x0, '2.1f')}"
nlag = 100
nplot = len(all_samples)

figure, axis = pyplot.subplots(figsize=(10, 7))
axis.set_title(title)
axis.set_xlabel("Time Lag")
axis.set_xlim([0.0, nlag])
for i in range(nplot):
    thinned_range = range(burn_in, nsample, thin[i])
    ac = stats.autocorrelate(all_samples[i][thinned_range])
    axis.plot(range(nlag), numpy.real(ac[:nlag]), label=f"thinning step={format(thin[i], '2.0f')}")
axis.legend(bbox_to_anchor=(0.9, 0.8))
config.save_post_asset(figure, "metropolis_hastings_sampling", "normal_proposal_thinning-thined-autocorrelation")

# %%

μ = stats.weibull_mean(k, λ)
title = r"Weibull Target, Normal Proposal, $μ_E$ Convergence: " + f"stepsize={format(stepsize, '2.2f')}, " + r"$X_0$="+f"{format(x0, '2.1f')}"
nplot = len(all_samples)

figure, axis = pyplot.subplots(figsize=(10, 7))
axis.set_xlabel("Time")
axis.set_ylabel(r"$μ$")
axis.set_title(title)
axis.set_xlim([1.0, nsample])
axis.set_ylim([-0.2, 1.7])
axis.yaxis.set_ticks([0.0, 0.5, 1.0, 1.5])
axis.semilogx(range(0, nsample - burn_in), numpy.full(nsample - burn_in, μ), label="Target μ", color="#000000")
for i in range(nplot):
    thinned_range = range(burn_in, nsample, thin[i])
    axis.semilogx(range(0, nsample - burn_in, thin[i]), stats.cummean(all_samples[i][thinned_range]), label=f"thinning step={format(thin[i], '2.0f')}")
axis.legend(bbox_to_anchor=(0.6, 0.5))
config.save_post_asset(figure, "metropolis_hastings_sampling", "normal_proposal_thinning-mean-convergence")

# %%

σ = stats.weibull_sigma(k, λ)
title = r"Weibull Target, Normal Proposal, $σ_E$ Convergence: " + f"stepsize={format(stepsize, '2.2f')}, " + r"$X_0$="+f"{format(x0, '2.1f')}"
nplot = len(all_samples)

figure, axis = pyplot.subplots(figsize=(10, 7))
axis.set_xlabel("Time")
axis.set_ylabel(r"$μ$")
axis.set_title(title)
axis.set_xlim([1.0, nsample])
axis.set_ylim([-0.1, 0.6])
axis.yaxis.set_ticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
axis.semilogx(range(0, nsample - burn_in), numpy.full(nsample - burn_in, σ), label="Target μ", color="#000000")
for i in range(nplot):
    thinned_range = range(burn_in, nsample, thin[i])
    axis.semilogx(range(0, nsample - burn_in, thin[i]), stats.cumsigma(all_samples[i][thinned_range]), label=f"thinning step={format(thin[i], '2.0f')}")
axis.legend(bbox_to_anchor=(0.6, 0.55))
config.save_post_asset(figure, "metropolis_hastings_sampling", "normal_proposal_thinning-sigms-convergence")

# %%

figure, axis = pyplot.subplots(nrows=nplots, ncols=1, sharex=True, figsize=(10, 3*nplots))
axis[0].set_title(title)
axis[-1].set_xlabel("Time")

bbox = dict(boxstyle='square,pad=1', facecolor="#FFFFFF", edgecolor="white", alpha=0.75)
for i in range(nplots):
    axis[i].set_xlim([time[0], time[-1] + 1])
    axis[i].set_ylim(ylim)
    axis[i].plot(time, samples[i], lw="2")
    axis[i].text(text_pos[0], text_pos[1], f"stepsize={format(stepsize[i], '2.2f')}", fontsize=13, bbox=bbox)
config.save_post_asset(figure, "metropolis_hastings_sampling", "normal_proposal_thinning-time-series")
