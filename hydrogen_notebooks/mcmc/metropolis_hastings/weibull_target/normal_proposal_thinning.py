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

nsample = 50000
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

title = r"Weibull Target, Normal Proposal, Autocorrelation: " + f"stepsize={format(stepsize, '2.2f')}, " + r"$X_0$="+f"{format(x0, '2.1f')}"
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

title = r"Thinned Weibull Target, Normal Proposal, Autocorrelation: " + f"stepsize={format(stepsize, '2.2f')}, " + r"$X_0$="+f"{format(x0, '2.1f')}"
nlag = 100
nplot = len(all_samples)

figure, axis = pyplot.subplots(figsize=(10, 7))
axis.set_title(title)
axis.set_xlabel("Time Lag")
axis.set_xlim([0.0, nlag])
for i in range(nplot):
    thinned_range = range(burn_in, nsample, i+1)
    ac = stats.autocorrelate(all_samples[0][thinned_range])
    axis.plot(range(nlag), numpy.real(ac[:nlag]), label=f"η={format(i+1, '2.0f')}")
axis.legend(bbox_to_anchor=(0.9, 0.8))
config.save_post_asset(figure, "metropolis_hastings_sampling", "normal_proposal_thinning-thined-autocorrelation")

# %%

μ = stats.weibull_mean(k, λ)
title = r"Thinned Weibull Target, Normal Proposal, $μ_E$ Convergence: " + f"stepsize={format(stepsize, '2.2f')}, " + r"$X_0$="+f"{format(x0, '2.1f')}"
nplot = len(all_samples)

figure, axis = pyplot.subplots(figsize=(10, 7))
axis.set_xlabel("Time")
axis.set_ylabel(r"$μ$")
axis.set_title(title)
axis.set_xlim([1.0, nsample])
axis.set_ylim([0.4, 1.2])
axis.yaxis.set_ticks([0.5, 0.75, 1.0])
axis.semilogx(range(0, nsample - burn_in), numpy.full(nsample - burn_in, μ), label="Target μ", color="#000000")
for i in range(nplot):
    thinned_range = range(burn_in, nsample, i+1)
    mean = stats.cummean(all_samples[0][thinned_range])
    axis.semilogx(range(len(mean)), mean, label=f"η={format(i+1, '2.0f')}")
axis.legend(bbox_to_anchor=(0.95, 0.5))
config.save_post_asset(figure, "metropolis_hastings_sampling", "normal_proposal_thinning-mean-convergence")

# %%

σ = stats.weibull_sigma(k, λ)
title = r"Thinned Weibull Target, Normal Proposal, $σ_E$ Convergence: " + f"stepsize={format(stepsize, '2.2f')}, " + r"$X_0$="+f"{format(x0, '2.1f')}"
nplot = len(all_samples)

figure, axis = pyplot.subplots(figsize=(10, 7))
axis.set_xlabel("Time")
axis.set_ylabel(r"$μ$")
axis.set_title(title)
axis.set_xlim([1.0, nsample])
axis.set_ylim([-0.01, 0.35])
axis.yaxis.set_ticks([0.0, 0.1, 0.2, 0.3])
axis.semilogx(range(0, nsample - burn_in), numpy.full(nsample - burn_in, σ), label="Target μ", color="#000000")
for i in range(nplot):
    thinned_range = range(burn_in, nsample, i+1)
    sigma = stats.cumsigma(all_samples[0][thinned_range])
    axis.semilogx(range(len(sigma)), sigma, label=f"η={format(i+1, '2.0f')}")
axis.legend(bbox_to_anchor=(0.95, 0.55))
config.save_post_asset(figure, "metropolis_hastings_sampling", "normal_proposal_thinning-sigma-convergence")

# %%

thin = [1, 2, 5]
plot_interval = [20000, 20500]
text_pos = [[20050, 0.025], [10050, 0.025], [4050, 0.025]]
title = r"Thinned Weibull Target, Normal Proposal, Time Series: " + f"stepsize={format(stepsize, '2.2f')}, " + r"$X_0$="+f"{format(x0, '2.1f')}"

figure, axis = pyplot.subplots(nrows=3, ncols=1, figsize=(10, 9))
axis[0].set_title(title)
axis[-1].set_xlabel("Time")

bbox = dict(boxstyle='square,pad=1', facecolor="#FFFFFF", edgecolor="white", alpha=0.75)
for i in range(3):
    thinned_range = range(burn_in, nsample, thin[i])
    plot_start = int(plot_interval[0]/thin[i])
    plot_range = range(plot_start, plot_start + plot_interval[1] - plot_interval[0])
    thinned_samples = all_samples[0][thinned_range]
    axis[i].set_xlim(plot_start, plot_start + plot_interval[1] - plot_interval[0])
    axis[i].set_ylim([-0.2, 1.7])
    axis[i].plot(plot_range, thinned_samples[plot_range], lw=2)
    axis[i].text(text_pos[i][0], text_pos[i][1], f"η={format(thin[i], '2.0f')}", fontsize=15, bbox=bbox)
config.save_post_asset(figure, "metropolis_hastings_sampling", "normal_proposal_thinning-time-series")

# %%

thin = 1
thinned_range = range(burn_in, nsample, thin)
title = f"Thinned Weibull Target, Normal Proposal: η={format(thin, '2.0f')}, stepsize={format(stepsize, '2.2f')}, "+r"$X_0$="+f"{format(x0, '2.1f')}"
gplot.pdf_samples(title, target_pdf, all_samples[0][thinned_range], "metropolis_hastings_sampling", f"normal_proposal_sampled_pdf_thinning-{thin}", ylimit=[0.0, 2.2])

# %%

thin = 3
thinned_range = range(burn_in, nsample, thin)
title = f"Thinned Weibull Target, Normal Proposal: η={format(thin, '2.0f')}, stepsize={format(stepsize, '2.2f')}, "+r"$X_0$="+f"{format(x0, '2.1f')}"
gplot.pdf_samples(title, target_pdf, all_samples[0][thinned_range], "metropolis_hastings_sampling", f"normal_proposal_sampled_pdf_thinning-{thin}", ylimit=[0.0, 2.2])

# %%

thin = 6
thinned_range = range(burn_in, nsample, thin)
title = f"Thinned Weibull Target, Normal Proposal: η={format(thin, '2.0f')}, stepsize={format(stepsize, '2.2f')}, "+r"$X_0$="+f"{format(x0, '2.1f')}"
gplot.pdf_samples(title, target_pdf, all_samples[0][thinned_range], "metropolis_hastings_sampling", f"normal_proposal_sampled_pdf_thinning-{thin}", ylimit=[0.0, 2.2])

# %%
# perform mimulations that scan the step size

nsample = 500000
samples, accepted = mh.metropolis_hastings(target_pdf, mh.normal_proposal, mh.normal_generator, stepsize, nsample=nsample, x0=x0)

# %%

thin = 6
thinned_range = range(burn_in, nsample, thin)
title = f"Thinned Weibull Target, Normal Proposal: η={format(thin, '2.0f')}, stepsize={format(stepsize, '2.2f')}, "+r"$X_0$="+f"{format(x0, '2.1f')}"
gplot.pdf_samples(title, target_pdf, samples[thinned_range], "metropolis_hastings_sampling", f"normal_proposal_sampled_pdf_thinning-large-run-{thin}", ylimit=[0.0, 2.2])
