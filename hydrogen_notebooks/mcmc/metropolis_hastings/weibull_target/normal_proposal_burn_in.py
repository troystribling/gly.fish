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
target_pdf = stats.weibull(k, λ)

x = numpy.linspace(0.001, 1.6, 500)
figure, axis = pyplot.subplots(figsize=(10, 7))
axis.set_xlabel("X")
axis.set_ylabel("PDF")
axis.set_xlim([0.0, x[-1]])
axis.set_title(f"Weibull Distribution: k={k}, λ={λ}")
axis.plot(x, [target_pdf(j) for j in x])

# %%

nsample = 100000
stepsize = 0.15
x0 = [0.001, 1.0, 2.0, 2.5, 3.0, 3.5]

# %%
# perform mimulations that scan the step size
all_samples = []
all_accepted = []
for i in range(0, len(x0)):
    samples, accepted = mh.metropolis_hastings(target_pdf, mh.normal_proposal, mh.normal_generator, stepsize, nsample=nsample, x0=x0[i])
    all_samples.append(samples)
    all_accepted.append(accepted)

all_samples = numpy.array(all_samples)
all_accepted = numpy.array(all_accepted)

# %%

acceptance = [100.0*a/nsample for a in all_accepted]
title = f"Weibull Distribution, Normal Proposal, k={k}, λ={λ}"
figure, axis = pyplot.subplots(figsize=(10, 7))
axis.set_xlabel(r"$X_0$")
axis.set_ylabel("Acceptance %")
axis.set_title(title)
axis.set_ylim([0.0, 100])
axis.plot(x0, acceptance, zorder=5, marker='o', markersize=15.0, linestyle="None", markeredgewidth=1.0)
config.save_post_asset(figure, "metropolis_hastings_sampling", "normal_proposal_burnin_acceptance")

# %%

sample_idx = 0
all_samples[sample_idx]
title = f"Weibull Distribution, Normal Proposal: Accepted {format(acceptance[sample_idx], '2.0f')}%, " + r"$X_0$=" + f"{x0[sample_idx]}"
gplot.pdf_samples(title, target_pdf, all_samples[sample_idx], "metropolis_hastings_sampling", "normal_proposal_sampled_pdf_burnin-x-001")

# %%

sample_idx = 1
title = f"Weibull Distribution, Normal Proposal: Accepted {format(acceptance[sample_idx], '2.0f')}%, " + r"$X_0$=" + f"{x0[sample_idx]}"
gplot.pdf_samples(title, target_pdf, all_samples[sample_idx], "metropolis_hastings_sampling", "normal_proposal_sampled_pdf_burnin-x-1")

# %%

sample_idx = 2
title = f"Weibull Distribution, Normal Proposal: Accepted {format(acceptance[sample_idx], '2.0f')}%, " + r"$X_0$=" + f"{x0[sample_idx]}"
gplot.pdf_samples(title, target_pdf, all_samples[sample_idx], "metropolis_hastings_sampling", "normal_proposal_sampled_pdf_burnin-x-2")

# %%

sample_idx = 3
title = f"Weibull Distribution, Normal Proposal: Accepted {format(acceptance[sample_idx], '2.0f')}%, " + r"$X_0$=" + f"{x0[sample_idx]}"
gplot.pdf_samples(title, target_pdf, all_samples[sample_idx], "metropolis_hastings_sampling", "normal_proposal_sampled_pdf_burnin-x-25")

# %%

sample_idx = 4
title = f"Weibull Distribution, Normal Proposal: Accepted {format(acceptance[sample_idx], '2.0f')}%, " + r"$X_0$=" + f"{x0[sample_idx]}"
gplot.pdf_samples(title, target_pdf, all_samples[sample_idx], "metropolis_hastings_sampling", "normal_proposal_sampled_pdf_burnin-x-3")

# %%

μ = stats.weibull_mean(k, λ)
title = r"Weibull Distribution, Normal Proposal, $μ_E$ Convergence"
time = range(nsample)
nplot = len(all_samples)

figure, axis = pyplot.subplots(figsize=(10, 7))
axis.set_xlabel("Time")
axis.set_ylabel(r"$μ$")
axis.set_title(title)
axis.set_xlim([1.0, nsample])
axis.set_ylim([-0.2, 3.7])
axis.yaxis.set_ticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
axis.semilogx(time, numpy.full(nsample, μ), label="Target μ", color="#000000")
for i in range(nplot):
    axis.semilogx(time, stats.cummean(all_samples[i]), label=r"$X_0$="+f"{format(x0[i], '2.2f')}")
axis.legend(bbox_to_anchor=(0.9, 0.95))
config.save_post_asset(figure, "metropolis_hastings_sampling", "normal_proposal_burnin-mean-convergence")

# %%

σ = stats.weibull_sigma(k, λ)
title = r"Weibull Distribution, Normal Proposal, $σ_E$ Convergence"
time = range(nsample)
nplot = len(all_samples)

figure, axis = pyplot.subplots(figsize=(10, 7))
axis.set_xlabel("Time")
axis.set_ylabel(r"$σ$")
axis.set_title(title)
axis.set_xlim([1.0, nsample])
axis.set_ylim([-0.05, 1.1])
axis.semilogx(time, numpy.full(nsample, σ), label="Target σ", color="#000000")
for i in range(nplot):
    axis.semilogx(time, stats.cumsigma(all_samples[i]), label=r"$X_0$="+f"{format(x0[i], '2.2f')}")
axis.legend(bbox_to_anchor=(0.95, 0.95))
config.save_post_asset(figure, "metropolis_hastings_sampling", "normal_proposal_burnin-sigma-convergence")

# %%

sample_idx = [0, 5, 9, 11]
title = title = r"Weibull Distribution, Normal Proposal, Autocorrelation"
auto_core_range = range(20000, 50000)
nlag = 100
nplot = len(all_samples)

figure, axis = pyplot.subplots(figsize=(10, 7))
axis.set_title(title)
axis.set_xlabel("Time Lag")
axis.set_xlim([0.0, nlag])
for i in range(nplot):
    ac = stats.autocorrelate(all_samples[i][auto_core_range])
    axis.plot(range(nlag), numpy.real(ac[:nlag]), label=r"$X_0$="+f"{format(x0[i], '2.2f')}")
axis.legend(bbox_to_anchor=(0.9, 0.95))
config.save_post_asset(figure, "metropolis_hastings_sampling", "normal_proposal_burnin-autocorrelation")

# %%

sample_idx = 4
title = f"Weibull Distribution, Normal Proposal: Accepted {format(acceptance[sample_idx], '2.0f')}%, " + r"$X_0$=" + f"{x0[sample_idx]}"
gplot.pdf_samples(title, target_pdf, all_samples[sample_idx][10000:], "metropolis_hastings_sampling", "normal_proposal_sampled_pdf_burnin-removed-x-3")
