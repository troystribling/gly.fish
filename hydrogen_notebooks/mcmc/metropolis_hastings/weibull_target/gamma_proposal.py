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

x = numpy.linspace(0.001, 2.0, 500)
figure, axis = pyplot.subplots(figsize=(12, 5))
axis.set_xlabel("X")
axis.set_ylabel("PDF")
axis.set_xlim([0.0, x[-1]])
axis.set_title(f"Weibull Distribution, k={k}, λ={λ}")
axis.plot(x, [target_pdf(j) for j in x])

# %%

nsample = 100000
x0 = 1.0
npts = 25
stepsize = 10**numpy.linspace(-5.0, 1.0, npts)
x0 = 1.0

# %%
# perform mimulations that scan the step size

all_samples = []
all_accepted = []
for i in range(0, len(stepsize)):
    samples, accepted = mh.metropolis_hastings(target_pdf, mh.gamma_proposal, mh.gamma_generator, stepsize[i], nsample=nsample, x0=x0)
    all_samples.append(samples)
    all_accepted.append(accepted)

all_samples = numpy.array(all_samples)
all_accepted = numpy.array(all_accepted)
acceptance = 100.0*all_accepted/nsample

# %%

title = f"Weibull Distribution, Gamma Proposal: k={k}, λ={λ}"
gplot.acceptance(title, stepsize, acceptance, [0.0005, 20.0], "metropolis_hastings_sampling", "gamma_proposal_acceptance")

# %%

sample_idx = 5
title = f"Weibull Distribution, Gamma Proposal: Accepted {format(acceptance[sample_idx], '2.0f')}%, stepsize={format(stepsize[sample_idx], '2.3f')}"
gplot.pdf_samples(title, target_pdf, all_samples[sample_idx], "metropolis_hastings_sampling", "gamma_proposal_sampled_pdf-98")

# %%

sample_idx = 13
title = f"Weibull Distribution, Gamma Proposal: Accepted {format(acceptance[sample_idx], '2.0f')}%, stepsize={format(stepsize[sample_idx], '2.3f')}"
gplot.pdf_samples(title, target_pdf, all_samples[sample_idx], "metropolis_hastings_sampling", "gamma_proposal_sampled_pdf-80")

# %%

sample_idx = 17
title = f"Weibull Distribution, Gamma Proposal: Accepted {format(acceptance[sample_idx], '2.0f')}%, stepsize={format(stepsize[sample_idx], '2.3f')}"
gplot.pdf_samples(title, target_pdf, all_samples[sample_idx], "metropolis_hastings_sampling", "gamma_proposal_sampled_pdf-48")

# %%

sample_idx = 20
title = f"Weibull Distribution, Gamma Proposal, Accepted {format(acceptance[sample_idx], '2.0f')}%, stepsize={format(stepsize[sample_idx], '2.3f')}"
gplot.pdf_samples(title, target_pdf, all_samples[sample_idx], "metropolis_hastings_sampling", "gamma_proposal_sampled_pdf-07")

# %%

sample_idx = [5, 13, 20]
title = f"Weibull Distribution Samples, Gamma Proposal, Stepsize comparison"
time = range(51000, 51500)
time_series_samples = [all_samples[i][time] for i in sample_idx]
time_series_stepsize = stepsize[sample_idx]
time_series_acceptance = acceptance[sample_idx]
gplot.steps_size_time_series(title, time_series_samples, time, time_series_stepsize, time_series_acceptance, [-0.2, 1.75], [51010, 0.15], "metropolis_hastings_sampling", "gamma_proposal_time_series_stepsize_comparison")

# %%

μ = stats.weibull_mean(5.0, 1.0)
title = f"Weibull Distribution, Gamma Proposal, Sample μ Convergence"
time = range(nsample)
mean_samples = [all_samples[i][time] for i in sample_idx]
mean_stepsize = stepsize[sample_idx]
gplot.step_size_mean(title, mean_samples, time, μ, mean_stepsize, "metropolis_hastings_sampling", "gamma_proposal_mean_convergence_stepsize_comparison")

# %%

σ = stats.weibull_sigma(5.0, 1.0)
title = f"Weibull Distribution, Gamma Proposal, sample σ convergence stepsize comparison"
time = range(nsample)
sigma_samples = [all_samples[i][time] for i in sample_idx]
sigma_stepsize = stepsize[sample_idx]
gplot.step_size_sigma(title, sigma_samples, time, σ, sigma_stepsize, "metropolis_hastings_sampling", "gamma_proposal_sigma_convergence_stepsize_comparison")

# %%

title = f"Weibull Distribution, Gamma Proposal, Autocorrelation stepsize comparison"
auto_core_range = range(20000, 50000)
autocorr_samples = [all_samples[i][auto_core_range] for i in sample_idx]
autocorr_stepsize = stepsize[sample_idx]
nplot = 200
gplot.step_size_autocor(title, autocorr_samples, autocorr_stepsize, nplot, "metropolis_hastings_sampling", "gamma_proposal_autocorrelation_convergence_stepsize_comparison")
