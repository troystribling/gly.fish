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
figure, axis = pyplot.subplots(figsize=(10, 7))
axis.set_xlabel("X")
axis.set_ylabel("PDF")
axis.set_xlim([0.0, x[-1]])
axis.set_title(f"Weibull Distribution, k={k}, λ={λ}")
axis.plot(x, [target_pdf(j) for j in x])

# %%

nsample = 100000
x0 = 1.0
npts = 25
stepsize = 10**numpy.linspace(-3.0, 2, npts)

# %%
# perform mimulations that scan the step size

all_samples = []
all_accepted = []
for i in range(0, len(stepsize)):
    samples, accepted = mh.metropolis_hastings(target_pdf, mh.normal_proposal, mh.normal_generator, stepsize[i], nsample=nsample, x0=x0)
    all_samples.append(samples)
    all_accepted.append(accepted)

all_samples = numpy.array(all_samples)
all_accepted = numpy.array(all_accepted)
acceptance = 100.0*all_accepted/nsample

# %%

title = r"Weibull Target, Normal Proposal: $X_0$="+f"{format(x0, '2.1f')}"
mh.acceptance(title, stepsize, acceptance, [0.0005, 20.0], [4, 10, 15], "metropolis_hastings_sampling", "normal_proposal_acceptance")

# %%

sample_idx = 4
title = f"Weibull Target, Normal Proposal: Accepted {format(acceptance[sample_idx], '2.0f')}%, stepsize={format(stepsize[sample_idx], '2.2f')}, "+r"$X_0$="+f"{format(x0, '2.1f')}"
gplot.pdf_samples(title, target_pdf, all_samples[sample_idx], "metropolis_hastings_sampling", "normal_proposal_sampled_pdf-99", xrange=numpy.arange(0.1, 1.7, 0.01), )

# %%

sample_idx = 10
title = f"Weibull Target, Normal Proposal: Accepted {format(acceptance[sample_idx], '2.0f')}%, stepsize={format(stepsize[sample_idx], '2.2f')}, "+r"$X_0$="+f"{format(x0, '2.1f')}"
gplot.pdf_samples(title, target_pdf, all_samples[sample_idx], "metropolis_hastings_sampling", "normal_proposal_sampled_pdf-82", ylimit=[0.0, 2.2])

# %%

sample_idx = 13
title = f"Weibull Target, Normal Proposal: Accepted {format(acceptance[sample_idx], '2.0f')}%, stepsize={format(stepsize[sample_idx], '2.2f')}, "+r"$X_0$="+f"{format(x0, '2.1f')}"
gplot.pdf_samples(title, target_pdf, all_samples[sample_idx], "metropolis_hastings_sampling", "normal_proposal_sampled_pdf-44", ylimit=[0.0, 2.2])

# %%

sample_idx = 15
title = f"Weibull Distribution, Normal Proposal: Accepted {format(acceptance[sample_idx], '2.0f')}%, stepsize={format(stepsize[sample_idx], '2.2f')}, "+r"$X_0$="+f"{format(x0, '2.1f')}"
gplot.pdf_samples(title, target_pdf, all_samples[sample_idx], "metropolis_hastings_sampling", "normal_proposal_sampled_pdf-19", ylimit=[0.0, 2.2])

# %%

sample_idx = 16
title = f"Weibull Target, Normal Proposal: Accepted {format(acceptance[sample_idx], '2.0f')}%, stepsize={format(stepsize[sample_idx], '2.2f')}, "+r"$X_0$="+f"{format(x0, '2.1f')}"
gplot.pdf_samples(title, target_pdf, all_samples[sample_idx], "metropolis_hastings_sampling", "normal_proposal_sampled_pdf-12", ylimit=[0.0, 2.2])

# %%

sample_idx = [4, 10, 15]
title = r"Weibull Target, Normal Proposal, Stepsize comparison: $X_0$="+f"{format(x0, '2.1f')}"
time = range(51000, 51500)
time_series_samples = [all_samples[i][time] for i in sample_idx]
time_series_stepsize = stepsize[sample_idx]
time_series_acceptance = acceptance[sample_idx]
mh.steps_size_time_series(title, time_series_samples, time, time_series_stepsize, time_series_acceptance, [-0.2, 1.75], [51250, 0.025], "metropolis_hastings_sampling", "normal_proposal_time_series_stepsize_comparison")

# %%

μ = stats.weibull_mean(k, λ)
title = r"Weibull Target, Normal Proposal, $μ_E$ Convergence: $X_0$="+f"{format(x0, '2.1f')}"
time = range(nsample)
mean_samples = [all_samples[i][time] for i in sample_idx]
mean_stepsize = stepsize[sample_idx]
mh.step_size_mean(title, mean_samples, time, μ, mean_stepsize, "metropolis_hastings_sampling", "normal_proposal_mean_convergence_stepsize_comparison", (0.75, 0.7))

# %%

σ = stats.weibull_sigma(k, λ)
title = r"Weibull Target, Normal Proposal, $σ_E$ Convergence: $X_0$="+f"{format(x0, '2.1f')}"
time = range(nsample)
sigma_samples = [all_samples[i][time] for i in sample_idx]
sigma_stepsize = stepsize[sample_idx]
mh.step_size_sigma(title, sigma_samples, time, σ, sigma_stepsize, "metropolis_hastings_sampling", "normal_proposal_sigma_convergence_stepsize_comparison", (0.7, 0.3))

# %%

title = f"Weibull Target, Normal Proposal, Autocorrelation, stepsize comparison"
auto_core_range = range(20000, 50000)
autocorr_samples = [all_samples[i][auto_core_range] for i in sample_idx]
autocorr_stepsize = stepsize[sample_idx]
nplot = 100

# %%

mh.step_size_autocor(title, autocorr_samples, autocorr_stepsize, nplot, "metropolis_hastings_sampling", "normal_proposal_autocorrelation_convergence_stepsize_comparison")
