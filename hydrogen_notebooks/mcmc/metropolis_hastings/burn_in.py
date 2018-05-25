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
stepsize = 0.15
x0 = [0.01, 0.5, 1.0, 1.5, 2.0]

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
figure, axis = pyplot.subplots(figsize=(12, 5))
axis.set_xlabel(r"%X_0%")
axis.set_ylabel("Acceptance %")
axis.set_title(title)
axis.set_ylim([0.0, 100])
axis.plot(x0, acceptance, zorder=5, marker='o', color="#336699", markersize=15.0, linestyle="None", markeredgewidth=1.0, alpha=0.5, label="Simulation")
axis.legend()

# %%

sample_idx = 0
all_samples[sample_idx]
title = f"Weibull Distribution, Normal Proposal, Accepted {format(acceptance[sample_idx], '2.0f')}%, X0={x0[sample_idx]}"
gplot.pdf_samples(title, target_pdf, all_samples[sample_idx])

# %%

sample_idx = 1
title = f"Weibull Distribution, Normal Proposal, Accepted {format(acceptance[sample_idx], '2.0f')}%, stepsize={stepsize[sample_idx]}"
gplot.pdf_samples(title, target_pdf, all_samples[sample_idx])


# %%

sample_idx = 2
title = f"Weibull Distribution, Normal Proposal, Accepted {format(acceptance[sample_idx], '2.0f')}%, stepsize={stepsize[sample_idx]}"
gplot.pdf_samples(title, target_pdf, all_samples[sample_idx])

# %%

sample_idx = 3
title = f"Weibull Distribution, Normal Proposal, Accepted {format(acceptance[sample_idx], '2.0f')}%, stepsize={stepsize[sample_idx]}"
gplot.pdf_samples(title, target_pdf, all_samples[sample_idx])

# %%

sample_idx = 4
title = f"Weibull Distribution, Normal Proposal, Accepted {format(acceptance[sample_idx], '2.0f')}%, stepsize={stepsize[sample_idx]}"
gplot.pdf_samples(title, target_pdf, all_samples[sample_idx])

# %%

title = f"Weibull Distribution Samples, Normal Proposal, Stepsize comparison"
time = range(51000, 51500)
time_series_samples = [all_samples[i][time] for i in sample_idx]
time_series_stepsize = [stepsize[i] for i in sample_idx]
time_series_acceptance = [acceptance[i] for i in sample_idx]
gplot.steps_size_time_series(title, time_series_samples, time, time_series_stepsize, time_series_acceptance, [0.0, 1.75], [51010, 0.185])


# %%

μ = stats.weibull_mean(5.0, 1.0)
sample_idx = [0, 5, 9, 11]
title = f"Weibull Distribution, Normal Proposal, sample μ convergence stepsize comparison"
time = range(nsample)
mean_samples = [all_samples[i][time] for i in sample_idx]
mean_stepsize = [stepsize[i] for i in sample_idx]
gplot.step_size_mean(title, mean_samples, time, μ, mean_stepsize)

# %%

σ = stats.weibull_sigma(5.0, 1.0)
sample_idx = [0, 5, 9, 11]
title = f"Weibull Distribution, Normal Proposal, sample σ convergence stepsize comparison"
time = range(nsample)
sigma_samples = [all_samples[i][time] for i in sample_idx]
sigma_stepsize = [stepsize[i] for i in sample_idx]
gplot.step_size_sigma(title, sigma_samples, time, σ, sigma_stepsize)

# %%

sample_idx = [0, 5, 9, 11]
title = f"Weibull Distribution, Normal Proposal, Autocorrelation, stepsize comparison"
auto_core_range = range(20000, 50000)
autocorr_samples = [all_samples[i][auto_core_range] for i in sample_idx]
autocorr_stepsize = [stepsize[i] for i in sample_idx]
nplot = 100
gplot.step_size_autocor(title, autocorr_samples, autocorr_stepsize, nplot)
