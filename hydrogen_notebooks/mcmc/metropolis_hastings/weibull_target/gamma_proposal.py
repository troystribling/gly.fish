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
stepsize = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.012, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0]
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

# %%

acceptance = [100.0*a/nsample for a in all_accepted]
title = f"Weibull Distribution, Gamma Proposal, k={k}, λ={λ}"
gplot.acceptance(title, stepsize, acceptance, [0.0005, 2.0])

# %%

sample_idx = 0
title = f"Weibull Distribution, Gamma Proposal, Accepted {format(acceptance[sample_idx], '2.0f')}%, stepsize={stepsize[sample_idx]}"
gplot.pdf_samples(title, target_pdf, all_samples[sample_idx])

# %%

sample_idx = 5
title = f"Weibull Distribution, Gamma Proposal, Accepted {format(acceptance[sample_idx], '2.0f')}%, stepsize={stepsize[sample_idx]}"
gplot.pdf_samples(title, target_pdf, all_samples[sample_idx])


# %%

sample_idx = 8
title = f"Weibull Distribution, Gamma Proposal, Accepted {format(acceptance[sample_idx], '2.0f')}%, stepsize={stepsize[sample_idx]}"
gplot.pdf_samples(title, target_pdf, all_samples[sample_idx])

# %%

sample_idx = 11
title = f"Weibull Distribution, Gamma Proposal, Accepted {format(acceptance[sample_idx], '2.0f')}%, stepsize={stepsize[sample_idx]}"
gplot.pdf_samples(title, target_pdf, all_samples[sample_idx])

# %%

sample_idx = [0, 5, 8, 11]
title = f"Weibull Distribution Samples, Gamma Proposal, Stepsize comparison"
time = range(51000, 51500)
time_series_samples = [all_samples[i][time] for i in sample_idx]
time_series_stepsize = [stepsize[i] for i in sample_idx]
time_series_acceptance = [acceptance[i] for i in sample_idx]
gplot.steps_size_time_series(title, time_series_samples, time, time_series_stepsize, time_series_acceptance, [0.0, 1.75], [51010, 0.185])


# %%

μ = stats.weibull_mean(5.0, 1.0)
sample_idx = [0, 5, 8, 11]
title = f"Weibull Distribution, Gamma Proposal, sample μ convergence stepsize comparison"
time = range(nsample)
mean_samples = [all_samples[i][time] for i in sample_idx]
mean_stepsize = [stepsize[i] for i in sample_idx]
gplot.step_size_mean(title, mean_samples, time, μ, mean_stepsize)

# %%

σ = stats.weibull_sigma(5.0, 1.0)
sample_idx = [0, 5, 8, 11]
title = f"Weibull Distribution, Gamma Proposal, sample σ convergence stepsize comparison"
time = range(nsample)
sigma_samples = [all_samples[i][time] for i in sample_idx]
sigma_stepsize = [stepsize[i] for i in sample_idx]
gplot.step_size_sigma(title, sigma_samples, time, σ, sigma_stepsize)

# %%

sample_idx = [0, 5, 8, 11]
title = f"Weibull Distribution, Gamma Proposal, Autocorrelation stepsize comparison"
time = range(nsample)
autocorr_samples = [all_samples[i][time] for i in sample_idx]
autocorr_stepsize = [stepsize[i] for i in sample_idx]
