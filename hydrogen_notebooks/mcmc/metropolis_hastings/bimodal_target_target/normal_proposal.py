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

target_pdf = stats.bimodal_normal

x = numpy.linspace(-7.0, 7.0, 500)
figure, axis = pyplot.subplots(figsize=(12, 5))
axis.set_xlabel("X")
axis.set_ylabel("PDF")
axis.set_xlim([-7.0, 7.0])
axis.set_title(f"Bimodal Normal Distribution")
axis.plot(x, [target_pdf(j) for j in x])

# %%

nsample = 100000
npts = 25
stepsize = 10**numpy.linspace(-3.0, 2, npts)
x0 = 0.5

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

title = f"Bimodal Normal Distribution, Normal Proposal"
gplot.acceptance(title, stepsize, acceptance, [0.0005, 200.0])

# %%

sample_idx = 8
title = f"Bimodal Normal Distribution, Normal Proposal, Accepted {format(acceptance[sample_idx], '2.0f')}%, stepsize={format(stepsize[sample_idx], '2.3f')}"
gplot.pdf_samples(title, target_pdf, all_samples[sample_idx], xrange=numpy.arange(-7.0, 7.0, 0.05))

# %%

sample_idx = 13
title = f"Bimodal Normal Distribution, Normal Proposal, Accepted {format(acceptance[sample_idx], '2.0f')}%, stepsize={format(stepsize[sample_idx], '2.3f')}"
gplot.pdf_samples(title, target_pdf, all_samples[sample_idx])

# %%

sample_idx = 15
title = f"Bimodal Normal Distribution, Normal Proposal, Accepted {format(acceptance[sample_idx], '2.0f')}%, stepsize={format(stepsize[sample_idx], '2.3f')}"
gplot.pdf_samples(title, target_pdf, all_samples[sample_idx])

# %%

sample_idx = 20
title = f"Bimodal Normal Distribution, Normal Proposal, Accepted {format(acceptance[sample_idx], '2.0f')}%, stepsize={format(stepsize[sample_idx], '2.3f')}"
gplot.pdf_samples(title, target_pdf, all_samples[sample_idx])

# %%

sample_idx = [8, 13, 15, 20]
title = f"Bimodal Normal Distribution Samples, Normal Proposal, Stepsize comparison"
time = range(51000, 51500)
time_series_samples = [all_samples[i][time] for i in sample_idx]
time_series_stepsize = stepsize[sample_idx]
time_series_acceptance = acceptance[sample_idx]
gplot.steps_size_time_series(title, time_series_samples, time, time_series_stepsize, time_series_acceptance, [-5.0, 5.0], [51010, -3.8])

# %%

μ = stats.bimodal_normal_mean()
title = f"Weibull Distribution, Normal Proposal, sample μ convergence stepsize comparison"
time = range(nsample)
mean_samples = [all_samples[i][time] for i in sample_idx]
mean_stepsize = stepsize[sample_idx]
gplot.step_size_mean(title, mean_samples, time, μ, mean_stepsize)

# %%

σ = stats.bimodal_normal_sigma()
title = f"Weibull Distribution, Normal Proposal, sample σ convergence stepsize comparison"
time = range(nsample)
sigma_samples = [all_samples[i][time] for i in sample_idx]
sigma_stepsize = stepsize[sample_idx]
gplot.step_size_sigma(title, sigma_samples, time, σ, sigma_stepsize)

# %%

title = f"Weibull Distribution, Normal Proposal, Autocorrelation, stepsize comparison"
auto_core_range = range(20000, 50000)
autocorr_samples = [all_samples[i][auto_core_range] for i in sample_idx]
autocorr_stepsize = stepsize[sample_idx]
nplot = 200
gplot.step_size_autocor(title, autocorr_samples, autocorr_stepsize, nplot)
