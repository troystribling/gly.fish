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

target_pdf = stats.arcsine

x = numpy.linspace(0.001, 0.999, 500)
figure, axis = pyplot.subplots(figsize=(12, 5))
axis.set_xlabel("X")
axis.set_ylabel("PDF")
axis.set_xlim([0.0, 1.0])
axis.set_title(f"Arcsine Distribution")
axis.plot(x, [target_pdf(j) for j in x])

# %%

nsample = 100000
stepsize = [0.01]
x0 = 0.5

# %%
# perform mimulations that scan the step size

all_samples = []
all_accepted = []
for i in range(0, len(stepsize)):
    samples, accepted = mh.metropolis_hastings(target_pdf, mh.uniform_proposal, mh.uniform_generator, stepsize[i], nsample=nsample, x0=x0)
    all_samples.append(samples)
    all_accepted.append(accepted)

all_samples = numpy.array(all_samples)
all_accepted = numpy.array(all_accepted)
acceptance = 100.0*all_accepted/nsample

# %%

sample_idx = 0
title = f"Arcsine Distribution, Uniform Proposal, Accepted {format(acceptance[sample_idx], '2.0f')}%"
gplot.pdf_samples(title, target_pdf, all_samples[sample_idx], xrange=numpy.arange(0.001, 0.999, .001), ylimit=[0.0, 8.0])

# %%

title = f"Arcsine Distribution Samples, Uniform Proposal, accepted={format(acceptance[0], '2.0f')}%"
time = range(51000, 51500)
gplot.time_series(title, all_samples[0][time], time, [0.0, 1.0])

# %%

μ = 0.5
sample_idx = [0]
title = f"Arcsine Distribution, Uniform Proposal, sample μ"
time = range(nsample)
mean_samples = all_samples[0][time]

figure, axis = pyplot.subplots(figsize=(12, 6))
axis.set_xlabel("Time")
axis.set_ylabel(r"$μ$")
axis.set_title(title)
axis.set_xlim([10.0, nsample])
axis.semilogx(time, numpy.full((len(time)), μ), label="Target μ", color="#000000")
axis.semilogx(time, stats.cummean(samples), label=f"Samples")
axis.legend()

# %%

σ = numpy.sqrt(0.125)
title = f"Arcsine Distribution, Uniform Proposal, sample σ convergence"
time = range(nsample)
sigma_samples = all_samples[0]

figure, axis = pyplot.subplots(figsize=(12, 6))
axis.set_xlabel("Time")
axis.set_ylabel(r"$σ$")
axis.set_title(title)
axis.set_xlim([1.0, nsample])
axis.semilogx(time, numpy.full((len(time)), σ), label="Target σ", color="#000000")
axis.semilogx(time, stats.cumsigma(samples), label=f"Samples")
axis.legend()

# %%

title = f"Arcsine Distribution, Uniform Proposal, Autocorrelation"
auto_core_range = range(20000, 50000)
autocorr_samples = all_samples[0][auto_core_range]
nplot = 100

figure, axis = pyplot.subplots(figsize=(12, 9))
axis.set_title(title)
axis.set_xlabel("Time Lag")
axis.set_xlim([0, nplot])
ac = stats.autocorrelate(samples)
axis.plot(range(nplot), numpy.real(ac[:nplot]))
