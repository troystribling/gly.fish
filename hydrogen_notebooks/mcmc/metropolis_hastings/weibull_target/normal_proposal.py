# %%
%load_ext autoreload
%autoreload 2

import numpy
from matplotlib import pyplot
from scipy import stats
from glyfish import config
from glyfish import metropolis_hastings as mh
from glyfish import gplot

%matplotlib inline

pyplot.style.use(config.glyfish_style)

#%%

k = 5.0
λ = 1.0
target_pdf = mh.weibull(5.0, 1.0)

#%%

x = numpy.linspace(0.001, 2.0, 500)
figure, axis = pyplot.subplots(figsize=(12, 5))
axis.set_xlabel("X")
axis.set_ylabel("PDF")
axis.set_xlim([0.0, x[-1]])
axis.set_title(f"Weibull Distribution, k={k}, λ={λ}")
axis.plot(x, [target_pdf(j) for j in x])

#%%

nsample=100000
stepsize = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
x0 = 1.0

#%%

all_samples = []
all_accepted = []
for i in range(0, len(stepsize)):
    samples, accepted = mh.metropolis_hastings(target_pdf, mh.normal_proposal, mh.normal_generator, stepsize[i], nsample=nsample, x0=x0)
    all_samples.append(samples)
    all_accepted.append(accepted)

all_samples = numpy.array(all_samples)
all_accepted = numpy.array(all_accepted)

#%%

acceptance = [100.0*a/nsample for a in all_accepted]
title = f"Weibull Distribution, k={k}, λ={λ}"
gplot.acceptance(title, stepsize, acceptance)

#%%

title = f"Weibull Distribution, Normal Proposal, Accepted {format(acceptance[2], '2.0f')}%"
gplot.pdf_samples(title, target_pdf, all_samples[2])

# %%

start = 5000
end = 5500

figure, axis = pyplot.subplots(figsize=(12, 5))
axis.set_xlabel("Time")
axis.set_ylabel("X")
axis.set_xlim([start, end])
axis.set_title(f"Wiebull Timeseries, Accepted {format(accepted_percent, '2.0f')}%")
axis.plot(time[start:end], samples[start:end], lw="1")
