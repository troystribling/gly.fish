# %%

import numpy
from matplotlib import pyplot
from scipy import stats
from glyfish import config
from glyfish import metropolis_hastings as mh

%matplotlib inline

pyplot.style.use(config.glyfish_style)

#%%

nsample=100000
stepsize = 0.1
pdf = mh.gamma(5.0)
samples, accepted = mh.metropolis_hastings(pdf, mh.gamma_proposal, mh.gamma_generator, stepsize, nsample=nsample, x0=1.0)
accepted_percent = 100.0*float(accepted)/float(nsample)

figure, axis = pyplot.subplots(figsize=(12, 5))
axis.set_xlabel("X")
axis.set_ylabel("PDF")
axis.set_title(f"Gamma Distribution, Gamma Proposal, Accepted {format(accepted_percent, '2.0f')}%")
_, bins, _ = axis.hist(samples, 50, density=True, color="#336699", alpha=0.6, label=f"Sampled Distribution", edgecolor="#336699", zorder=5)
delta = (bins[-1] - bins[0]) / 200.0
sample_distribution = [pdf(val) for val in numpy.arange(bins[0], bins[-1], delta)]
axis.plot(numpy.arange(bins[0], bins[-1], delta), sample_distribution, color="#A60628", label=f"Sampled Function", zorder=6)
axis.legend()
