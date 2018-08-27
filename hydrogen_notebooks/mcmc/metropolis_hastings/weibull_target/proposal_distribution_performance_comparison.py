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

nsample = 100000
x0 = 1.0
npts = 25
stepsize = 10**numpy.linspace(-3.0, 1.0, npts)
x0 = 1.0

# %%
# perform mimulations that scan the step size

gamma_samples = []
gamma_accepted = []
for i in range(0, len(stepsize)):
    samples, accepted = mh.metropolis_hastings(target_pdf, mh.gamma_proposal, mh.gamma_generator, stepsize[i], nsample=nsample, x0=x0)
    gamma_samples.append(samples)
    gamma_accepted.append(accepted)

gamma_samples = numpy.array(gamma_samples)
gamma_accepted = numpy.array(gamma_accepted)
gamma_acceptance = 100.0*gamma_accepted/nsample

# perform mimulations that scan the step size

normal_samples = []
normal_accepted = []
for i in range(0, len(stepsize)):
    samples, accepted = mh.metropolis_hastings(target_pdf, mh.normal_proposal, mh.normal_generator, stepsize[i], nsample=nsample, x0=x0)
    normal_samples.append(samples)
    normal_accepted.append(accepted)

normal_samples = numpy.array(normal_samples)
normal_accepted = numpy.array(normal_accepted)
normal_acceptance = 100.0*normal_accepted/nsample

# %%

title = f"Weibull Distribution, Gamma Proposal: k={k}, λ={λ}"
gplot.acceptance(title, stepsize, gamma_acceptance, [0.0005, 20.0], "metropolis_hastings_sampling", "gamma_proposal_acceptance")

# %%

title = f"Weibull Distribution, Normal Proposal: k={k}, λ={λ}"
gplot.acceptance(title, stepsize, normal_acceptance, [0.0005, 20.0], "metropolis_hastings_sampling", "normal_proposal_acceptance")
