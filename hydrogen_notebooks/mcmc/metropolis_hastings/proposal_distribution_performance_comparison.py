# %%
%load_ext autoreload
%autoreload 2

import numpy
import scipy
from matplotlib import pyplot
from glyfish import metropolis_hastings as mh
from glyfish import config
from glyfish import gplot
from glyfish import stats

%matplotlib inline

pyplot.style.use(config.glyfish_style)

# %%

def acceptance_plot(title, x, y, idx, text_pos, best_idx, xlim, plot):
    slope, y0, r2 = acceptance_regress(x[idx:], y[idx:])
    xfit = numpy.linspace(-1.0, 3.0, 100)
    bbox = dict(boxstyle='square,pad=1', facecolor="#FFFFFF", edgecolor="white")
    figure, axis = pyplot.subplots(figsize=(10, 7))
    axis.set_xlabel("Step Size")
    axis.set_ylabel("Acceptance %")
    axis.set_title(title)
    axis.set_xlim(xlim)
    axis.set_prop_cycle(config.alternate_cycler)
    axis.set_ylim([0.05, 200.0])
    axis.loglog(x, y, zorder=6, marker='o', markersize=15.0, linestyle="None", markeredgewidth=1.0, label="Simulations")
    axis.loglog(x[best_idx], y[best_idx], zorder=6, marker='o', markersize=15.0, linestyle="None", markeredgewidth=1.0, label="Best")
    axis.loglog(10**xfit, acceptance_fit(xfit, slope, y0), zorder=5, color="#320075", label="Fit")
    axis.loglog(numpy.linspace(xlim[0], xlim[1], 100), numpy.full((100), 100.0), zorder=5, color="#320075")
    axis.text(text_pos[0], text_pos[1], f"slope={format(slope, '2.2f')}", fontsize=16, bbox=bbox)
    axis.legend(bbox_to_anchor=(0.4, 0.5))
    config.save_post_asset(figure, "metropolis_hastings_sampling", plot)

def acceptance_regress(x, y):
    slope, y0, r_value, _, _ = scipy.stats.linregress(numpy.log10(x), numpy.log10(y))
    return slope, y0, r_value**2

def acceptance_fit(x, slope, y0):
    return 10**(slope*x + y0)

def run_simulation(stepsize, target_pdf, proposal, generator, nsample, x0):
    all_samples = []
    all_accepted = []
    for i in range(0, len(stepsize)):
        samples, accepted = mh.metropolis_hastings(target_pdf, proposal, generator, stepsize[i], nsample=nsample, x0=x0)
        all_samples.append(samples)
        all_accepted.append(accepted)
    all_samples = numpy.array(all_samples)
    all_accepted = numpy.array(all_accepted)
    acceptance = 100.0*all_accepted/nsample
    return all_samples, acceptance

# %%

k = 5.0
λ = 1.0
target_pdf = stats.weibull(k, λ)

nsample = 100000
x0 = 1.0
npts = 25
x0 = 1.0

# %%
# weibull target, normal proposal

nsample = 100000
x0 = 1.0
npts = 25
normal_stepsize = 10**numpy.linspace(-3.0, 2.0, npts)
normal_samples, normal_acceptance = run_simulation(normal_stepsize, target_pdf, mh.normal_proposal, mh.normal_generator, nsample, x0)

# %%

σ = stats.weibull_sigma(k, λ)
normalized_step_size = normal_stepsize/σ
title = f"Weibull Distribution, Normal Proposal, Normalized Stepsize: k={k}, λ={λ}"
acceptance_plot(title, normalized_step_size, normal_acceptance, 13, [50.0, 10.0], 10, [10.0**-3, 10.0**3], "normal_proposal_acceptance_fit")

# %%
# weibull target, gamma proposal

nsample = 100000
x0 = 1.0
npts = 25
gamma_stepsize = 10**numpy.linspace(-5.0, 1.0, npts)
gamma_samples, gamma_acceptance = run_simulation(gamma_stepsize, target_pdf, mh.gamma_proposal, mh.gamma_generator, nsample, x0)

# %%

shape = numpy.zeros(len(gamma_samples))
for i in range(len(shape)):
    shape[i] = (gamma_samples[i].sum()/gamma_stepsize[i])/len(gamma_samples[i])

σ = stats.weibull_sigma(k, λ)
normalized_step_size = numpy.sqrt(shape*gamma_stepsize**2)/σ
title = f"Weibull Distribution, Gamma Proposal, Normalized Stepsize, k={k}, λ={λ}"
acceptance_plot(title, normalized_step_size, gamma_acceptance, 20, [50.0, 10.0])
