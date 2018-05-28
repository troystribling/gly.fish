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

def acceptance_plot(title, x, y, idx, text_pos):
    xlim = [0.0001, 1000.0]
    slope, y0, r2 = acceptance_regress(x[idx:], y[idx:])
    xfit = numpy.linspace(-1.0, 3.0, 100)
    bbox = dict(boxstyle='square,pad=1', facecolor="#FFFFFF", edgecolor="lightgrey")
    figure, axis = pyplot.subplots(figsize=(12, 5))
    axis.set_xlabel("Step Size")
    axis.set_ylabel("Acceptance %")
    axis.set_title(title)
    axis.set_xlim(xlim)
    axis.set_ylim([0.1, 200.0])
    axis.loglog(x, y, zorder=6, marker='o', color="#336699", markersize=15.0, linestyle="None", markeredgewidth=1.0, alpha=0.5, label="Simulation")
    axis.loglog(10**xfit, acceptance_fit(xfit, slope, y0), zorder=5, color="#A60628", label="Fit")
    axis.loglog(numpy.linspace(xlim[0], xlim[1], 100), numpy.full((100), 100.0), zorder=5, color="#A60628")
    axis.text(text_pos[0], text_pos[1], f"slope={format(slope, '2.2f')}, "+r"$R^2$="+f"{format(r2, '2.2f')}", fontsize=14, bbox=bbox)
    axis.legend()

def acceptance_regress(x, y):
    slope, y0, r_value, _, _ = scipy.stats.linregress(numpy.log10(x), numpy.log10(y))
    return slope, y0, r_value**2

def acceptance_fit(x, slope, y0):
    return 10**(slope*x + y0)

def run_sumulation(stepsize, target_pdf, proposal, generator, nsample, x0):
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

x = numpy.linspace(0.001, 2.0, 500)
figure, axis = pyplot.subplots(figsize=(12, 5))
axis.set_xlabel("X")
axis.set_ylabel("PDF")
axis.set_xlim([0.0, x[-1]])
axis.set_title(f"Weibull Distribution, k={k}, λ={λ}")
axis.plot(x, [target_pdf(j) for j in x])

# %%
# weibull target, normal proposal

nsample = 100000
x0 = 1.0
npts = 25
stepsize = 10**numpy.linspace(-3.0, 2.0, npts)
normal_samples, normal_acceptance = run_sumulation(stepsize, target_pdf, mh.normal_proposal, mh.normal_generator, nsample, x0)

# %%

σ = stats.weibull_sigma(k, λ)
normalized_step_size = stepsize/σ
title = f"Weibull Distribution, Normal Proposal, Normalized Stepsize, k={k}, λ={λ}"
acceptance_plot(title, normalized_step_size, normal_acceptance, 13, [10.0, 10.0])

# %%
# weibull target, gamma proposal

nsample = 100000
x0 = 1.0
npts = 25
stepsize = 10**numpy.linspace(-5.0, 1.0, npts)
gamma_samples, gamma_acceptance = run_sumulation(stepsize, target_pdf, mh.gamma_proposal, mh.gamma_generator, nsample, x0)

# %%

shape = numpy.zeros(len(gamma_samples))
for i in range(len(shape)):
    shape[i] = (gamma_samples[i].sum()/stepsize[i])/len(gamma_samples[i])

σ = stats.weibull_sigma(k, λ)
normalized_step_size = numpy.sqrt(shape*stepsize**2)/σ
title = f"Weibull Distribution, Gamma Proposal, Normalized Stepsize, k={k}, λ={λ}"
acceptance_plot(title, normalized_step_size, gamma_acceptance, 20, [50.0, 10.0])

# %%
# weibull target, normal independence proposal

nsample = 100000
x0 = 1.0
npts = 25
μ = 1.0
stepsize = 10**numpy.linspace(-3.0, 2, npts)

normal_independence_samples, normal_independence_acceptance = run_sumulation(stepsize, target_pdf, mh.normal_proposal, mh.normal_independence_generator(μ), nsample, x0)

# %%

σ = stats.weibull_sigma(k, λ)
normalized_step_size = stepsize/σ
title = f"Weibull Distribution, Normal Independent Proposal, Normalized Stepsize, k={k}, λ={λ}"
acceptance_plot(title, normalized_step_size, normal_acceptance, 13, [10.0, 10.0])


# %%
# arcsine target, Normal proposal

target_pdf = stats.arcsine
nsample = 100000
stepsize = 10**numpy.linspace(-3.0, 2, npts)
x0 = 0.5
npts = 25
arcsine_samples, arcsine_acceptance = run_sumulation(stepsize, target_pdf, mh.normal_proposal, mh.normal_generator, nsample, x0)

# %%

σ = numpy.sqrt(1.0/8.0)
normalized_step_size = stepsize/σ
title = f"Arcsine Distribution, Normal Proposal, Normalized Stepsize, k={k}, λ={λ}"
acceptance_plot(title, normalized_step_size, arcsine_acceptance, 20, [50.0, 10.0])

# %%
# bimodal normal target, Normal proposal

target_pdf = stats.bimodal_normal
nsample = 100000
x0 = 0.5
npts = 25
stepsize = 10**numpy.linspace(-3.0, 2, npts)
arcsine_samples, arcsine_acceptance = run_sumulation(stepsize, target_pdf, mh.normal_proposal, mh.normal_generator, nsample, x0)

# %%

σ = stats.bimodal_normal_sigma()
normalized_step_size = stepsize/σ
title = f"Bimodal Normal Distribution, Normal Proposal, Normalized Stepsize"
acceptance_plot(title, normalized_step_size, arcsine_acceptance, 20, [50.0, 10.0])
