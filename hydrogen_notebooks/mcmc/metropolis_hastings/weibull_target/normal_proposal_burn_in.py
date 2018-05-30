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
x0 = [0.001, 1.0, 2.0, 2.5, 3.0, 3.5]

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
axis.set_xlabel(r"$X_0$")
axis.set_ylabel("Acceptance %")
axis.set_title(title)
axis.set_ylim([0.0, 100])
axis.plot(x0, acceptance, zorder=5, marker='o', color="#336699", markersize=15.0, linestyle="None", markeredgewidth=1.0, alpha=0.5, label="Simulation")

# %%

sample_idx = 0
all_samples[sample_idx]
title = f"Weibull Distribution, Normal Proposal, Accepted {format(acceptance[sample_idx], '2.0f')}%, " + r"$X_0$=" + f"{x0[sample_idx]}"
gplot.pdf_samples(title, target_pdf, all_samples[sample_idx])

# %%

sample_idx = 1
title = f"Weibull Distribution, Normal Proposal, Accepted {format(acceptance[sample_idx], '2.0f')}%, " + r"$X_0$=" + f"{x0[sample_idx]}"
gplot.pdf_samples(title, target_pdf, all_samples[sample_idx])


# %%

sample_idx = 2
title = f"Weibull Distribution, Normal Proposal, Accepted {format(acceptance[sample_idx], '2.0f')}%, " + r"$X_0$=" + f"{x0[sample_idx]}"
gplot.pdf_samples(title, target_pdf, all_samples[sample_idx])

# %%

sample_idx = 3
title = f"Weibull Distribution, Normal Proposal, Accepted {format(acceptance[sample_idx], '2.0f')}%, " + r"$X_0$=" + f"{x0[sample_idx]}"
gplot.pdf_samples(title, target_pdf, all_samples[sample_idx])

# %%

sample_idx = 4
title = f"Weibull Distribution, Normal Proposal, Accepted {format(acceptance[sample_idx], '2.0f')}%, " + r"$X_0$=" + f"{x0[sample_idx]}"
gplot.pdf_samples(title, target_pdf, all_samples[sample_idx])

# %%

title = r"Weibull Distribution Samples, Normal Proposal, $X_0$ comparison"
time = range(51000, 51500)

nplots = len(all_samples)
figure, axis = pyplot.subplots(nrows=nplots, ncols=1, sharex=True, figsize=(15, 3*nplots))
axis[0].set_title(title)
axis[-1].set_xlabel("Time")
bbox = dict(boxstyle='square,pad=1', facecolor="#FFFFFF", edgecolor="lightgrey")
for i in range(nplots):
    axis[i].set_xlim([time[0], time[-1] + 1])
    axis[i].set_ylim([0.0, 2.0])
    axis[i].plot(time, all_samples[i][time], lw="1")
    axis[i].text(51010.0, 1.6, r"$X_0$=" + f"{format(x0[i], '2.0f')}", fontsize=13, bbox=bbox)

# %%

μ = stats.weibull_mean(k, λ)
title = r"Weibull Distribution, Normal Proposal, sample μ convergence, $X_0$ comparison"
time = range(nsample)
nplot = len(all_samples)

figure, axis = pyplot.subplots(figsize=(12, 6))
axis.set_xlabel("Time")
axis.set_ylabel(r"$μ$")
axis.set_title(title)
axis.set_xlim([1.0, nsample])
axis.set_ylim([0.0, 4.0])
axis.semilogx(time, numpy.full(nsample, μ), label="Target μ", color="#000000")
for i in range(nplot):
    axis.semilogx(time, stats.cummean(all_samples[i]), label=r"$X_0$="+f"{format(x0[i], '2.2f')}")
axis.legend()

# %%

σ = stats.weibull_sigma(k, λ)
title = r"Weibull Distribution, Normal Proposal, sample σ convergence, $X_0$ comparison"
time = range(nsample)
nplot = len(all_samples)

figure, axis = pyplot.subplots(figsize=(12, 6))
axis.set_xlabel("Time")
axis.set_ylabel(r"$σ$")
axis.set_title(title)
axis.set_xlim([1.0, nsample])
axis.set_ylim([0.0, 1.0])
axis.semilogx(time, numpy.full(nsample, σ), label="Target σ", color="#000000")
for i in range(nplot):
    axis.semilogx(time, stats.cumsigma(all_samples[i]), label=r"$X_0$="+f"{format(x0[i], '2.2f')}")
axis.legend()

# %%

sample_idx = [0, 5, 9, 11]
title = title = r"Weibull Distribution, Normal Proposal, Autocorrelation, $X_0$ comparison"
auto_core_range = range(20000, 50000)
nlag = 100
nplot = len(all_samples)

figure, axis = pyplot.subplots(figsize=(12, 9))
axis.set_title(title)
axis.set_xlabel("Time Lag")
axis.set_xlim([0, nlag])
for i in range(nplot):
    ac = stats.autocorrelate(all_samples[i][auto_core_range])
    axis.plot(range(nlag), numpy.real(ac[:nlag]), label=r"$X_0$="+f"{format(x0[i], '2.2f')}")
axis.legend()
