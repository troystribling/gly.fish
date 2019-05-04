# %%
%load_ext autoreload
%autoreload 2

import numpy
from matplotlib import pyplot
from glyfish import metropolis_hastings as mh
from glyfish import bivariate_normal_distribution as bvd
from glyfish import hamiltonian_monte_carlo as hmc
from glyfish import config
from glyfish import gplot
from glyfish import stats

%matplotlib inline

pyplot.style.use(config.glyfish_style)

# %%

nsample = 10000
x0 = numpy.array([1.0, -1.0])

stepsize = 1.0

μ1 = 0.0
μ2 = 0.0
σ1 =  1.0
σ2 = 1.0
γ = 0.9

target_pdf = bvd.metropolis_hastings_target_pdf(μ1, μ2, σ1, σ2, γ)

q1range = [-3.1*σ1, 3.1*σ1]
q2range = [-3.1*σ2, 3.1*σ2]

file_prefix = "mh-bivariate-normal-γ-0.9"

# %%

bvd.contour_plot(μ1, μ2, σ1, σ2, γ, [0.01, 0.1, 0.2, 0.3], "hamiltonian_monte_carlo", "metropolis-hastings-bivariate-normal-target-pdf")

# %%

q, accepted = mh.component_metropolis_hastings(target_pdf, mh.normal_proposal, mh.normal_generator, stepsize, nsample=nsample, x0=x0)

# %%

title = f"MH BivariateNormal" + f": γ={γ}, nsample={nsample}, accepted={int(100.0*float(accepted)/float(nsample))}%"
hmc.pdf_samples_contour(bvd.pdf(μ1, μ2, σ1, σ2, γ), q[:,0], q[:,1], q1range, q2range, [0.01, 0.05, 0.15, 0.2, 0.25, 0.3, 0.35], [r"$q_1$", r"$q_2$"], title, f"{file_prefix}-samples-contour")
