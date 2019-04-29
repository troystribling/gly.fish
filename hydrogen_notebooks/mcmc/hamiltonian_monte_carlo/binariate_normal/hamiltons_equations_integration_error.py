%load_ext autoreload
%autoreload 2

import numpy
from matplotlib import pyplot
from glyfish import config
from glyfish import gplot
from glyfish import hamiltonian_monte_carlo as hmc
from glyfish import stats
from glyfish import hamiltons_equations as he

%matplotlib inline

pyplot.style.use(config.glyfish_style)

# %%
# momentum verlet integrator validation

# p0 = numpy.array([-1.0, -2.0])
# q0 = numpy.array([1.0, -1.0])

p0 = numpy.array([-0.35686864, -0.88875008])
q0 = numpy.array([1.0, -1.0])

m1 = 1.0
m2 = 1.0

σ1 = 1.0
σ2 = 1.0

γ = 0.0
α = 1 / (1.0 - γ**2)

ε = 0.01
ω_plus = numpy.complex(0.0, numpy.sqrt(α*(1.0 + γ)))
ω_minus = numpy.complex(0.0, numpy.sqrt(α*(1.0 - γ)))
t_plus = 2.0*numpy.pi / numpy.abs(ω_plus)
t_minus = 2.0*numpy.pi / numpy.abs(ω_minus)

nsteps = int(2.0*t_minus/(3.0*ε))

# %%

U = he.bivariate_normal_U(γ, σ1, σ2)
K = he.bivariate_normal_K(m1, m2)
dUdq = he.bivariate_normal_dUdq(γ, σ1, σ2)
dKdp = he.bivariate_normal_dKdp(m1, m2)

phe, qhe = he.momentum_verlet(p0, q0, dUdq, dKdp, nsteps, ε)

print(f"p = {phe[-1]}")
print(f"q = {qhe[-1]}")

# %%

U = hmc.bivariate_normal_U(γ, σ1, σ2)
K = hmc.bivariate_normal_K(m1, m2)
dUdq = hmc.bivariate_normal_dUdq(γ, σ1, σ2)
dKdp = hmc.bivariate_normal_dKdp(m1, m2)

phmc, qhmc = hmc.momentum_verlet_integrator(p0, q0, dUdq, dKdp, nsteps, ε)

print(f"p = {phmc}")
print(f"q = {qhmc}")
