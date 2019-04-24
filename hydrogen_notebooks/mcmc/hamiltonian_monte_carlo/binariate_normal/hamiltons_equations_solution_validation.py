# %%
%load_ext autoreload
%autoreload 2

import numpy
from numpy import linalg
from matplotlib import pyplot
from glyfish import config
from glyfish import hamiltonian_monte_carlo as hmc

%matplotlib inline

pyplot.style.use(config.glyfish_style)

# %%

def hamiltonian_matrix(γ, α):
    m = [[0.0, 0.0, -α, α*γ],
         [0.0, 0.0, α*γ, -α],
         [1.0, 0.0, 0.0, 0.0],
         [0.0, 1.0, 0.0, 0.0]]
    return numpy.matrix(m)

def eigenvector_matrix(γ, α):
    ω_plus = numpy.complex(0.0, numpy.sqrt(α*(1.0 + γ)))
    ω_minus = numpy.complex(0.0, numpy.sqrt(α*(1.0 - γ)))
    m = [[ω_plus, numpy.conj(ω_plus), ω_minus, numpy.conj(ω_minus)],
         [numpy.conj(ω_plus), ω_plus, ω_minus, numpy.conj(ω_minus)],
         [1.0, 1.0, 1.0, 1.0],
         [-1.0, -1.0, 1.0, 1.0]]
    m = numpy.matrix(m)
    _, col = m.shape
    for i in range(0, col):
        m[:,i] = m[:,i] / linalg.norm(m[:,i])
    return m

def eigenvalues(γ, α):
    ω_plus = numpy.complex(0.0, numpy.sqrt(α*(1.0 + γ)))
    ω_minus = numpy.complex(0.0, numpy.sqrt(α*(1.0 - γ)))
    return [ω_plus, numpy.conj(ω_plus), ω_minus, numpy.conj(ω_minus)]

# %%
# Configuration

γ = 0.9
α = 1 / (1.0 - γ**2)

ω_plus = numpy.complex(0.0, numpy.sqrt(α*(1.0 + γ)))
ω_minus = numpy.complex(0.0, numpy.sqrt(α*(1.0 - γ)))
t_plus = 2.0*numpy.pi / numpy.abs(ω_plus)
t_minus = 2.0*numpy.pi / numpy.abs(ω_minus)

U = hmc.bivariate_normal_U(γ, 1.0, 1.0)
K = hmc.bivariate_normal_K(1.0, 1.0)

# %%
# Compute solutions using eigenvalues and eigenvectots computed from numercically diagonalizing Hamiltonian Matrix
PQ0 = numpy.matrix([[1.0], [1.0], [1.0], [1.0]])
λ = eigenvalues(γ, α)
E = eigenvector_matrix(γ, α)
Einv = linalg.inv(E)
Einv * PQ0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
