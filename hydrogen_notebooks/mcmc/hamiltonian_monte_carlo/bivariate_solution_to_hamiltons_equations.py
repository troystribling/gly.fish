# %%
%load_ext autoreload
%autoreload 2

import numpy
from numpy import linalg

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

γ = 0.9 # correlation coefficient
pq0 = numpy.matrix([[1.0], [1.0], [1.0], [1.0]]) # initial conditions

ω_plus = numpy.complex(0.0, numpy.sqrt(α*(1.0 + γ)))
ω_minus = numpy.complex(0.0, numpy.sqrt(α*(1.0 - γ)))

tmax = 4.0*numpy.pi / numpy.abs(ω_plus)
nsteps = 500

α = 1 / (1.0 - γ**2)
time = numpy.linspace(0.0, tmax, nsteps)

# %%
# Compute eigenvalues form hamiltonian_matrix

# compute eigenvalues and eigenvectors
H = hamiltonian_matrix(γ, α)
λ, E = linalg.eig(H)

# compute coeficients from initial conditions
Einv = linalg.inv(E)
C = Einv * pq0

# %%
# verify algebraic calculations

λ = eigenvalues(γ, α)
E = eigenvector_matrix(γ, α)

# compute coeficients from initial conditions
Einv = linalg.inv(E)
C = Einv * pq0
