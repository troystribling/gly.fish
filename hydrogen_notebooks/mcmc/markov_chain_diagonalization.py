# %%
%load_ext autoreload
%autoreload 2

import numpy

# %%

t = [[0.0, 0.9, 0.1, 0.0],
     [0.8, 0.1, 0.0, 0.1],
     [0.0, 0.5, 0.3, 0.2],
     [0.1, 0.0, 0.0, 0.9]]
p = numpy.matrix(t)

# %%

λ, v = numpy.linalg.eig(p)
λ
v
Λ = numpy.diag(λ)
V = numpy.matrix(v)
Vinv = numpy.linalg.inv(V)

Λ_t = Λ**100
V * Λ_t

V * Λ_t * Vinv

p**100

# %%
Vinv[2]/numpy.sum(Vinv[2])
