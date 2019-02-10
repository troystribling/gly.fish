# %%
%load_ext autoreload
%autoreload 2

import numpy
from matplotlib import pyplot
from glyfish import config
from glyfish import stats
from scipy import special

%matplotlib inline

pyplot.style.use(config.glyfish_style)

# %%

def volume(n, r):
    return numpy.pi**(n/2.0)*r**n / special.gamma(n/2.0 + 1.0)

def plot_volume(npts, r):
    figure, axis = pyplot.subplots(figsize=(10, 7))
    axis.set_title(f"Hypersphere Volume for fixed R={r}")
    axis.set_xlabel(r"Number of Dimensions")
    axis.set_ylabel("Volume")
    dims = numpy.arange(npts, dtype=float)[1:npts]
    v = [volume(n, r) for n in dims]
    axis.plot(dims, v)

# %%

npts = 26
r = 1.0
plot_volume(npts, r)

# %%

npts = 11
r = 0.5
plot_volume(npts, r)

# %%

npts = 51
r = 2.0
plot_volume(npts, r)

# %%

npts = 101
r = 3.0
plot_volume(npts, r)
