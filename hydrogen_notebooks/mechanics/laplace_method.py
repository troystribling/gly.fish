# %%
%load_ext autoreload
%autoreload 2

import numpy
from matplotlib import pyplot
from glyfish import config
from glyfish import stats
from scipy import fftpack

%matplotlib inline

pyplot.style.use(config.glyfish_style)

# %%

npts = 200
def sinc(x):
    return numpy.sinc(x/numpy.pi)

def limit_sinc(m, x):
    σ = 3.0/(m)
    return numpy.exp(m)*numpy.exp(-x**2/(2.0*σ))

def approximation_plot(m):
    figure, axis = pyplot.subplots(figsize=(10, 7))
    axis.set_title(r"Laplace's Method Approximation to $f(x)=e^{M\sin(x)/x}$, " + f"M={m}")
    axis.set_xlabel(r"$x$")
    axis.set_ylabel(r"$f(x)$")
    x = numpy.linspace(-10.0, 10.0, npts)
    f = numpy.exp(m*sinc(x))
    axis.plot(x, f, label=r"$f(x)$")
    axis.plot(x, limit_sinc(m, x), label=f"Approximation")
    axis.legend(bbox_to_anchor=(0.7, 0.85))


# %%

approximation_plot(1.0)

# %%

approximation_plot(2.0)

# %%

approximation_plot(5.0)
