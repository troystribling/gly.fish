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

npts = 100
def f1(x):
    return numpy.sinc(x/numpy.pi)

figure, axis = pyplot.subplots(figsize=(10, 7))
axis.set_title(r"$f(x)=e^{M\sin(x)/x}$")
axis.set_xlabel(r"$x$")
axis.set_ylabel(r"$f(x)$")

m_values = [1.0, 2.0]
x = numpy.linspace(-10.0, 10.0, npts)

for m in m_values:
    f = numpy.exp(m*f1(x))
    axis.plot(x, f, label=f"M={m}")
axis.legend(bbox_to_anchor=(1.0, 0.85))
