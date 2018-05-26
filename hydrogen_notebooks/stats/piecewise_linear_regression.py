# %%
%load_ext autoreload
%autoreload 2

import numpy
import scipy
from matplotlib import pyplot
from glyfish import config
from glyfish import stats

pyplot.style.use(config.glyfish_style)

# %%

x = numpy.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,11, 12, 13, 14, 15], dtype=float)
y = numpy.array([5, 7, 9, 11, 13, 15, 28.92, 42.81, 56.7, 70.59, 84.47, 98.36, 112.25, 126.14, 140.03])

def piecewise_linear(x, x0, y0, k1, k2):
    l1 = lambda x:y0 + k1*(x - x0)
    l2 = lambda x:y0 + k2*(x - x0)
    return numpy.piecewise(x, [x < x0], [l1, l2])

params, error = scipy.optimize.curve_fit(piecewise_linear, x, y)
xfit = numpy.linspace(0, 15, 100)
numpy.sqrt(numpy.diag(error))
params
pyplot.plot(x, y, "o")
pyplot.plot(xfit, piecewise_linear(xfit, *params))
