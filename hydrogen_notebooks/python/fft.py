# %%
%load_ext autoreload
%autoreload 2

import numpy
from matplotlib import pyplot
from glyfish import config
from scipy import fftpack

pyplot.style.use(config.glyfish_style)

# %%
# FFT example

f = numpy.array([8, 4, 8, 0])
g = numpy.array([6, 3, 9, 3])
F = numpy.array([20.0, -4.0j, 12.0, 4.0j])
G = numpy.array([21, -3, 9, -3])
x = numpy.array([0, 1, 2, 3])

# %%

figure, axis = pyplot.subplots(figsize=(6, 5))
axis.set_xlabel("Time")
axis.set_ylabel("Value")
axis.set_xlim([x[0]-0.5, x[-1]+0.5])
axis.set_xticks(x)
axis.bar(x, f, 1.0, color="#348ABD", alpha=0.6, edgecolor="#348ABD", zorder=5)

# %%
# test FFT

f_fft = fftpack.fft(f)
f_fft == F
f_ifft = fftpack.ifft(f_fft)
f_ifft == f

g_fft = fftpack.fft(g)
g_fft == G
g_ifft = fftpack.ifft(g_fft)
g_ifft == g


# %%
