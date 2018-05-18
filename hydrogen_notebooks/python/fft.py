# %%
%load_ext autoreload
%autoreload 2

import numpy
from matplotlib import pyplot
from glyfish import config
from scipy import fftpack

pyplot.style.use(config.glyfish_style)

# %%

def convolve(x, y):
    n = len(x)
    x_padded = numpy.concatenate((x, numpy.zeros(n-1)))
    y_padded = numpy.concatenate((y, numpy.zeros(n-1)))
    x_fft = numpy.fft.fft(x_padded)
    y_fft = numpy.fft.fft(y_padded)
    h_fft = x_fft * y_fft
    h = numpy.fft.ifft(h_fft)
    return h[0:n]

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
# test scipy FFT

f_fft = fftpack.fft(f)
f_fft == F
f_ifft = fftpack.ifft(f_fft)
f_ifft == f

g_fft = fftpack.fft(g)
g_fft == G
g_ifft = fftpack.ifft(g_fft)
g_ifft == g

# %%
# test numpy FFT

f_fft = numpy.fft.fft(f)
f_fft == F
f_ifft = numpy.fft.ifft(f_fft)
f_ifft == f

g_fft = numpy.fft.fft(g)
g_fft == G
g_ifft = numpy.fft.ifft(g_fft)
g_ifft == g

# %%
# Test convolution

f_padded = numpy.concatenate((f, numpy.zeros(len(f)-1)))
g_padded = numpy.concatenate((g, numpy.zeros(len(g)-1)))
f_fft = numpy.fft.fft(f_padded)
g_fft = numpy.fft.fft(g_padded)
h_fft = f_fft * g_fft
h = numpy.fft.ifft(h_fft)

convolve(f, g)

numpy.convolve(f, g, 'full')
numpy.convolve(f, g, 'same')
numpy.convolve(f, g, 'valid')

# %%
