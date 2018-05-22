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

def convolve_sum(x, y):
    n = len(y)
    convolution_sum = numpy.zeros(len(x))
    for t in range(n):
        for k in range(0,t+1):
            convolution_sum[t] += x[k] * y[t-k]
    return convolution_sum

def cross_correlate(x, y):
    n = len(x)
    x_padded = numpy.concatenate((x, numpy.zeros(n-1)))
    y_padded = numpy.concatenate((y, numpy.zeros(n-1)))
    x_fft = numpy.fft.fft(x_padded)
    y_fft = numpy.fft.fft(y_padded)
    h_fft = numpy.conj(x_fft) * y_fft
    cc = numpy.fft.ifft(h_fft)
    return cc[0:n]

def cross_correlate_sum(x, y):
    n = len(x)
    correlation = numpy.zeros(len(x))
    for t in range(n):
        for k in range(0, n - t):
            correlation[t] += x[k] * y[k + t]
    return correlation

# %%
# Example vectors

f = numpy.array([8, 4, 8, 0])
g = numpy.array([6, 3, 9, 3])
F = numpy.array([20.0, -4.0j, 12.0, 4.0j])
G = numpy.array([21, -3, 9, -3])

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

# Convolution

# %%
# f and g are not zero padded
f_fft = numpy.fft.fft(f)
g_fft = numpy.fft.fft(g)
h_fft = f_fft * g_fft
h = numpy.fft.ifft(h_fft)
h

# %%
# f and g are zero padded
f_padded = numpy.concatenate((f, numpy.zeros(len(f)-1)))
g_padded = numpy.concatenate((g, numpy.zeros(len(g)-1)))
f_fft = numpy.fft.fft(f_padded)
g_fft = numpy.fft.fft(g_padded)
h_fft = f_fft * g_fft
h = numpy.fft.ifft(h_fft)
h

# %%
# compare convolution theorem result with direct calculation of sums
convolve(f, g)
convolve_sum(f, g)

# compare with numpy.convolve
numpy.convolve(f, g, 'full')
numpy.convolve(f, g, 'same')
numpy.convolve(f, g, 'valid')

# Cross Correlation

# %%
# f and g are not zero padded
f_fft = numpy.fft.fft(f)
g_fft = numpy.fft.fft(g)
cc_fft = numpy.conj(f_fft) * g_fft
cc = numpy.fft.ifft(cc_fft)
cc

# %%
# f and g are zero padded
f_padded = numpy.concatenate((f, numpy.zeros(len(f)-1)))
g_padded = numpy.concatenate((g, numpy.zeros(len(g)-1)))
f_fft = numpy.fft.fft(f_padded)
g_fft = numpy.fft.fft(g_padded)
cc_fft = numpy.conj(f_fft) * g_fft
cc = numpy.fft.ifft(cc_fft)
cc

# %%
# compare convolution theorem result with direct calculation of sums
cross_correlate(f, g)
cross_correlate_sum(f, g)

# compare with numpy.convolve
numpy.correlate(f, g, 'full')
numpy.correlate(g, f, 'full')
numpy.correlate(f, g, 'same')
numpy.correlate(g, f, 'same')
numpy.correlate(f, g, 'valid')
