# %%

%load_ext autoreload
%autoreload 2

import numpy

from matplotlib import pyplot
import scipy
from glyfish import config
from glyfish import stats

%matplotlib inline

pyplot.style.use(config.glyfish_style)

# %%

def f_pdf(xrange, yrange, npts):
    f_xy = lambda x, y: x**2*numpy.exp(-x*y**2 - y**2 + 2.0*y - 4.0*x)

    x = numpy.linspace(xrange[0], xrange[1], npts)
    y = numpy.linspace(yrange[0], yrange[1], npts)

    x_grid, y_grid = numpy.meshgrid(x, y)
    f = numpy.zeros((npts, npts))
    for i in numpy.arange(npts):
        for j in numpy.arange(npts):
            f[i, j] = f_xy(x_grid[i,j], y_grid[i,j])

    dx = (xrange[1] - xrange[0])/npts
    dy = (yrange[1] - yrange[0])/npts

    return f/(dx*dy*numpy.sum(f)), x_grid, y_grid

def f_xy_pdf(x,y):
    γ = 3.0
    β = y**2 + 4
    return scipy.stats.gamma.pdf(x, γ, scale=1.0/β)

def f_yx_pdf(x, y):
    μ = 1.0/(1.0 + x)
    σ = 1.0/numpy.sqrt(2.0*(1.0 + x))
    return scipy.stats.norm.pdf(x, loc=μ, scale=σ)

def f_xy_sample(y):
    γ = 3.0
    β = y**2 + 4
    return scipy.stats.gamma.rvs(γ, scale=1.0/β)

def f_yx_sample(x):
    μ = 1.0/(1.0 + x)
    σ = 1.0/numpy.sqrt(2.0*(1.0 + x))
    return scipy.stats.norm.rvs(loc=μ, scale=σ)

def gibbs_sample(nsample, x0, y0):
    samples = numpy.zeros((2*nsample+1, 2))
    samples[0, 0] = x0
    samples[0, 1] = y0
    for i in range(1, 2*nsample, 2):
        samples[i, 0] = f_xy_sample(samples[i-1, 1])
        samples[i, 1] = samples[i-1, 1]
        samples[i+1, 1] = f_yx_sample(samples[i, 0])
        samples[i+1, 0] = samples[i, 0]

    return samples

# %%

xrange = [0.0, 3.0]
yrange = [-0.75, 2.25]
npts = 500
nsample = 100000
pdf, x, y = f_pdf(xrange, yrange, npts)
x0 = 1.0
y0 = 2.0

# %%

figure, axis = pyplot.subplots(figsize=(9, 9))
axis.set_xlabel("x")
axis.set_ylabel("y")
axis.set_title("PDF Contours")
contour = axis.contour(x, y, pdf, cmap=pyplot.cm.tab10, linewidths=2)
axis.clabel(contour, contour.levels[::2], fmt="%.1f", inline=True, fontsize=15)

# %%

samples = gibbs_sample(nsample, x0, y0)
bins = [numpy.linspace(xrange[0], xrange[1], 100), numpy.linspace(yrange[0], yrange[1], 100)]

# %%

figure, axis = pyplot.subplots(figsize=(11, 9))
axis.set_xlabel("x")
axis.set_ylabel("y")
axis.set_xlim(xrange)
axis.set_title("PDF Samples")
hist, _, _, image = axis.hist2d(samples[:,0], samples[:,1], normed=True, bins=bins, cmap=pyplot.cm.Blues)
contour = axis.contour(x, y, pdf, cmap=pyplot.cm.gray, linewidths=2)
axis.clabel(contour, contour.levels[::2], fmt="%.1f", inline=True, fontsize=15)
figure.colorbar(image)

# %%

time_range = [0, 30]
figure, axis = pyplot.subplots(figsize=(9, 9))
axis.set_xlabel("x")
axis.set_ylabel("y")
axis.set_title(f"PDF Sampled Markov Chain, Steps {time_range[0]} to {time_range[1]}")
contour = axis.contour(x, y, pdf, cmap=pyplot.cm.tab10, linewidths=2)
axis.clabel(contour, contour.levels[::2], fmt="%.1f", inline=True, fontsize=15)
axis.plot(samples[time_range[0]:time_range[1],0], samples[time_range[0]:time_range[1],1], lw=1, alpha=0.65, color="#000000")
axis.plot(samples[time_range[0], 0], samples[time_range[0], 1], marker='o', color="r", markersize=13.0)

# %%

time_range = [20000, 20030]
figure, axis = pyplot.subplots(figsize=(9, 9))
axis.set_xlabel("x")
axis.set_ylabel("y")
axis.set_title(f"PDF Sampled Markov Chain, Steps {time_range[0]} to {time_range[1]}")
contour = axis.contour(x, y, pdf, cmap=pyplot.cm.tab10, linewidths=2)
axis.clabel(contour, contour.levels[::2], fmt="%.1f", inline=True, fontsize=15)
axis.plot(samples[time_range[0]:time_range[1],0], samples[time_range[0]:time_range[1],1], lw=1, alpha=0.75, color="#000000")
axis.plot(samples[time_range[0], 0], samples[time_range[0],1], marker='o', color="r", markersize=13.0)
