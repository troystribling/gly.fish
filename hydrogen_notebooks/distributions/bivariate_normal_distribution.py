# %%
%load_ext autoreload
%autoreload 2

import numpy
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
from glyfish import config

%matplotlib inline

pyplot.style.use(config.glyfish_style)

# %%

def bivariate_normal_pdf(μ1, μ2, σ1, σ2, ρ):
    def f(x1, x2):
        y1 = (x1 - μ1) / σ1
        y2 = (x2 - μ2) / σ2
        c = 2 * numpy.pi * σ1 * σ2 * numpy.sqrt(1.0 - ρ**2)
        ε = (y1**2 + y2**2 - ρ * y1 * y2) / (2.0 * (1.- ρ**2))
        return numpy.exp(-ε) / c
    return f

def bivariate_normal_pdf_iso(μ1, μ2, σ1, σ2, ρ):
    def f(x1, x2):
        y1 = (x1 - μ1) / σ1
        y2 = (x2 - μ2) / σ2
        return (y1**2 + y2**2 - ρ * y1 * y2) / (2.0 * (1.- ρ**2))
    return f

def bivariate_normal_conditional_pdf(μ1, μ2, σ1, σ2, ρ):
    def f(x1, x2):
        y1 = (x1 - μ1)
        y2 = (x2 - μ2)
        c = 2 * numpy.pi * σ1 * numpy.sqrt(1.0 - ρ**2)
        ε = (y1**2 - ρ * σ1 * y2 / σ2) / (2.0 * (1.- ρ**2))
        return numpy.exp(ε) / c
    return f

def normal(x, σ=1.0, μ=0.0):
    ε = (x - μ)**2/(2.0*σ**2)
    return numpy.exp(-ε)/numpy.sqrt(2.0*numpy.pi*σ**2)

def bivariate_normal_conditional_pdf_generator(y, μ1, μ2, σ1, σ2, ρ):
    loc = μ1 + ρ * σ1 * (y - μ2) / σ2
    scale = numpy.sqrt((1.0 - ρ**2) * σ1**2)
    return numpy.random.normal(loc, scale)

# %%

σ1 = 1.0
σ2 = 1.0
μ1 = 0.0
μ2 = 0.0
ρ = 0.5

npts = 500
x1 = numpy.linspace(-σ1*2.0, σ1*2.0, npts)
x2 = numpy.linspace(-σ2*2.0, σ2*2.0, npts)

# %%

pdf = bivariate_normal_pdf(μ1, μ2, σ1, σ2, ρ)
x1_grid, y1_grid = numpy.meshgrid(x1, x2)
f_x1_x2 = numpy.zeros((npts, npts))
for i in numpy.arange(npts):
    for j in numpy.arange(npts):
        f_x1_x2[i, j] = pdf(x1_grid[i,j], y1_grid[i,j])

# %%

figure, axis = pyplot.subplots(figsize=(10, 7))
axis.set_xlabel("x")
axis.set_ylabel("y")
axis.set_title(f"Bivariate Normal PDF: ρ={ρ}")
contour = axis.contour(x1_grid, y1_grid, f_x1_x2, cmap=config.contour_color_map)
axis.clabel(contour, contour.levels[::2], fmt="%.2f", inline=True, fontsize=15)
config.save_post_asset(figure, "bivariate_normal_distribution", "bivariate_pdf_contours")


# %%

figure, axis = pyplot.subplots(figsize=(10, 7))
axis_z = figure.add_subplot(111, projection='3d')
axis_z.plot_wireframe(x1_grid, y1_grid, f_x1_x2, rstride=20, cstride=20)
config.save_post_asset(figure, "bivariate_normal_distribution", "bivariate_pdf_surface")
