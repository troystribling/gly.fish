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

def pdf(μ1, μ2, σ1, σ2, ρ):
    def f(x1, x2):
        y1 = (x1 - μ1) / σ1
        y2 = (x2 - μ2) / σ2
        c = 2 * numpy.pi * σ1 * σ2 * numpy.sqrt(1.0 - ρ**2)
        ε = (y1**2 + y2**2 - ρ * y1 * y2) / (2.0 * (1.- ρ**2))
        return numpy.exp(-ε) / c
    return f

def conditional_pdf_y1_y2(μ1, μ2, σ1, σ2, ρ):
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

def conditional_pdf_generator(y, μ1, μ2, σ1, σ2, ρ):
    loc = μ1 + ρ * σ1 * (y - μ2) / σ2
    scale = numpy.sqrt((1.0 - ρ**2) * σ1**2)
    return numpy.random.normal(loc, scale)

def pdf_contour(μ1, μ2, σ1, σ2, ρ):
    def f(θ, c):
        y1 = c * σ1 * (numpy.sin(θ) + numpy.cos(θ) * ρ / numpy.sqrt(1.0 - ρ**2)) + μ1
        y1 = c * σ2 * numpy.cos(θ) / numpy.sqrt(1.0 - ρ**2) + μ2
        return (y1, y2)
    return f

def pdf_transform_y1_constant(μ1, μ2, σ1, σ2, ρ, c1):
    def f(y2):
        x1 = (c1 - μ1) / σ1
        x2 = ((σ2 * ρ)(c1 - μ1) / σ1 - (y2 - μ2)) / (σ2 * numpy.sqrt(1.0 - ρ**2))
        return (x1, x2)
    return f

def pdf_transform_y2_constant(μ1, μ2, σ1, σ2, ρ, c2):
    def f(y1):
        x1 = (y1 - μ1) / σ1
        x2 = ((σ2 * ρ)(y1 - μ1) / σ1 - (c2 - μ2)) / (σ2 * numpy.sqrt(1.0 - ρ**2))
        return (x1, x2)
    return f

def pdf_contour_constant(μ1, μ2, σ1, σ2, ρ, v):
    return numpy.sqrt(2.0 * (1.0 - ρ**2) * numpy.log(numpy.pi * σ1 * σ2 * v * numpy.sqrt(1.0 - ρ**2)))

# %%

σ1 = 1.0
σ2 = 1.0
μ1 = 0.0
μ2 = 0.0
ρ = 0.5

npts = 500
x1 = numpy.linspace(-σ1*3.0, σ1*3.0, npts)
x2 = numpy.linspace(-σ2*3.0, σ2*3.0, npts)

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
contour = axis.contour(x1_grid, y1_grid, f_x1_x2, [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15], cmap=config.contour_color_map)
axis.clabel(contour, contour.levels[::2], fmt="%.3f", inline=True, fontsize=15)
config.save_post_asset(figure, "bivariate_normal_distribution", "bivariate_pdf_contours")

# %%

figure, axis = pyplot.subplots(figsize=(10, 7))
axis.set_title(f"Bivariate Normal PDF: ρ={ρ}")
axis.set_yticklabels([])
axis.set_xticklabels([])

axis_z = figure.add_subplot(111, projection='3d')
axis_z.set_ylabel('y')
axis_z.set_xlabel('x')
axis_z.set_zticks([0.0, 0.05, 0.1, 0.15])
axis_z.set_xticks([-σ1*3.0, -σ1*2.0, -σ1, 0.0, σ1, σ1*2.0, σ1*3.0])
axis_z.set_yticks([-σ2*3.0, -σ2*2.0, -σ2, 0.0, σ2, σ2*2.0, σ2*3.0])
axis_z.plot_wireframe(x1_grid, y1_grid, f_x1_x2, rstride=25, cstride=25)
config.save_post_asset(figure, "bivariate_normal_distribution", "bivariate_pdf_surface")

# %%

σ1 = 1.0
σ2 = 1.0
μ1 = 0.0
μ2 = 0.0
npts = 500
ρ = 0.0
θ = numpy.linspace(0.0, 2.0 * numpy.pi, npts)

c = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15]
for n in range(len(c)):
    f = bivariate_normal_pdf_contour(μ1, μ2, σ1, σ2, ρ)
