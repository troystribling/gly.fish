import numpy
from scipy import stats
from scipy import special
from matplotlib import pyplot
from glyfish import config

def metropolis_hastings_target_pdf(μ1, μ2, σ1, σ2, γ):
    def f(x, i, x_current):
        if i == 0:
            y1 = (x - μ1) / σ1
            y2 = (x_current[1] - μ2) / σ2
        else:
            y1 = (x_current[0] - μ1) / σ1
            y2 = (x - μ2) / σ2
        c = 2 * numpy.pi * σ1 * σ2 * numpy.sqrt(1.0 - γ**2)
        ε = (y1**2 + y2**2 - 2.0 * γ * y1 * y2) / (2.0 * (1.0 - γ**2))
        return numpy.exp(-ε) / c
    return f

def marginal(μ, σ):
    def f(x):
        y = (x - μ) / σ
        return numpy.exp(-y**2/2.0)/numpy.sqrt(2.0*numpy.pi*σ**2.0)
    return f

def pdf(μ1, μ2, σ1, σ2, γ):
    def f(x):
        y1 = (x[0] - μ1) / σ1
        y2 = (x[1] - μ2) / σ2
        c = 2 * numpy.pi * σ1 * σ2 * numpy.sqrt(1.0 - γ**2)
        ε = (y1**2 + y2**2 - 2.0 * γ * y1 * y2) / (2.0 * (1.0 - γ**2))
        return numpy.exp(-ε) / c
    return f

def conditional_pdf_y1_y2(μ1, μ2, σ1, σ2, γ):
    def f(x1, x2):
        y1 = (x1 - μ1)
        y2 = (x2 - μ2)
        c = numpy.sqrt(2 * numpy.pi * σ1 * (1.0 - γ**2))
        ε = (y1 - γ * σ1 * y2 / σ2)**2 / (2.0 * σ1**2 * (1.0 - γ**2))
        return numpy.exp(-ε) / c
    return f

def conditional_pdf_xy_generator(μ1, μ2, σ1, σ2, γ):
    def f(y):
        loc = μ1 + γ * σ1 * (y - μ2) / σ2
        scale = numpy.sqrt((1.0 - γ**2) * σ1**2)
        return numpy.random.normal(loc, scale)
    return f

def conditional_pdf_yx_generator(μ1, μ2, σ1, σ2, γ):
    def f(y):
        loc = μ2 + γ * σ2 * (y - μ1) / σ1
        scale = numpy.sqrt((1.0 - γ**2) * σ1**2)
        return numpy.random.normal(loc, scale)
    return f

def max_pdf_value(σ1, σ2, γ):
    return 1.0/(2.0 * numpy.pi * σ1 * σ2 * numopy.sqrt(1.0 - γ**2))

def normal(x, σ=1.0, μ=0.0):
    ε = (x - μ)**2/(2.0*σ**2)
    return numpy.exp(-ε)/numpy.sqrt(2.0*numpy.pi*σ**2)

def pdf_mesh(μ1, μ2, σ1, σ2, γ):
    npts = 500
    if (σ1 > σ2):
        x1 = numpy.linspace(-σ1*3.0, σ1*3.0, npts)
        x2 = numpy.linspace(-σ1*3.0, σ1*3.0, npts)
    elif (σ2 > σ1):
        x1 = numpy.linspace(-σ2*3.0, σ2*3.0, npts)
        x2 = numpy.linspace(-σ2*3.0, σ2*3.0, npts)
    else:
        x1 = numpy.linspace(-σ1*3.0, σ1*3.0, npts)
        x2 = numpy.linspace(-σ2*3.0, σ2*3.0, npts)
    f = pdf(μ1, μ2, σ1, σ2, γ)
    x1_grid, x2_grid = numpy.meshgrid(x1, x2)
    f_x1_x2 = numpy.zeros((npts, npts))
    for i in numpy.arange(npts):
        for j in numpy.arange(npts):
            f_x1_x2[i, j] = f([x1_grid[i,j], x2_grid[i,j]])
    return (x1_grid, x2_grid, f_x1_x2)

def contour_plot(μ1, μ2, σ1, σ2, γ, contour_values, post, plot_name):
    npts = 500
    x1_grid, x2_grid, f_x1_x2 = pdf_mesh(μ1, μ2, σ1, σ2, γ)
    figure, axis = pyplot.subplots(figsize=(8, 8))
    axis.set_xlabel(r"$u$")
    axis.set_ylabel(r"$v$")
    if (σ1 > σ2):
        axis.set_xlim([-3.2*σ1, 3.2*σ1])
        axis.set_ylim([-3.2*σ1, 3.2*σ1])
    elif (σ2 > σ1):
        axis.set_xlim([-3.2*σ2, 3.2*σ2])
        axis.set_ylim([-3.2*σ2, 3.2*σ2])
    else:
        axis.set_xlim([-3.2*σ1, 3.2*σ1])
        axis.set_ylim([-3.2*σ2, 3.2*σ2])
    title = f"Bivariate Normal Distribution: γ={format(γ, '2.1f')}, " + \
             r"$σ_u$=" + f"{format(σ1, '2.1f')}, " + r"$σ_v$=" + \
             f"{format(σ2, '2.1f')}"
    axis.set_title(title)
    contour = axis.contour(x1_grid, x2_grid, f_x1_x2, contour_values, cmap=config.contour_color_map)
    axis.clabel(contour, contour.levels[::2], fmt="%.3f", inline=True, fontsize=15)
    config.save_post_asset(figure, post, plot_name)
