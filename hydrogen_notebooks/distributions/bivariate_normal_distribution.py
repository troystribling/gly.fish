# %%
%load_ext autoreload
%autoreload 2

import numpy
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
from glyfish import config
from glyfish import stats

%matplotlib inline

pyplot.style.use(config.glyfish_style)

# %%

def pdf(μ1, μ2, σ1, σ2, ρ):
    def f(x1, x2):
        y1 = (x1 - μ1) / σ1
        y2 = (x2 - μ2) / σ2
        c = 2 * numpy.pi * σ1 * σ2 * numpy.sqrt(1.0 - ρ**2)
        ε = (y1**2 + y2**2 - 2.0 * ρ * y1 * y2) / (2.0 * (1.0 - ρ**2))
        return numpy.exp(-ε) / c
    return f

def conditional_pdf_y1_y2(μ1, μ2, σ1, σ2, ρ):
    def f(x1, x2):
        y1 = (x1 - μ1)
        y2 = (x2 - μ2)
        c = numpy.sqrt(2 * numpy.pi * σ1 * (1.0 - ρ**2))
        ε = (y1 - ρ * σ1 * y2 / σ2)**2 / (2.0 * σ1**2 * (1.0 - ρ**2))
        return numpy.exp(-ε) / c
    return f

def normal(x, σ=1.0, μ=0.0):
    ε = (x - μ)**2/(2.0*σ**2)
    return numpy.exp(-ε)/numpy.sqrt(2.0*numpy.pi*σ**2)

def conditional_pdf_generator(y, μ1, μ2, σ1, σ2, ρ):
    loc = μ1 + ρ * σ1 * (y - μ2) / σ2
    scale = numpy.sqrt((1.0 - ρ**2) * σ1**2)
    return numpy.random.normal(loc, scale)

def pdf_transform_y1_constant(μ1, μ2, σ1, σ2, ρ, c1):
    def f(y2):
        x1 = (c1 - μ1) / σ1
        x2 = ((σ2 * ρ) * (c1 - μ1) / σ1 - (y2 - μ2)) / (σ2 * numpy.sqrt(1.0 - ρ**2))
        return (numpy.full((len(x2)), x1), x2)
    return f

def pdf_transform_y2_constant(μ1, μ2, σ1, σ2, ρ, c2):
    def f(y1):
        x1 = (y1 - μ1) / σ1
        x2 = ((σ2 * ρ) * (y1 - μ1) / σ1 - (c2 - μ2)) / (σ2 * numpy.sqrt(1.0 - ρ**2))
        return (x1, x2)
    return f

def pdf_contour_constant(μ1, μ2, σ1, σ2, ρ, v):
    return numpy.sqrt(-2.0 * (1.0 - ρ**2) * numpy.log(2.0*numpy.pi * σ1 * σ2 * v * numpy.sqrt(1.0 - ρ**2)))

def max_pdf_value(σ1, σ2, ρ):
    return 1.0/(2.0 * numpy.pi * σ1 * σ2 * numopy.sqrt(1.0 - ρ**2))

def pdf_parametric_contour(μ1, μ2, σ1, σ2, ρ):
    def f(θ, c):
        y1 = c * σ1 * (numpy.sin(θ) + numpy.cos(θ) * ρ / numpy.sqrt(1.0 - ρ**2)) + μ1
        y2 = c * σ2 * numpy.cos(θ) / numpy.sqrt(1.0 - ρ**2) + μ2
        return (y1, y2)
    return f

# %%

def pdf_mesh(μ1, μ2, σ1, σ2, ρ):
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

    f = pdf(μ1, μ2, σ1, σ2, ρ)
    x1_grid, x2_grid = numpy.meshgrid(x1, x2)
    f_x1_x2 = numpy.zeros((npts, npts))
    for i in numpy.arange(npts):
        for j in numpy.arange(npts):
            f_x1_x2[i, j] = f(x1_grid[i,j], x2_grid[i,j])
    return (x1_grid, x2_grid, f_x1_x2)

def contour_plot(μ1, μ2, σ1, σ2, ρ, contour_values, plot_name):
    x1_grid, x2_grid, f_x1_x2 = pdf_mesh(μ1, μ2, σ1, σ2, ρ)
    figure, axis = pyplot.subplots(figsize=(8, 8))
    axis.set_xlabel(r"$x$")
    axis.set_ylabel(r"$y$")
    axis.set_title(f"Bivariate Normal PDF: ρ={format(ρ, '2.1f')}, σ1={format(σ1, '2.1f')}, σ2={format(σ2, '2.1f')}")
    contour = axis.contour(x1_grid, x2_grid, f_x1_x2, contour_values, cmap=config.contour_color_map)
    axis.clabel(contour, contour.levels[::2], fmt="%.3f", inline=True, fontsize=15)
    config.save_post_asset(figure, "bivariate_normal_distribution", plot_name)

def surface_plot(μ1, μ2, σ1, σ2, ρ, zticks, plot_name):
    x1_grid, x2_grid, f_x1_x2 = pdf_mesh(μ1, μ2, σ1, σ2, ρ)
    figure, axis = pyplot.subplots(figsize=(10, 10))
    axis.set_title(f"Bivariate Normal PDF: ρ={format(ρ, '2.1f')}, σ1={format(σ1, '2.1f')}, σ2={format(σ2, '2.1f')}")
    axis.set_yticklabels([])
    axis.set_xticklabels([])

    axis_z = figure.add_subplot(111, projection='3d')
    axis.set_xlabel(r"$x$")
    axis.set_ylabel(r"$y$")
    axis_z.set_zticks(zticks)
    axis_z.set_xticks([-σ1*3.0, -σ1*2.0, -σ1, 0.0, σ1, σ1*2.0, σ1*3.0])
    axis_z.set_yticks([-σ2*3.0, -σ2*2.0, -σ2, 0.0, σ2, σ2*2.0, σ2*3.0])
    axis_z.plot_wireframe(x1_grid, x2_grid, f_x1_x2, rstride=25, cstride=25)
    config.save_post_asset(figure, "bivariate_normal_distribution", plot_name)

# %%
# Surface Contour Plot comparison
σ1 = 1.0
σ2 = 1.0
μ1 = 0.0
μ2 = 0.0
ρ = 0.5

# %%

surface_plot(μ1, μ2, σ1, σ2, ρ, [0.00, 0.05, 0.1, 0.15], "bivariate_pdf_surface_plot")

# %%

contour_plot(μ1, μ2, σ1, σ2, ρ,
             [0.005, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15],
             "bivariate_pdf_contour_plot")

# %%

def parametric_contour_plot(μ1, μ2, σ1, σ2, ρ, contour_values, plot_name):
    npts = 500
    θ = numpy.linspace(0.0, 2.0 * numpy.pi, npts)
    c = [pdf_contour_constant(μ1, μ2, σ1, σ2, ρ, v) for v in contour_values]
    f = pdf_parametric_contour(μ1, μ2, σ1, σ2, ρ)


    figure, axis = pyplot.subplots(figsize=(8, 8))
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    if (σ1 > σ2):
        axis.set_xlim([-3.0*σ1, 3.0*σ1])
        axis.set_ylim([-3.0*σ1, 3.0*σ1])
    elif (σ2 > σ1):
        axis.set_xlim([-3.0*σ2, 3.0*σ2])
        axis.set_ylim([-3.0*σ2, 3.0*σ2])
    else:
        axis.set_xlim([-3.0*σ1, 3.0*σ1])
        axis.set_ylim([-3.0*σ2, 3.0*σ2])
    axis.set_title(f"Bivariate Normal PDF: ρ={format(ρ, '2.1f')}, σ1={format(σ1, '2.1f')}, σ2={format(σ1, '2.1f')}")

    for n in range(len(c)):
        y1, y2 = f(θ, c[n])
        axis.plot(y1, y2, label=f"= {format(contour_values[n], '2.3f')}")

    axis.legend()
    config.save_post_asset(figure, "bivariate_normal_distribution", plot_name)

# %%
# Parametric countor plot validation

σ1 = 1.0
σ2 = 1.0
μ1 = 0.0
μ2 = 0.0
ρ = 0.0

# %%

contour_plot(μ1, μ2, σ1, σ2, ρ,
             [0.005, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15],
             "bivariate_pdf_contours_correlation_0.0")

# %%

parametric_contour_plot(μ1, μ2, σ1, σ2, ρ,
                        [0.005, 0.05, 0.1, 0.15],
                        'bivariate_pdf_parameterized_contour_correlation_0.0')

# %%

σ1 = 1.0
σ2 = 1.0
μ1 = 0.0
μ2 = 0.0
ρ = 0.5

# %%

contour_plot(μ1, μ2, σ1, σ2, ρ,
             [0.005, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15],
             "bivariate_pdf_contours_correlation_0.5")

# %%

parametric_contour_plot(μ1, μ2, σ1, σ2, ρ,
                        [0.005, 0.05, 0.1, 0.15],
                        'bivariate_pdf_parameterized_contour_correlation_0.5')

# %%

σ1 = 1.0
σ2 = 2.0
μ1 = 0.0
μ2 = 0.0
ρ = 0.0

# %%

contour_plot(μ1, μ2, σ1, σ2, ρ,
             [0.005, 0.0125, 0.025, 0.0375, 0.05, 0.0625, 0.075],
             "bivariate_pdf_contours_sigma_2.0")

# %%

parametric_contour_plot(μ1, μ2, σ1, σ2, ρ,
                        [0.005, 0.025, 0.05, 0.075],
                        'bivariate_pdf_parameterized_contours_sigma_2.0')

# %%
# Distribution variation with ρ

σ1 = 1.0
σ2 = 1.0
μ1 = 0.0
μ2 = 0.0
ρ = [-0.95, -0.5, 0.0, 0.5, 0.95]
contour_value = 0.1

npts = 500
θ = numpy.linspace(0.0, 2.0 * numpy.pi, npts)

figure, axis = pyplot.subplots(figsize=(8, 8))
axis.set_xlabel("x")
axis.set_ylabel("y")
axis.set_xlim([-3.5*σ1, 3.5*σ1])
axis.set_ylim([-3.5*σ2, 3.5*σ2])
axis.set_title(f"Bivariate Normal PDF: σ1={format(σ1, '2.1f')}, σ2={format(σ2, '2.1f')}, v={format(contour_value, '2.1f')}")

for i in range(len(ρ)):
    c = pdf_contour_constant(μ1, μ2, σ1, σ2, ρ[i], contour_value)
    f = pdf_parametric_contour(μ1, μ2, σ1, σ2, ρ[i])
    y1, y2 = f(θ, c)
    axis.plot(y1, y2, label=f"ρ = {format(ρ[i], '2.3f')}")

axis.legend(bbox_to_anchor=(0.5, 0.29))
config.save_post_asset(figure, "bivariate_normal_distribution", "bivariate_pdf_parameterized_contour_correlation_scan")

# %%
# Distribution variation with σ

σ1 = 1.0
σ2 = [1.0, 2.0, 3.0, 4.0]
μ1 = 0.0
μ2 = 0.0
ρ = 0.0
contour_value = 0.02

npts = 500
θ = numpy.linspace(0.0, 2.0 * numpy.pi, npts)

figure, axis = pyplot.subplots(figsize=(8, 8))
axis.set_xlabel(r"$x$")
axis.set_ylabel(r"$y$")
axis.set_xlim([-5.0, 5.0])
axis.set_ylim([-5.0, 5.0])
axis.set_title(f"Bivariate Normal PDF: σ1={format(σ1, '2.1f')}, ρ={format(ρ, '2.1f')}, v={format(contour_value, '2.1f')}")

for i in range(len(σ2)):
    c = pdf_contour_constant(μ1, μ2, σ1, σ2[i], ρ, contour_value)
    f = pdf_parametric_contour(μ1, μ2, σ1, σ2[i], ρ)
    y1, y2 = f(θ, c)
    axis.plot(y1, y2, label=f"σ2 = {format(σ2[i], '2.3f')}")

axis.legend(bbox_to_anchor=(0.7, 0.95))
config.save_post_asset(figure, "bivariate_normal_distribution", "bivariate_pdf_parameterized_contour_sigma_scan")

# %%
## normal distribution examples

x = numpy.linspace(-7.0, 7.0, 500)
figure, axis = pyplot.subplots(figsize=(10, 7))
axis.set_xlabel(r"$x$")
axis.set_ylabel("PDF")
axis.set_ylim([0.0, 1.5])
axis.set_title("Normal Distribution")
axis.yaxis.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
σ = [0.3, 0.5, 1.0, 2.0]
μ = [-4.0, -2.0, 0.0, 2.0]
for i in range(len(σ)):
    pdf = [stats.normal(j, σ[i], μ[i]) for j in x]
    axis.plot(x, pdf, label=f"σ={σ[i]}, μ={μ[i]}")
axis.legend(bbox_to_anchor=(0.9, 0.95))
config.save_post_asset(figure, "bivariate_normal_distribution", "normal_distribution_parameters")

# %%
# Bivariate conditional distribution
σ1 = 1.0
σ2 = 1.0
μ1 = 0.0
μ2 = 0.0
ρ = [-0.95, -0.5, 0.0, 0.5, 0.95]
x2 = 1.0

x1 = numpy.linspace(-4.0, 4.0, 500)
figure, axis = pyplot.subplots(figsize=(10, 7))
axis.set_xlabel(r"$x$")
axis.set_ylabel(r"$g_{X|Y}$")
axis.set_ylim([0.0, 1.5])
axis.set_title("Bivariate Conditional PDF: "+r"$X_{2}=$"+f"{format(x2, '2.1f')}")
axis.yaxis.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
for i in range(len(ρ)):
    f = conditional_pdf_y1_y2(μ1, μ2, σ1, σ2, ρ[i])
    axis.plot(x1, f(x1, x2), label=f"ρ={ρ[i]}")
axis.legend(bbox_to_anchor=(0.95, 0.9))
config.save_post_asset(figure, "bivariate_normal_distribution", "bivariate_conditional_pdf_correlation_scan")

# %%

σ1 = 1.0
σ2 = 1.0
μ1 = 0.0
μ2 = 0.0
ρ = 0.5
x2 = [-2.0, -1.0, 0.0, 1.0, 2.0]

x1 = numpy.linspace(-4.0, 4.0, 500)
figure, axis = pyplot.subplots(figsize=(10, 7))
axis.set_xlabel(r"$x$")
axis.set_ylabel(r"$g_{X|Y}$")
axis.set_ylim([0.0, 0.65])
axis.yaxis.set_ticks([0.0, 0.2, 0.4, 0.6])
axis.set_title(f"Bivariate Conditional PDF: ρ={ρ}")
for i in range(len(x2)):
    f = conditional_pdf_y1_y2(μ1, μ2, σ1, σ2, ρ)
    axis.plot(x1, f(x1, x2[i]), label=r"$x_{2}=$"+f"{format(x2[i], '2.1f')}")
axis.legend(bbox_to_anchor=(0.95, 0.95))
config.save_post_asset(figure, "bivariate_normal_distribution", "bivariate_conditional_pdf_y_scan")

# %%

def transform_plot(μ1, μ2, σ1, σ2, ρ, xrange, yrange, xnudge, ynudge, unudge, anudge, legend, plot_name):
    npts = 500

    figure, axis = pyplot.subplots(figsize=(10, 7))
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_ylim(yrange)
    axis.set_xlim(xrange)
    title = f"Bivariate Normal Transformation: ρ={format(ρ, '2.1f')}, " + \
             r"$σ_1$=" + f"{format(σ1, '2.1f')}, " + r"$σ_2$=" + \
             f"{format(σ1, '2.1f')}"
    axis.set_title(title)

    ar = (yrange[1] - yrange[0]) / (xrange[1] - xrange[0])
    θ = (180.0 / numpy.pi) * numpy.arctan(ρ / (ar * numpy.sqrt(1-ρ**2))) + anudge

    c = [-4.0, -2.0, 0.0, 2.0, 4.0]
    x1 = numpy.zeros(len(c))
    for i in range(len(c)):
        y2 = numpy.linspace(2.0 * yrange[0], 2.0 * yrange[1], npts)
        transform_y1 = pdf_transform_y1_constant(μ1, μ2, σ1, σ2, ρ, c[i])
        x1_y1, x2_y1 = transform_y1(y2)
        x1[i] = x1_y1[0]
        axis.text(x1[i] - 0.35, unudge[i], f"u={format(c[i], '2.0f')}", fontsize=18, rotation=90.0)
        if i == 0:
            axis.plot(x1_y1, x2_y1, color="#0067C4", label=r"Constant $u$")
        else:
            axis.plot(x1_y1, x2_y1, color="#0067C4")

    for i in range(len(c)):
        y1 = numpy.linspace(2.0 * yrange[0], 2.0 * yrange[1], npts)
        transform_y2 = pdf_transform_y2_constant(μ1, μ2, σ1, σ2, ρ, c[i])
        x1_y2, x2_y2 = transform_y2(y1)
        if i == len(c) - 1:
            xoffset = x1[i] - 2.0 * xnudge
        else:
            xoffset = x1[i] + xnudge
        x2 = (ρ * xoffset - (c[i] - μ2) / σ2) / numpy.sqrt(1.0 - ρ**2)
        axis.text(xoffset, x2 + ynudge, f"v={format(c[i], '2.0f')}", fontsize=18, rotation=θ)
        if i == 0:
            axis.plot(x1_y2, x2_y2, color="#FF9500", label=r"Constant $v$")
        else:
            axis.plot(x1_y2, x2_y2, color="#FF9500")

    axis.legend(bbox_to_anchor=legend)
    config.save_post_asset(figure, "bivariate_normal_distribution", "bivariate_normal_transformation")

# Bivariate transformation
# %%

σ1 = 1.0
σ2 = 1.0
μ1 = 0.0
μ2 = 0.0
ρ = 0.95

transform_plot(μ1, μ2, σ1, σ2, ρ, [-5.0, 5.0], [-10.0, 10.0], 0.65, 2.5, [-2.5, -2.75, -3.5, -4.0, 2.0], -10.0, (0.75, 0.2), "bivariate_normal_transformation_correlation_0.95")

# %%

σ1 = 1.0
σ2 = 1.0
μ1 = 0.0
μ2 = 0.0
ρ = 0.5

transform_plot(μ1, μ2, σ1, σ2, ρ, [-5.0, 5.0], [-5.0, 5.0], 0.65, 0.75, [-1.0, -2.25, -1.25, -2.4, 3.5], -10.0, (0.75, 0.2), "bivariate_normal_transformation_correlation_0.5")

# %%

σ1 = 1.0
σ2 = 1.0
μ1 = 0.0
μ2 = 0.0
ρ = 0.0

transform_plot(μ1, μ2, σ1, σ2, ρ, [-5.0, 5.0], [-5.0, 5.0], 0.65, 0.15, [-0.75, -0.75, -0.75, -0.75, 1.25], 0.0, (0.35, 0.1), "bivariate_normal_transformation_correlation_0")
