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

def pdf(μ1, μ2, σ1, σ2, γ):
    def f(x1, x2):
        y1 = (x1 - μ1) / σ1
        y2 = (x2 - μ2) / σ2
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

def normal(x, σ=1.0, μ=0.0):
    ε = (x - μ)**2/(2.0*σ**2)
    return numpy.exp(-ε)/numpy.sqrt(2.0*numpy.pi*σ**2)

def conditional_pdf_generator(y, μ1, μ2, σ1, σ2, γ):
    loc = μ1 + γ * σ1 * (y - μ2) / σ2
    scale = numpy.sqrt((1.0 - γ**2) * σ1**2)
    return numpy.random.normal(loc, scale)

def pdf_transform_y1_constant(μ1, μ2, σ1, σ2, γ, c1):
    def f(y2):
        x1 = (c1 - μ1) / σ1
        x2 = ((σ2 * γ) * (c1 - μ1) / σ1 - (y2 - μ2)) / (σ2 * numpy.sqrt(1.0 - γ**2))
        return (numpy.full((len(x2)), x1), x2)
    return f

def pdf_transform_y2_constant(μ1, μ2, σ1, σ2, γ, c2):
    def f(y1):
        x1 = (y1 - μ1) / σ1
        x2 = ((c2 - μ2) - (σ2 * γ) * (y1 - μ1) / σ1) / (σ2 * numpy.sqrt(1.0 - γ**2))
        return (x1, x2)
    return f

def pdf_contour_constant(μ1, μ2, σ1, σ2, γ, v):
    t1 = 2.0*numpy.pi * σ1 * σ2 * v * numpy.sqrt(1.0 - γ**2)
    t2 = -2.0 * (1.0 - γ**2)
    return numpy.sqrt(t2 * numpy.log(t1))

def max_pdf_value(σ1, σ2, γ):
    return 1.0/(2.0 * numpy.pi * σ1 * σ2 * numopy.sqrt(1.0 - γ**2))

def pdf_parametric_contour(μ1, μ2, σ1, σ2, γ):
    def f(θ, c):
        y1 = c * σ1 * (numpy.sin(θ) + numpy.cos(θ) * γ / numpy.sqrt(1.0 - γ**2)) + μ1
        y2 = c * σ2 * numpy.cos(θ) / numpy.sqrt(1.0 - γ**2) + μ2
        return (y1, y2)
    return f

def r(μ1, μ2, σ1, σ2, γ):
    def f(θ, c):
        t1 = numpy.sin(θ)**2
        t2 = numpy.sin(θ)*numpy.cos(θ)*(2.0*γ/numpy.sqrt(1-γ**2))
        t3 = numpy.cos(θ)**2/(1-γ**2)
        t4 = γ**2 + σ2**2/σ1**2
        return σ1*c*numpy.sqrt(t1 + t2 + t3*t4)
    return f

# %%

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
            f_x1_x2[i, j] = f(x1_grid[i,j], x2_grid[i,j])
    return (x1_grid, x2_grid, f_x1_x2)

def contour_plot(μ1, μ2, σ1, σ2, γ, contour_values, plot_name):
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
    config.save_post_asset(figure, "bivariate_normal_distribution", plot_name)

def surface_plot(μ1, μ2, σ1, σ2, γ, zticks, plot_name):
    x1_grid, x2_grid, f_x1_x2 = pdf_mesh(μ1, μ2, σ1, σ2, γ)
    figure, axis = pyplot.subplots(figsize=(10, 10))
    title = f"Bivariate Normal Distribution: γ={format(γ, '2.1f')}, " + \
             r"$σ_u$=" + f"{format(σ1, '2.1f')}, " + r"$σ_v$=" + \
             f"{format(σ2, '2.1f')}"
    axis.set_title(title)
    axis.set_yticklabels([])
    axis.set_xticklabels([])

    axis_z = figure.add_subplot(111, projection='3d')
    axis_z.set_xlabel(r"$u$")
    axis_z.set_ylabel(r"$v$")
    axis_z.set_zticks(zticks)
    axis_z.set_xticks([-σ1*3.0, -σ1*2.0, -σ1, 0.0, σ1, σ1*2.0, σ1*3.0])
    axis_z.set_yticks([-σ2*3.0, -σ2*2.0, -σ2, 0.0, σ2, σ2*2.0, σ2*3.0])
    axis_z.plot_wireframe(x1_grid, x2_grid, f_x1_x2, rstride=25, cstride=25)
    axis_z.elev = 55.0
    axis_z.azim = 45.0
    config.save_post_asset(figure, "bivariate_normal_distribution", plot_name)

# %%
# Surface Contour Plot comparison
σ1 = 1.0
σ2 = 1.0
μ1 = 0.0
μ2 = 0.0
γ = 0.5

# %%

surface_plot(μ1, μ2, σ1, σ2, γ, [0.00, 0.05, 0.1, 0.15], "bivariate_pdf_surface_plot_0.5_1")

# %%

contour_plot(μ1, μ2, σ1, σ2, γ,
             [0.005, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15],
             "bivariate_pdf_contour_plot_0.5_1")

# %%

# Surface Contour Plot comparison
σ1 = 1.0
σ2 = 1.0
μ1 = 0.0
μ2 = 0.0
γ = 0.0
# %%

surface_plot(μ1, μ2, σ1, σ2, γ, [0.00, 0.05, 0.1, 0.15], "bivariate_pdf_surface_plot_0.0_1")

# %%

# Surface Contour Plot comparison
contour_plot(μ1, μ2, σ1, σ2, γ,
             [0.005, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15],
             "bivariate_pdf_contour_plot_0.0_1")

# %%
σ1 = 1.0
σ2 = 2.0
μ1 = 0.0
μ2 = 0.0
γ = 0.0

surface_plot(μ1, μ2, σ1, σ2, γ, [0.00, 0.05, 0.1, 0.15], "bivariate_pdf_surface_plot_0.0_2")

# %%

contour_plot(μ1, μ2, σ1, σ2, γ,
             [0.005, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15],
             "bivariate_pdf_contour_plot_0.0_2")

# %%

def parametric_contour_plot(μ1, μ2, σ1, σ2, γ, legend_box, contour_values, plot_name):
    npts = 500
    θ = numpy.linspace(0.0, 2.0 * numpy.pi, npts)
    c = [pdf_contour_constant(μ1, μ2, σ1, σ2, γ, v) for v in contour_values]
    f = pdf_parametric_contour(μ1, μ2, σ1, σ2, γ)

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

    for n in range(len(c)):
        y1, y2 = f(θ, c[n])
        axis.plot(y1, y2, label=f"= {format(contour_values[n], '2.3f')}")

    axis.legend(bbox_to_anchor=legend_box)
    config.save_post_asset(figure, "bivariate_normal_distribution", plot_name)

# %%
# Parametric countor plot validation

σ1 = 1.0
σ2 = 1.0
μ1 = 0.0
μ2 = 0.0
γ = 0.0

# %%

contour_plot(μ1, μ2, σ1, σ2, γ,
             [0.005, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15],
             "bivariate_pdf_contours_correlation_0.0")

# %%

parametric_contour_plot(μ1, μ2, σ1, σ2, γ, (0.7, 0.7),
                        [0.005, 0.05, 0.1, 0.15],
                        'bivariate_pdf_parameterized_contour_correlation_0.0')

# %%

σ1 = 1.0
σ2 = 1.0
μ1 = 0.0
μ2 = 0.0
γ = 0.5

# %%

contour_plot(μ1, μ2, σ1, σ2, γ,
             [0.005, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15],
             "bivariate_pdf_contours_correlation_0.5")

# %%

parametric_contour_plot(μ1, μ2, σ1, σ2, γ, (0.32, 0.75),
                        [0.005, 0.05, 0.1, 0.15],
                        'bivariate_pdf_parameterized_contour_correlation_0.5')

# %%

σ1 = 1.0
σ2 = 2.0
μ1 = 0.0
μ2 = 0.0
γ = 0.0

# %%

contour_plot(μ1, μ2, σ1, σ2, γ,
             [0.005, 0.0125, 0.025, 0.0375, 0.05, 0.0625, 0.075],
             "bivariate_pdf_contours_sigma_2.0")

# %%

parametric_contour_plot(μ1, μ2, σ1, σ2, γ, (0.32, 0.95),
                        [0.005, 0.025, 0.05, 0.075],
                        'bivariate_pdf_parameterized_contours_sigma_2.0')

# %%

σ1 = 1.0
σ2 = 2.0
μ1 = 0.0
μ2 = 0.0
γ = 0.5

# %%

contour_plot(μ1, μ2, σ1, σ2, γ,
             [0.005, 0.0125, 0.025, 0.0375, 0.05, 0.0625, 0.075],
             "bivariate_pdf_contours_sigma_2.0_correlation_0.5")

# %%

parametric_contour_plot(μ1, μ2, σ1, σ2, γ, (0.32, 0.95),
                        [0.005, 0.025, 0.05, 0.075],
                        'bivariate_pdf_parameterized_contours_sigma_2.0_correlation_0.5')

# %%

def contour_plot_correlation_scan(μ1, μ2, σ1, σ2, γ, contour_value, legend, plot_name):
    npts = 500
    θ = numpy.linspace(0.0, 2.0 * numpy.pi, npts)

    figure, axis = pyplot.subplots(figsize=(8, 8))
    axis.set_xlabel(r"$u$")
    axis.set_ylabel(r"$v$")
    σ = numpy.amax([σ1, σ2])
    axis.set_xlim([-2.2*σ, 2.2*σ])
    axis.set_ylim([-2.2*σ, 2.2*σ])
    title = f"Bivariate Normal Distribution: " + \
             r"$σ_u$=" + f"{format(σ1, '2.1f')}, " + r"$σ_v$=" + \
             f"{format(σ2, '2.1f')}, K={format(contour_value, '2.1f')}"
    axis.set_title(title)

    for i in range(len(γ)):
        c = pdf_contour_constant(μ1, μ2, σ1, σ2, γ[i], contour_value)
        f = pdf_parametric_contour(μ1, μ2, σ1, σ2, γ[i])
        y1, y2 = f(θ, c)
        axis.plot(y1, y2, label=f"γ = {format(γ[i], '2.2f')}", zorder = 7)

    x1 = numpy.linspace(-2.0*σ, 2.0*σ, npts)
    slope = γ[-1]*σ2/σ1/abs(γ[-1])
    axis.plot(x1, slope*x1, zorder = 6, color="#003B6F", alpha=0.1)

    axis.legend(bbox_to_anchor=legend, ncol=2)
    config.save_post_asset(figure, "bivariate_normal_distribution", plot_name)

# %%
# Distribution variation with positive γ

σ1 = 1.0
σ2 = 1.0
μ1 = 0.0
μ2 = 0.0
γ = [0.0, 0.25, 0.5, 0.75, 0.9]
contour_value = 0.1
contour_plot_correlation_scan(μ1, μ2, σ1, σ2, γ, contour_value, (0.45, 0.22), "bivariate_pdf_parameterized_contour_positive_correlation_scan")

# %%
# Distribution variation with positive γ

σ1 = 1.0
σ2 = 1.0
μ1 = 0.0
μ2 = 0.0
γ = [0.0, -0.25, -0.5, -0.75, -0.9]
contour_value = 0.1

contour_plot_correlation_scan(μ1, μ2, σ1, σ2, γ, contour_value, (0.6, 0.22), "bivariate_pdf_parameterized_contour_negative_correlation_scan")

# %%

def contour_axis_plot_correlation_scan(μ1, μ2, σ1, σ2, γ, contour_value, legend, plot_name):
    npts = 500
    θ = numpy.linspace(0.0, 2.0 * numpy.pi, npts)
    figure, axis = pyplot.subplots(figsize=(8, 8))
    title = f"Bivariate Normal Distribution: " + \
             r"$σ_u$=" + f"{format(σ1, '2.1f')}, " + r"$σ_v$=" + \
             f"{format(σ2, '2.1f')}, K={format(contour_value, '2.2f')}"
    axis.set_title(title)
    axis.set_xlabel(r"$\theta$")
    axis.set_ylabel(r"$r(\theta)$")
    for i in range(len(γ)):
        c = pdf_contour_constant(μ1, μ2, σ1, σ2, γ[i], contour_value)
        rf = r(μ1, μ2, σ1, σ2, γ[i])
        axis.plot(θ, rf(θ, c), label=f"γ = {format(γ[i], '2.2f')}")

    axis.legend(bbox_to_anchor=legend, ncol=2)
    config.save_post_asset(figure, "bivariate_normal_distribution", plot_name)

# %%

σ1 = 1.0
σ2 = 1.0
μ1 = 0.0
μ2 = 0.0
γ = [0.0, 0.25, 0.5, 0.75, 0.9]
contour_value = 0.1

contour_axis_plot_correlation_scan(μ1, μ2, σ1, σ2, γ, contour_value, (0.45, 0.22), "bivariate_pdf_parameterized_contour_axis_correlation_scan")

# %%

def contour_axis_plot_sigma_scan(μ1, μ2, σ1, σ2, γ, contour_value, legend, plot_name):
    npts = 500
    θ = numpy.linspace(0.0, 2.0 * numpy.pi, npts)
    figure, axis = pyplot.subplots(figsize=(8, 8))
    title = f"Bivariate Normal Distribution: " + \
             r"$σ_u$=" + f"{format(σ1, '2.1f')}, " + r"$\gamma$=" + \
             f"{format(γ, '2.1f')}, K={format(contour_value, '2.2f')}"
    axis.set_title(title)
    axis.set_xlabel(r"$\theta$")
    axis.set_ylabel(r"$r(\theta)$")
    for i in range(len(σ2)):
        c = pdf_contour_constant(μ1, μ2, σ1, σ2[i], γ, contour_value)
        rf = r(μ1, μ2, σ1, σ2[i], γ)
        axis.plot(θ, rf(θ, c), label=f"$σ_v$ = {format(σ2[i], '2.2f')}")

    axis.legend(bbox_to_anchor=legend)
    config.save_post_asset(figure, "bivariate_normal_distribution", plot_name)

# %%

σ1 = 1.0
σ2 = [1.0, 2.0, 3.0, 4.0]
μ1 = 0.0
μ2 = 0.0
γ = 0.0
contour_value = 0.02

contour_axis_plot_sigma_scan(μ1, μ2, σ1, σ2, γ, contour_value, (0.5, 0.25), "bivariate_pdf_parameterized_contour_axis_sigma2_scan")

# %%

σ1 = 1.0
σ2 = 1.0
μ1 = 0.0
μ2 = 0.0
γ = [0.0, 0.25, 0.5, 0.75, 0.9]
contour_value = 0.1

# %%

def contour_plot_sigma2_scan(μ1, μ2, σ1, σ2, γ, contour_value, legend, plot_name):
    npts = 500
    θ = numpy.linspace(0.0, 2.0 * numpy.pi, npts)

    figure, axis = pyplot.subplots(figsize=(8, 8))
    axis.set_xlabel(r"$u$")
    axis.set_ylabel(r"$v$")
    axis.set_xlim([-5.0, 5.0])
    axis.set_ylim([-5.0, 5.0])
    title = f"Bivariate Normal Distribution: γ={format(γ, '2.1f')}, " + \
         r"$σ_u$=" + f"{format(σ1, '2.1f')}, " + r"$K$=" + \
         f"{format(contour_value, '2.2f')}"
    axis.set_title(title)

    for i in range(len(σ2)):
        c = pdf_contour_constant(μ1, μ2, σ1, σ2[i], γ, contour_value)
        f = pdf_parametric_contour(μ1, μ2, σ1, σ2[i], γ)
        y1, y2 = f(θ, c)
        axis.plot(y1, y2, label=r"$σ_v$" + f" = {format(σ2[i], '2.3f')}")

    axis.legend(bbox_to_anchor=legend)
    config.save_post_asset(figure, "bivariate_normal_distribution", plot_name)

# %%
# Distribution variation with σ

σ1 = 1.0
σ2 = [1.0, 2.0, 3.0, 4.0]
μ1 = 0.0
μ2 = 0.0
γ = 0.0
contour_value = 0.02

contour_plot_sigma2_scan(μ1, μ2, σ1, σ2, γ, contour_value, (0.7, 0.95), "bivariate_pdf_parameterized_contour_sigma2_scan")

# %%

def contour_plot_sigma1_scan(μ1, μ2, σ1, σ2, γ, contour_value, legend, plot_name):
    npts = 500
    θ = numpy.linspace(0.0, 2.0 * numpy.pi, npts)

    figure, axis = pyplot.subplots(figsize=(8, 8))
    axis.set_xlabel(r"$u$")
    axis.set_ylabel(r"$v$")
    axis.set_xlim([-5.0, 5.0])
    axis.set_ylim([-5.0, 5.0])
    title = f"Bivariate Normal Distribution: γ={format(γ, '2.1f')}, " + \
         r"$σ_v$=" + f"{format(σ2, '2.1f')}, " + r"$K$=" + \
         f"{format(contour_value, '2.2f')}"
    axis.set_title(title)

    for i in range(len(σ1)):
        c = pdf_contour_constant(μ1, μ2, σ1[i], σ2, γ, contour_value)
        f = pdf_parametric_contour(μ1, μ2, σ1[i], σ2, γ)
        y1, y2 = f(θ, c)
        axis.plot(y1, y2, label=r"$σ_u$" + f" = {format(σ1[i], '2.3f')}")

    axis.legend(bbox_to_anchor=legend)
    config.save_post_asset(figure, "bivariate_normal_distribution", plot_name)

# %%
# Distribution variation with σ

σ1 = [1.0, 2.0, 3.0, 4.0]
σ2 = 1.0
μ1 = 0.0
μ2 = 0.0
γ = 0.0
contour_value = 0.02

contour_plot_sigma1_scan(μ1, μ2, σ1, σ2, γ, contour_value, (0.4, 0.95), "bivariate_pdf_parameterized_contour_sigma1_scan")

# %%

def contour_plot_sigma_correlation_scan(μ1, μ2, σ1, σ2, γ, contour_value, legend, plot_name):
    npts = 500
    θ = numpy.linspace(0.0, 2.0 * numpy.pi, npts)

    figure, axis = pyplot.subplots(figsize=(8, 8))
    axis.set_xlabel(r"$u$")
    axis.set_ylabel(r"$v$")
    σ = numpy.amax([σ1, σ2])
    axis.set_xlim([-2.5*σ, 2.5*σ])
    axis.set_ylim([-2.5*σ, 2.5*σ])
    title = f"Bivariate Normal Distribution: " + \
             r"$σ_u$=" + f"{format(σ1, '2.1f')}, " + r"$σ_v$=" + \
             f"{format(σ2, '2.1f')}, K={format(contour_value, '2.2f')}"
    axis.set_title(title)

    for i in range(len(γ)):
        c = pdf_contour_constant(μ1, μ2, σ1, σ2, γ[i], contour_value)
        f = pdf_parametric_contour(μ1, μ2, σ1, σ2, γ[i])
        y1, y2 = f(θ, c)
        axis.plot(y1, y2, label=f"γ = {format(γ[i], '2.2f')}")

    x1 = numpy.linspace(-2.5*σ, 2.5*σ, npts)
    slope = γ[-1]*σ2/σ1/abs(γ[-1])
    axis.plot(x1, slope*x1, zorder = 6, color="#003B6F", alpha=0.1)

    axis.legend(bbox_to_anchor=legend)
    config.save_post_asset(figure, "bivariate_normal_distribution", plot_name)

# %%
# Distribution variation with γ

σ1 = 1.0
σ2 = 2.0
μ1 = 0.0
μ2 = 0.0
γ = [0.0, 0.25, 0.5, 0.75, 0.95]
contour_value = 0.02

contour_plot_sigma_correlation_scan(μ1, μ2, σ1, σ2, γ, contour_value, (0.75, 0.4), "bivariate_pdf_parameterized_contour_correlation_sigma2_scan")

# %%
# Distribution variation with γ

σ1 = 2.0
σ2 = 1.0
μ1 = 0.0
μ2 = 0.0
γ = [0.0, 0.25, 0.5, 0.75, 0.95]
contour_value = 0.02

contour_plot_sigma_correlation_scan(μ1, μ2, σ1, σ2, γ, contour_value, (0.7, 0.35), "bivariate_pdf_parameterized_contour_correlation_sigma1_scan")

# %%
## normal distribution examples

x = numpy.linspace(-7.0, 7.0, 500)
figure, axis = pyplot.subplots(figsize=(10, 7))
axis.set_xlabel(r"$u$")
axis.set_ylabel("PDF")
axis.set_ylim([0.0, 1.5])
axis.set_title("Normal Distribution")
axis.yaxis.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
σ = [0.3, 0.5, 1.0, 2.0]
μ = [-4.0, -2.0, 0.0, 2.0]
for i in range(len(σ)):
    normal_pdf = [stats.normal(j, σ[i], μ[i]) for j in x]
    axis.plot(x, normal_pdf, label=f"σ={σ[i]}, μ={μ[i]}")
axis.legend(bbox_to_anchor=(0.9, 0.95))
config.save_post_asset(figure, "bivariate_normal_distribution", "normal_distribution_parameters")

# %%
# Bivariate conditional distribution
σ1 = 1.0
σ2 = 1.0
μ1 = 0.0
μ2 = 0.0
γ = [-0.95, -0.5, 0.0, 0.5, 0.95]
x2 = 1.0

x1 = numpy.linspace(-4.0, 4.0, 500)
figure, axis = pyplot.subplots(figsize=(10, 7))
axis.set_xlabel(r"$v$")
axis.set_ylabel(r"$g(u|v)$")
axis.set_ylim([0.0, 1.5])
axis.set_title("Bivariate Conditional PDF: "+r"$v=$"+f"{format(x2, '2.1f')}")
axis.yaxis.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
for i in range(len(γ)):
    f = conditional_pdf_y1_y2(μ1, μ2, σ1, σ2, γ[i])
    axis.plot(x1, f(x1, x2), label=f"γ={γ[i]}")
axis.legend(bbox_to_anchor=(0.95, 0.9))
config.save_post_asset(figure, "bivariate_normal_distribution", "bivariate_conditional_pdf_correlation_scan")

# %%

σ1 = 1.0
σ2 = 1.0
μ1 = 0.0
μ2 = 0.0
γ = 0.5
x2 = [-2.0, -1.0, 0.0, 1.0, 2.0]

x1 = numpy.linspace(-4.0, 4.0, 500)
figure, axis = pyplot.subplots(figsize=(10, 7))
axis.set_xlabel(r"$u$")
axis.set_ylabel(r"$g(u|v)$")
axis.set_ylim([0.0, 0.65])
axis.yaxis.set_ticks([0.0, 0.2, 0.4, 0.6])
axis.set_title(f"Bivariate Conditional PDF: γ={γ}")
for i in range(len(x2)):
    f = conditional_pdf_y1_y2(μ1, μ2, σ1, σ2, γ)
    axis.plot(x1, f(x1, x2[i]), label=r"$v=$"+f"{format(x2[i], '2.1f')}")
axis.legend(bbox_to_anchor=(0.95, 0.95))
config.save_post_asset(figure, "bivariate_normal_distribution", "bivariate_conditional_pdf_y_scan")

# %%

def transform_plot(μ1, μ2, σ1, σ2, γ, xrange, yrange, xnudge, ynudge, unudge, anudge, legend, plot_name):
    npts = 500
    figure, axis = pyplot.subplots(figsize=(9, 9))
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_ylim(yrange)
    axis.set_xlim(xrange)
    title = f"Bivariate Normal Transformation: γ={format(γ, '2.1f')}, " + \
             r"$σ_u$=" + f"{format(σ1, '2.1f')}, " + r"$σ_v$=" + \
             f"{format(σ2, '2.1f')}"
    axis.set_title(title)

    ar = (yrange[1] - yrange[0]) / (xrange[1] - xrange[0])
    θ = (180.0 / numpy.pi) * numpy.arctan(-γ / (ar * numpy.sqrt(1-γ**2))) + anudge

    c = [-4.0, -2.0, 0.0, 2.0, 4.0]
    x1 = numpy.zeros(len(c))
    for i in range(len(c)):
        y2 = numpy.linspace(3.0 * yrange[0], 3.0 * yrange[1], npts)
        transform_y1 = pdf_transform_y1_constant(μ1, μ2, σ1, σ2, γ, c[i])
        x1_y1, x2_y1 = transform_y1(y2)
        x1[i] = x1_y1[0]
        axis.text(x1[i] - 0.35, unudge[i], f"u={format(c[i], '2.0f')}", fontsize=18, rotation=90.0)
        if i == 0:
            axis.plot(x1_y1, x2_y1, color="#0067C4", label=r"Constant $u$")
        else:
            axis.plot(x1_y1, x2_y1, color="#0067C4")

    for i in range(len(c)):
        y1 = numpy.linspace(3.0 * yrange[0], 3.0 * yrange[1], npts)
        transform_y2 = pdf_transform_y2_constant(μ1, μ2, σ1, σ2, γ, c[i])
        x1_y2, x2_y2 = transform_y2(y1)
        if i == len(c) - 1:
            xoffset = x1[i] - 2.0 * xnudge
        else:
            xoffset = x1[i] + xnudge
        x2 = (γ * xoffset - (c[i] - μ2) / σ2) / numpy.sqrt(1.0 - γ**2)
        axis.text(xoffset, x2 + ynudge[i], f"v={format(c[i], '2.0f')}", fontsize=18, rotation=θ)
        if i == 0:
            axis.plot(x1_y2, x2_y2, color="#FF9500", label=r"Constant $v$")
        else:
            axis.plot(x1_y2, x2_y2, color="#FF9500")

    axis.legend(bbox_to_anchor=legend)
    config.save_post_asset(figure, "bivariate_normal_distribution", plot_name)

# Bivariate transformation
# %%

σ1 = 1.0
σ2 = 1.0
μ1 = 0.0
μ2 = 0.0
γ = 0.95

transform_plot(μ1, μ2, σ1, σ2, γ, [-5.0, 5.0], [-10.0, 10.0], 0.0, [-1.0, -0.2, 0.5, 1.0, 2.0], [-4.0, -2.25, -1.5, -2.0, 5.0], 0.0, (0.55, 0.8), "bivariate_normal_transformation_correlation_0.95")

# %%

σ1 = 1.0
σ2 = 1.0
μ1 = 0.0
μ2 = 0.0
γ = 0.5

transform_plot(μ1, μ2, σ1, σ2, γ, [-5.0, 5.0], [-5.0, 5.0], 0.0, [-4.5, -2.15, 0.25, 2.5, 4.75], [1.5, -2.0, -1.0, -2., 3.5], 0.0, (0.4, 0.85), "bivariate_normal_transformation_correlation_0.5")

# %%

σ1 = 1.0
σ2 = 2.0
μ1 = 0.0
μ2 = 0.0
γ = 0.5

transform_plot(μ1, μ2, σ1, σ2, γ, [-5.0, 5.0], [-3.0, 3.0], 0.0, [0.1, 0.1, 0.1, 0.1, 0.1], [-1.65, -1.65, -1.5, -1.5, -1.5], 0.0, (0.5, 0.8), "bivariate_normal_transformation_correlation_0.5_sigma")

# %%

σ1 = 1.0
σ2 = 1.0
μ1 = 0.0
μ2 = 0.0
γ = 0.0

transform_plot(μ1, μ2, σ1, σ2, γ, [-5.0, 5.0], [-5.0, 5.0], 0.65, [0.15, 0.15, 0.15, 0.15, 0.15], [-0.75, -0.75, -0.75, -0.75, 1.25], 0.0, (0.85, 0.8), "bivariate_normal_transformation_correlation_0")

# %%

σ1 = 1.0
σ2 = 2.0
μ1 = 0.0
μ2 = 0.0
γ = 0.0

transform_plot(μ1, μ2, σ1, σ2, γ, [-5.0, 5.0], [-5.0, 5.0], 0.65, [0.15, 0.15, 0.15, 0.15, 0.15], [-3.0, -3.0, -3.0, -3.0, -3.0], 0.0, (0.6, 0.8), "bivariate_normal_transformation_correlation_0_sigma")
