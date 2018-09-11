import os
import pathlib
import pygraphviz
import matplotlib

from IPython.display import Image
from cycler import cycler

def save_post_asset(figure, post, plot):
    path = os.path.join(post_asset_path, post, plot) + ".png"
    figure.savefig(path, bbox_inches="tight")

def draw(dot, post, plot):
    path = os.path.join(post_asset_path, post, plot) + ".png"
    pygraphviz.AGraph(dot).draw(path, format='png', prog='dot')
    return Image(pygraphviz.AGraph(dot).draw(format='png', prog='dot'))

color = matplotlib.colors.ColorConverter().to_rgb

color("#329EFF")
color("#FFE800")
color("#FF9500")


histogram_color_map_cdict = {'red':   ((0.0,  1.0, 1.0),
                                       (0.5,  0.19, 0.19),
                                       (0.75, 1.0, 1.0),
                                       (1.0,  1.0, 1.0)),

                             'green': ((0.0,  1.0, 1.0),
                                       (0.5, 0.62, 0.62),
                                       (0.75, 0.58, 0.58),
                                       (1.0,  0.91, 0.91)),

                             'blue':  ((0.0,  1.0, 1.0),
                                       (0.5,  1.0, 1.0),
                                       (0.75, 0.0, 0.0),
                                       (1.0,  0.0, 0.0))
                             }
histogram_color_map = matplotlib.colors.LinearSegmentedColormap('HistogramMap', histogram_color_map_cdict)

style_file = os.path.join(os.getcwd(), 'gly.fish.mplstyle')
glyfish_style = pathlib.Path(style_file).as_uri()
post_asset_path = os.path.join(os.getcwd(), 'assets', 'posts')

contour_color_map = matplotlib.colors.ListedColormap(["#0067C4", "#FFE800", "#320075", "#FF9500",
                                                      "#329EFF", "#AC9C00", "#5600C9", "#FFC574",
                                                      "#003B6F", "#FFEB22", "#8C35FF", "#AC6500"])
distribution_sample_cycler = cycler("color", ["#329EFF", "#320075"])
alternate_cycler = cycler("color", ["#0067C4", "#8C35FF", "#FF9500", "#FFE800", "#329EFF", "#FFC574", "#320075"])
bar_plot_colors = ["#0067C4", "#FF9500", "#320075", "#FFE800", "#329EFF", "#FFC574", "#8C35FF"]
bar_plot_cycler = cycler("color", bar_plot_colors)
