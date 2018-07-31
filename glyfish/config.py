import os
import pathlib
import pygraphviz
from IPython.display import Image
from cycler import cycler

style_file = os.path.join(os.getcwd(), 'gly.fish.mplstyle')
glyfish_style = pathlib.Path(style_file).as_uri()
post_asset_path = os.path.join(os.getcwd(), 'assets', 'posts')

def save_post_asset(figure, post, plot):
    path = os.path.join(post_asset_path, post, plot) + ".png"
    figure.savefig(path, bbox_inches="tight")

distribution_sample_cycler = cycler("color", ["#329EFF", "#320075"])
bar_plot_cycler = cycler("color", ["#0067C4", "#FF9500", "#320075"])

def draw(dot, post, plot):
    path = os.path.join(post_asset_path, post, plot) + ".png"
    pygraphviz.AGraph(dot).draw(path, format='png', prog='dot')
    return Image(pygraphviz.AGraph(dot).draw(format='png', prog='dot'))
