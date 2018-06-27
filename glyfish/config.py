import os
import pathlib
from cycler import cycler

style_file = os.path.join(os.getcwd(), 'gly.fish.mplstyle')
glyfish_style = pathlib.Path(style_file).as_uri()
post_asset_path = os.path.join(os.getcwd(), 'assets', 'posts')

def save_post_asset(figure, post, plot):
    path = os.path.join(post_asset_path, post, plot) + ".png"
    figure.savefig(path, bbox_inches="tight")

distribution_sample_cycler = cycler("color", ["#329EFF", "#320075"])
bar_plot_cycler = cycler("color", ["#329EFF", "#FF9500"])
