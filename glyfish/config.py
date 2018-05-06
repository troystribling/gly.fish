import os
import pathlib

style_file = os.path.join(os.getcwd(), 'gly.fish.mplstyle')
glyfish_style = pathlib.Path(style_file).as_uri()
