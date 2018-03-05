# %%
%load_ext autoreload
%autoreload 2

%aimport numpy
%aimport pygraphviz

from matplotlib import pyplot
from IPython.display import Image

%matplotlib inline

def draw(dot):
    return Image(pygraphviz.AGraph(dot).draw(format='png', prog='dot'))

# %%

g1 = """digraph markov_chain {
   size="5,6";
   ratio=fill;
   node[fontsize=24, fontname=Helvetica];
   edge[fontsize=24, fontname=Helvetica];
   graph[fontsize=24, fontname=Helvetica];
   labelloc="t";
   label="Markov Transition Matrix";
   1 -> 2 [label=" 0.9"];
   1 -> 3 [label=" 0.1"];
   2 -> 1 [label=" 0.8"];
   2 -> 2 [label=" 0.1"];
   2 -> 4 [label=" 0.1"];
   3 -> 2 [label=" 0.5"];
   3 -> 3 [label=" 0.3"];
   3 -> 4 [label=" 0.2"];
   4 -> 1 [label=" 0.1"];
   4 -> 4 [label=" 0.9"];
}"""
draw(g1)

# %%

t = [[0.0, 0.9, 0.1, 0.0],
     [0.8, 0.1, 0.0, 0.1],
     [0.0, 0.5, 0.3, 0.2],
     [0.1, 0.0, 0.0, 0.9]]
transition_matrix = numpy.array(t)
