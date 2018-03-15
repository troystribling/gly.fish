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
   0 -> 1 [label=" 0.9"];
   0 -> 2 [label=" 0.1"];
   1 -> 0 [label=" 0.8"];
   1 -> 1 [label=" 0.1"];
   1 -> 3 [label=" 0.1"];
   2 -> 1 [label=" 0.5"];
   2 -> 2 [label=" 0.3"];
   2 -> 3 [label=" 0.2"];
   3 -> 0 [label=" 0.1"];
   3 -> 3 [label=" 0.9"];
}"""
draw(g1)

# %%

t = [[0.0, 0.9, 0.1, 0.0],
     [0.8, 0.1, 0.0, 0.1],
     [0.0, 0.5, 0.3, 0.2],
     [0.1, 0.0, 0.0, 0.9]]
p = numpy.matrix(t)

# %%

def next_state(tpm, up, xt):
    txp = 0.0
    rows, cols = tpm.shape
    for xt1 in range(0, cols):
        txp += tpm[xt, xt1]
        if up <= txp:
            return xt1
    return None

def sample_chain(p, x0, nsample):
    xt = numpy.zeros(nsample, dtype=int)
    up = numpy.random.rand(nsample)
    xt[0] = x0
    for i in range(0, nsample - 1):
        xt1 = next_state(p, up[i], xt[i])
        if xt1 is None:
            continue
        xt[i + 1] = xt1
    return xt

chain_samples = sample_chain(p, 3, 100000) - 0.5

figure, axis = pyplot.subplots(figsize=(6, 5))
axis.set_xlabel("Sample")
axis.set_ylabel("PDF")
axis.set_title("Markov Chain")
axis.set_xlim([-0.5, 3.5])
axis.grid(True, zorder=5)
axis.set_xticks([0, 1, 2, 3])
_, bins, _ = axis.hist(samples, [-0.5, 0.5, 1.5, 2.5, 3.5], density=True, color="#348ABD", alpha=0.6, label=f"Sampled Density", edgecolor="#348ABD", lw="3", zorder=10)
