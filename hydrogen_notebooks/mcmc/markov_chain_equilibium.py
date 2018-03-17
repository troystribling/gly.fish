# %%
%load_ext autoreload
%autoreload 2

%aimport numpy
%aimport pygraphviz
%aimport sympy

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
    _, ncols = tpm.shape
    for xt1 in range(0, ncols):
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

def inv_cdf(π, x):
    intervals = []
    πlb = 0.0
    for i in range(0, len(π) - 1):
        intervals.append((i, sympy.Interval(πlb, π[i], False, True).contains(x)))
        πlb = π[i]
    intervals.append((len(π) - 1, sympy.Interval(πlb, 1.0, False, False).contains(x)))
    return sympy.Piecewise(*intervals)

# %%

nsamples = 100000
x0 = 1
chain_samples = sample_chain(p, x0, nsamples)

figure, axis = pyplot.subplots(figsize=(6, 5))
axis.set_xlabel("State")
axis.set_ylabel("PDF")
axis.set_title(f"Markov Chain PDF {nsamples} Samples")
axis.set_xlim([-0.5, 3.5])
axis.grid(True, zorder=5)
axis.set_xticks([0, 1, 2, 3])
_ = axis.hist(chain_samples - 0.5, [-0.5, 0.5, 1.5, 2.5, 3.5], density=True, color="#348ABD", alpha=0.6, label=f"Sampled Density", edgecolor="#348ABD", lw="3", zorder=10)


# %%

nsamples = 10000
x = sympy.symbols('x')
c = [[0.1],
     [0.5],
     [0.35],
     [0.05]]
π = numpy.matrix(c)

π_inv_cdf = inv_cdf(π, x)
x_values = [i / 100 for i in range(0, 101)]
π_values = [π_inv_cdf.subs(x, i) for i in x_values]
π_values[50]
figure, axis = pyplot.subplots(figsize=(6, 5))
axis.set_xlabel("State")
axis.set_ylabel("PDF")
axis.set_title(f"π PDF")
axis.set_xlim([-0.5, 3.5])
axis.set_xticks([0, 1, 2, 3])
axis.grid(True, zorder=5)
axis.bar([0, 1.0, 2.0, 3.0], [0.1, 0.5, 0.35, 0.05], 1.0, color="#A60628", label="π", alpha=0.6, lw="3", edgecolor="#A60628", zorder=10)

# %%

nsamples = 100
x = sympy.symbols('x')
c = [[0.1],
     [0.5],
     [0.35],
     [0.05]]
π = numpy.matrix(c)
π_inv_cdf = inv_cdf(π, x)
π_samples = [π_inv_cdf.subs(x, i) for i in numpy.random.rand(nsamples)]
figure, axis = pyplot.subplots(figsize=(6, 5))
axis.set_xlabel("State")
axis.set_ylabel("PDF")
axis.set_title(f"π Inverse CDF")
axis.set_xlim([-0.5, 3.5])
axis.grid(True, zorder=5)
axis.set_xticks([0, 1, 2, 3])
_ = axis.hist(π_samples, [-0.5, 0.5, 1.5, 2.5, 3.5], density=True, color="#348ABD", alpha=0.6, label=f"Sampled Density", edgecolor="#348ABD", lw="3", zorder=10)


# %%

π.T * (p * p)
(π.T * p) * p
