# %%

import numpy
from matplotlib import pyplot
from scipy import stats

%matplotlib inline

# %%

def pdf_plot(pdfs, labels, x, title):
    figure, axis = pyplot.subplots(figsize=(12, 5))
    axis.set_xlabel("Sample", fontsize=14)
    axis.tick_params(labelsize=13)
    axis.set_ylabel("PDF", fontsize=14)
    axis.set_title(title, fontsize=15)
    axis.grid(True, zorder=5)
    for i in range(len(pdf)):
        axis.plot(pdfs[i], x, color="#A60628", label=labels[i], lw="3", zorder=10)
    axis.legend(fontsize=13)

# %%
