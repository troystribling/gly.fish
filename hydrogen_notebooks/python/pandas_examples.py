# %%
import pandas
import numpy

# %% series
series = pandas.Series(numpy.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])
series.index

series = pandas.Series(numpy.random.randn(5))

d = {'a': 0.0, 'b': 1.0, 'c': 2.0}
series = pandas.Series(d)
series[0]

series = pandas.Series(d, index=['b', 'c', 'd', 'a'])
pandas.isnull(series['d'])
series['d'] is numpy.nan
series['b']
series[0]

'd' in series
'g' in series

try:
    x = series['g']
except KeyError:
    print("key missing")

# %%
d = {'one': pandas.Series([1., 2., 3.], index=['a', 'b', 'c']),
     'two': pandas.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])}
dataFrame = pandas.DataFrame(d)

dataFrame = pandas.DataFrame(d, index=['c', 'b', 'd'])

dataFrame = pandas.DataFrame(d, index=['d', 'b', 'a'], columns=['two', 'three'])

data = numpy.zeros((2,), dtype=[('A', 'i4'), ('B', 'f4'), ('C', 'a10')])
data[:] = [(1, 2., 'Hello'), (2, 3., "World")]
dataFrame = pandas.DataFrame(data)

data = [{'a': 1, 'b': 2}, {'a': 5, 'b': 10, 'c': 20}]
dataFrame = pandas.DataFrame(data)
dataFrame = pandas.DataFrame(data, index=['first', 'second'])
