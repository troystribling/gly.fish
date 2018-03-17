# %%
%load_ext autoreload
%autoreload 2

%aimport numpy

from matplotlib import pyplot
from scipy import stats
from sympy import Function, Symbol, S, oo, I, sin, Heaviside, Piecewise, Interval, plot, log

%matplotlib inline

# %%

x = Symbol('x')
f = Function('f')
g = Function('g')(x)
f
f(x)
g
f(x).diff(x)
g.diff(x)

# %%


class h(Function):

    @classmethod
    def eval(cls, x):
        if x.is_Number:
            if x is S.Zero:
                return S.One
            elif x is S.Infinity:
                return S.Zero

    def _eval_is_real(self):
        returnself.args[0].is_real


h(0)
h(oo)


# %%

plot(x*x, (x, 0, 6))

# %%

H = Heaviside(x - 2)
H.subs(x, 1)
plot(H)


# %%
y = Symbol('y')
f = y**2
g = log(y)
p = Piecewise((99, y < -1), (f, y <= 1), (g, True))
[p.subs(y, i) for i in range(-20, 20)]
