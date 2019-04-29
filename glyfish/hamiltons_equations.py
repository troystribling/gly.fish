import numpy

# %%
# Momentum Verlet integration of Hamiltons's equations
def momentum_verlet(p0, q0, dUdq, dKdp, nsteps, ε):
    ndim = len(p0)
    p = numpy.zeros((nsteps+1, ndim))
    q = numpy.zeros((nsteps+1, ndim))
    p[0] = p0
    q[0] = q0

    for i in range(nsteps):
        for j in range(ndim):
            p[i+1][j] = p[i][j] - ε*dUdq(q, i, j, True)/2.0
            q[i+1][j] = q[i][j] + ε*dKdp(p, i+1, j)
            p[i+1][j] = p[i+1][j] - ε*dUdq(q, i, j, False)/2.0

    return p, q

def bivariate_normal_U(γ, σ1, σ2):
    scale = σ1**2*σ2**2*(1.0 - γ**2)
    def f(q):
        return ((q[0]*σ2)**2 + (q[1]*σ1)**2 - 2.0*q[0]*q[1]*σ1*σ2*γ) / (2.0*scale)
    return f

def bivariate_normal_K(m1, m2):
    def f(p):
        return (p[0]**2/m1 + p[1]**2/m2) / 2.0
    return f

def bivariate_normal_dUdq(γ, σ1, σ2):
    scale = σ1**2*σ2**2*(1.0 - γ**2)
    def f(q, n, i, is_first_step):
        if i == 0:
            if is_first_step:
                return (q[n][0]*σ2**2 - q[n][1]*γ*σ1*σ2) / scale
            else:
                return (q[n+1][0]*σ2**2 - q[n][1]*γ*σ1*σ2) / scale
        elif i == 1:
            if is_first_step:
                return (q[n][1]*σ1**2 - q[n+1][0]*γ*σ1*σ2) / scale
            else:
                return (q[n+1][1]*σ1**2 - q[n+1][0]*γ*σ1*σ2) / scale
    return f

def bivariate_normal_dKdp(m1, m2):
    def f(p, n, i):
        if i == 0:
            return p[n][0]/m1
        elif i == 1:
            return p[n][1]/m2
    return f
