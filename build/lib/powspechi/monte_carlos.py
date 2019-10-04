import numpy as np

def fconst(x):
    return 1.

def ffexp_folder(x):
    return 1/3.17739*np.cosh(0.5*(x - np.pi/2.))*((x - np.pi/2.)**2 + 1.)

# Sinfying a function :P:
def fct_sin(x, fct, *args, **kwargs):
    y = fct(x, *args, **kwargs)*np.sin(x)
    return y

# Function based on Flow Fourier expansion; order n
def fphi_psi(x, psis=None, vns=None):
    if psis is None:
        n = len(vns)
        psis = np.zeros(n)
    if vns is None:
        n = len(psis)
        vns = np.zeros(n)
    n = len(vns)
    psis = psis[:, np.newaxis]
    return 1/(2*np.pi)*(1. + 2*np.sum(vns*np.cos(np.arange(1, n+1)*(x - psis).T), axis=1))

def iso_dist(mult, etacut=None):

	angs = np.zeros((mult, 2))
	angs[:, 0] = np.random.uniform(0., 2*np.pi, size=mult)

	if etacut:

		qi = 2*np.arctan(np.exp(-etacut))
		qf = 2*np.arctan(np.exp(etacut))

		vi = (1 + np.cos(qf))/2.
		vf = (1 + np.cos(qi))/2.

		angs[:, 1] = np.arccos(2*np.random.uniform(vi, vf, size=mult) - 1)

	else:
		angs[:, 1] = np.random.uniform(size=mult)
		angs[:, 1] = np.arccos(2.*angs[:, 1] - 1.)

	return angs

def from_fct(fct, size, xmin=0., xmax=1., *args, **kwargs):

    samples = np.array([])
    ymax = np.max(fct(np.linspace(xmin, xmax, 1000), *args, **kwargs))
    
    ii = 0
    while ii == 0:
        siz0 = size - len(samples)
        ux = np.random.uniform(xmin, xmax, size=3*siz0)
        uy = np.random.uniform(0., 2*ymax, size=3*siz0)
    
        samples = np.append(samples, ux[uy <= fct(ux, *args, **kwargs)])
    
        if len(samples) >= size:
            samples = samples[:size]
            ii += 1
    
    return samples
