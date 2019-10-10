import numpy as np

def fconst(x):

    r"""Constant function.

    Parameters
    ----------
    x : float, array_like
        Independent variable.

    Returns
    -------
    1 : float, scalar
        Output value, always 1.

    """

    return 1.

def ffexp(x):

    r"""A function symmetric around :math:`\pi/2`, which is also the value
    associated with the function's minimum.

    Parameters
    ----------
    x : float, array_like
        Independent variable.

    Returns
    -------
    y : float, ndarray
        Output value whose shape is the same as `x`.

    """
    y = 1/3.17739*np.cosh(0.5*(x - np.pi/2.))*((x - np.pi/2.)**2 + 1.)
    return y

# Sinfying a function :P:
def fct_sin(x, sfct, *args, **kwargs):

    r"""Multiply a function by :math:`\sin{(x)}`.

    Parameters
    ----------
    x : float, array_like
        Independent variable.
    sfct : function
        The chosen function to be multiplied by :math:`\sin{(x)}`.
    *args 
        Arguments to be passed to `sfct`.
    **kwargs
        Keyword-only arguments to be passed to `sfct`.

    Returns
    -------
    y : float, ndarray
        Output value associated with multiplying `sfct` by :math:`\sin{(x)}`.
        Its shape is the same as `x`.

    Notes
    -----
    When picking a random polar point from a function defined on a sphere, 
    just using `sfct` will not yield the right result. It is therefore 
    necessary to pick a point from :math:`\sin{(x)}`*`sfct(x)`. For a more
    detailed explanation, check 'Sphere Point Picking' on the internet or
    a text book. 

    """

    y = sfct(x, *args, **kwargs)*np.sin(x)
    return y

# Function based on Flow Fourier expansion; order n
def fphi_psi(x, vns=None, psis=None):

    r"""A function (Fourier expansion) based on the flow ansatz used in
    heavy-ion studies [1]_ [2]_. It has the following expression:

    .. math:: h(x) = \frac{1}{2\pi} \left[ 1 + 2\sum_{n = 1}^{\infty} v_n \cos[n(x - \Psi_n)] \right],

    where :math:`v_n,\Psi_n` are the amplitude and phase, respectively.

    Parameters
    ----------
    x : float, array_like
        Independent variable.
    vns : float, array_like, optional
        Amplitudes of the Fourier expansion. If not given, it will be
        taken as zero. Default: *None*.
    psis : float, array_like, optional
        Phases of the Fourier expansion. If not given, it will be taken
        as zero. Default: *None*.

    Returns
    -------
    y : float, array_like
        Output of the function. It has the same shape as `x`.


    References
    ----------
    .. [1] S. Voloshin and Y. Zhang, "Flow study in relativistic nuclear collisions by Fourier expansion of azimuthal particle distributions", Z. Phys. **C70**, 665 (1996).
    .. [2] A.M. Poskanzer and S.A. Voloshin, "Methods for analyzing flow in relativistic nuclear collisions", Phys. Rev. **C58**, 1671 (1998).

    """

    if (vns is None) and (psis is None):
        return 1./(2.*np.pi)*np.ones_like(x)
    elif psis is None:
        n = len(vns)
        psis = np.zeros(n)
    elif vns is None:
        n = len(psis)
        vns = np.zeros(n)
    else:
        n = len(vns)
        psis = np.asarray(psis)
        psis = psis[:, np.newaxis]

    y = 1/(2*np.pi)*(1. + 2*np.sum(vns*np.cos(np.arange(1, n + 1)*(x - psis).T), axis=1))
    return y

def isodist(mult, etacut=None):

    r"""Create an isotropic particle distribution with specified 
    multiplicity and limitation in :math:`\theta`.

    Parameters
    ----------
    mult : int, scalar
        Multiplicity or size of the output sample.
    etacut : float, scalar, optional
        Limit imposed on pseudorapidity (or polar angle), i.e.,
        :math:`|\eta|` < `etacut`. Default: *None*.

    Returns
    -------
    angs : float, ndarray
        A 2-D array with shape (`mult`, 2). Under column 0 are the
        azimuthal angles, while under column 1 are the polar angles.

    Notes
    -----
    In order to pick a random number on the surface of a sphere, 
    azimuthal and polar angles are drawn, respectively, from the 
    following expressions:

    .. math:: \phi = 2\pi u,\\
    .. math:: \theta = \arccos{(2v - 1)},

    where :math:`u,v` are random variables within the interval (0, 1). 

    """

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

    r"""Draw random values according to a chosen 1-D function within a
    specified interval.

    Parameters
    ----------
    fct : function
        Chosen function.
    size : int, scalar
        Desired number of samples.
    xmin : float, scalar, optional
        Lower boundary of the interval. Default: 0.
    xmax : float, scalar, optional
        Upper boundary of the interval. Output values will be less than
        `xmax`. Default: 1.
    *args
        Arguments to be passed to `fct`.
    **kwargs
        Keyword-only arguments to be passed to `fct`.

    Returns
    -------
    samples : float, ndarray
        A 1-D array whose length is determined by `size`. It contains
        random values distributed according to `fct`.

    """

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
