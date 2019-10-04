import numpy as np
import healpy as hp
from scipy.special import lpmv
from scipy.integrate import quad
from math import factorial
from powspechi.monte_carlos import fconst

def lns(nside):

    r"""Create a multipole (:math:`\ell`) array based on the chosen resolution.

    Parameters 
    ----------
    nside: int scalar
        A parameter related to the chosen HEALPix map resolution

    Returns
    -------
    l : int ndarray
        A 1D array of int type that satisfies the chosen nside

    """

    nlmax = 3*nside - 1
    l = np.arange(nlmax + 1)
    
    return l

def cld_from_maps(maps):

    r"""Calculate the angular power spectrum of a given map or maps.

    Parameters
    ----------
    maps : array_like
        A single map or array/list of maps. It must be a HEALPix map, i.e.,
        the number of indices must correspond to a nside value.

    Returns
    -------
    cld : dict
        A dictionary whose keys correspond to the *full* power spectrum
        and the same without the :math:`a_{\ell 0}` modes, denoted *mdz*. The 
        values of **cld** are ndarrays with dimentions dependent on the number 
        of entry maps and their resolution.
    averd : dict
        If more than one map is given, the averaged power spectrum is calculated.
        Its keys are also *full* and *mdz*. Its values are lists of arrays: index
        0 corresponds to the mean **cld** value, while index 1 is the error on the 
        mean.

    Notes
    -----
    A *full* angular power spectrum has the following expression:
    
    .. math:: C_{\ell} = \frac{1}{2\ell + 1}\sum_{m = -\ell}^{m = \ell} |a_{\ell m}|^2,

    while *mdz*, which stands for :math:`m\neq0` has the form

    .. math:: C^{m\neq0}_{\ell} = C_{\ell} - \frac{1}{2\ell + 1} |a_{\ell 0}|^2,

    :math:`a_{\ell m}` are the coefficients associated with the spherical harmonics :math:`Y_{\ell m}`.

    """

    if maps[0].ndim == 0:
        nevts, nside = 1, hp.get_nside(maps)
        maps = [maps]
    else:
        nevts, nside = len(maps), hp.get_nside(maps[0])

    js = np.arange(3*nside)
    cld = {'full': np.zeros((nevts, 3*nside)), 'mdz': np.zeros((nevts, 3*nside-1))}
    ii = 0

    for emap in maps:
        cld['full'][ii, :], alms = hp.anafast(emap, alm=True)
        c0s = 1./(2.*js + 1.)*np.abs(alms[js])**2
        cld['mdz'][ii, :] = cld['full'][ii, 1:] - c0s[1:]
        ii += 1

    if nevts != 1:
        averd = {}
        for key in cld.keys():
            averd[key] = [np.mean(cld[key], axis=0), np.std(cld[key], axis=0, ddof=1)/np.sqrt(nevts)]
        return cld, averd
    else:
        for key in cld.keys():
            cld[key] = cld[key][0]
        return cld

# Correction by N_lm subtraction: two functions
def iso_background(clsres_file, skip=True):

    r"""From a special type of file create a dictionary containing :math:`\langle N_{\ell} \rangle`, 
    i.e., an average power spectrum used to correct for the ensemble multiplicity.

    Parameters
    ----------
    clsres_file : string
        A file containing the average power spectrum :math:`\langle N_{\ell} \rangle`. It has four 
        columns which follow the order: *full*  *err_full*  *mdz*   *err_mdz*. Refer to **cld_from_maps**
        to see the meaning of *full* and *mdz*. As for *err_*, it indicates the error on the mean of
        its corresponding spectrum.
    skip : bool, optional
        If *True* it skips the first line of the file, whereas if set to *False* no line will be skipped.
        Default: *True*.

    Returns
    -------
    clsres : dict
        A dictionary with keys *full* and *mdz*, whose values are lists with the full spectrum and the same when
        :math:`m\neq0`. For each of these lists, the index 0 contains the mean, while index 1 contains the error
        on the mean. Both quantities are ndarrays.

    Notes
    -----
    While the correction for the :math:`m\neq0` average spectrum is simply :math:`\langle N^{m\neq0}_{\ell} \rangle`,
    in the case of the full spectrum, even and odd :math:`\ell` modes are treated differently. Odd modes are corrected
    with odd :math:`\langle N_{\ell} \rangle`, whreas even modes are corrected with even :math:`\langle N^{m\neq0}_{\ell} \rangle`.
    The reason lies in considering the artificial spectrum features which arise from limited sky maps. If :math:`\langle C_{\ell} \rangle`
    is simply subtracted by :math:`\langle N_{\ell} \rangle`, then such features will disapear, thus the resulting spectrum
    will not faithfully reproduce the expected full spectrum under said circumstances.

    """

    clsres = {}
    if skip:
        vals = np.genfromtxt(clsres_file, skip_header=1)
    else:
        vals = np.genfromtxt(clsres_file)
    clsres['mdz'] = [vals[1:, 2], vals[1:, 3]]
    vals2 = np.copy(vals)
    vals2[0::2, 0] = vals[0::2, 2]   
    vals2[0::2, 1] = vals[0::2, 3] # The odd full modes should be corrected by the iso odd full modes
    clsres['full'] = [vals2[:, 0], vals2[:, 1]]

    return clsres

def subiso_corr(averd, iso_bkg):

    r"""Subtracts the average spectrum calculated through HEALPix :math:`\langle C_{\ell} \rangle` from the 
    spectrum of ensemble multiplicity :math:`\langle N_{\ell}\rangle`.

    Parameters
    ----------
    averd : dict
        A dictionary containing the power spectra :math:`\langle C_{\ell} \rangle` and :math:`\langle C^{m\neq0}_{\ell}`.
        They should be contained in a list with index 0 for the mean and index 1 for its error. Such lists should be
        values corresponding to different keys. Their recommended names are *full* and *mdz*, respectively.
    iso_bkg : dict
        A dictionary following the same format, i.e., same keys and list types, as **averd**. It should contain the
        averaged spectrum used to correct for the ensemble's multiplicity distribution, :math:`\langle N_{\ell} \rangle`.

    Returns
    -------
    averd_sic : dict
        A dictionary following the same format as **averd**. It contains the corrected averaged spectra :math:`\langle S_{\ell}\rangle`
        and :math:`\langle S^{m\neq0}_{\ell}\rangle`, as well as their propagated error.   

    """

    averd_sic = {}

    for key in averd.keys():
        averd_sic[key] = [averd[key][0] - iso_bkg[key][0], np.sqrt(averd[key][1]**2 + iso_bkg[key][1]**2)]

    return averd_sic

# Averaging over vertices -> nevts should be a dictionary:
def av_over_zvtx(avcls, nevts):

    """Calculates the weighted average of spectra from distinct event ensembles.

    

    """

    mean_zvtx = {}
    for key in ['full', 'mdz']:
        mean_zvtx_k = np.average([np.abs(avcls[vtx][key][0]) for vtx in nevts.keys()], axis=0, weights=[nevts[vtx] for vtx in nevts.keys()])    
        std_zvtx = np.sqrt(np.sum([(nevts[vtx]*avcls[vtx][key][1])**2 for vtx in nevts.keys()], axis=0))
        sumw = np.sum([nevts[vtx] for vtx in nevts.keys()])
        mean_zvtx[key] = [mean_zvtx_k, std_zvtx/sumw]
    return mean_zvtx

# Calculates the alm coefficients of a function f(theta, phi) ~ g(theta)*h(phi)
# When vns is set to an array of ones and psis to zero, one can get the blm
# vns is an array with v_n values beginning with v_1, even if that is zero
def alm_dNdphi(l, m, eta_cut=0.9, vns=np.ones(4), psis=np.zeros(4), g_sim=fconst, *args, **kwargs):

    if eta_cut:
        qi, qf = 2.*np.arctan(np.exp(-np.array([eta_cut, -eta_cut])))
    else:
        qi, qf = 0., np.pi

    n = len(vns)

    if m > n:
        a_lm = 0.+0.j
    else:
        c0 = 1./np.sqrt(4.*np.pi)*quad(lambda theta: np.sin(theta)*g_sim(theta, *args, **kwargs), qi, qf)[0] # Sets a_00**2 = 4*pi
        b_lm = np.sqrt(4.*np.pi)/c0*np.sqrt((2.*l + 1.)/(4.*np.pi)*factorial(l - m)/factorial(l + m))*quad(lambda theta: np.sin(theta)*
            g_sim(theta, *args, **kwargs)*lpmv(m, l, np.cos(theta)), qi, qf)[0]

        if m == 0:
            a_lm = b_lm
        else:
            a_lm = b_lm * vns[m - 1] * np.exp(-1.j*n*psis[m - 1])
    
    return a_lm

# Calculates Cl analytically for certain alm coefficients until lsize
def cls_calc(lsize, alms, *args, **kwargs):
    cls_true = {'full': np.zeros(lsize), 'mdz': np.zeros(lsize)}

    for l in range(lsize):
        cls_true['full'][l] = np.abs(alms(l, 0, *args, **kwargs))**2
        c0s = np.copy(cls_true['full'][l])
        for m in range(1, l+1):
            cls_true['full'][l] += 2.*np.abs(alms(l, m, *args, **kwargs))**2
        
        cls_true['mdz'][l] = cls_true['full'][l] - c0s
        for key in cls_true.keys():
            cls_true[key][l] /= (2.*l + 1.)

    cls_true['mdz'] = cls_true['mdz'][1:]

    return cls_true

# Vn calculation - Cl: let mixed be True if you want to consider the mixed alm modes like a31 and a42 in the calculation
def vns_calc(n, averd, blms, mixed=True):
    bnn = blms[n]
    b00 = blms[0]
    Cn = averd['mdz'][0][n - 1]
    errCn = averd['mdz'][1][n - 1]
    C0 = averd['full'][0][0]
    errC0 = averd['full'][1][0]
    
    vn = np.sqrt((2.*n + 1.)*Cn*np.abs(b00)**2/(2.*np.abs(bnn)**2*C0))
    err = vn/2.*np.sqrt((errCn/Cn)**2 + (errC0/C0)**2)

    if mixed and (n == 3 or n == 4):
        bn_ = blms[n+2]
        b__ = blms[n-2]
        C_ = averd['mdz'][0][n-3]
        errC_ = averd['mdz'][1][n-3]
        v_ = np.sqrt((2.*(n-2) + 1.)*C_*np.abs(b00)**2/(2.*np.abs(b__)**2*C0))
        err_ = v_/2.*np.sqrt((errC_/C_)**2 + (errC0/C0)**2)
        
        vn = np.sqrt(((2.*n + 1.)*Cn/C0*np.abs(b00)**2 - 2.*np.abs(bn_)**2*v_**2)/(2.*np.abs(bnn)**2))
        a = ((2.*n + 1.)*np.abs(b00)**2)/(2.*np.abs(bnn)**2)
        b = np.abs(bn_)**2/np.abs(bnn)**2
        err = 1./(2.*vn)*np.sqrt((a*Cn/C0)**2*((errCn/Cn)**2 + (errC0/C0)**2) + (2*b*v_*err_)**2)
    
    return vn, err
