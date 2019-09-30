import numpy as np
import healpy as hp
from scipy.special import lpmv
from scipy.integrate import quad
from math import factorial

def lns(nside):

    nlmax = 3*nside - 1
    l = np.arange(nlmax + 1)
    
    return l

# Calculate power spectrum from given map dictionary with keys equal to event number
def cld_from_maps(maps):

    if maps[0].ndim == 0:
        nevts, nside = 1, hp.get_nside(maps)
        maps = [maps]
    else:
        nevts, nside = len(maps), hp.get_nside(maps[0])

    js = np.arange(3*nside)
    cld = {'norm': np.zeros((nevts, 3*nside)), 'mdz': np.zeros((nevts, 3*nside-1))}
    ii = 0

    for emap in maps:
        cld['norm'][ii, :], alms = hp.anafast(emap, alm=True)
        c0s = 1./(2.*js + 1.)*np.abs(alms[js])**2
        cld['mdz'][ii, :] = cld['norm'][ii, 1:] - c0s[1:]
        ii += 1

    if nevts != 0:
        averd = {}
        for key in cld.keys():
            averd[key] = [np.mean(cld[key], axis=0), np.std(cld[key], axis=0, ddof=1)/np.sqrt(nevts)]
        return cld, averd
    else:
        for key in cld.keys():
            cld[key] = cld[key][0]
        return cld

# Correction by iso subtraction:
def iso_background(clsres_file):
    clsres = {}
    vals = np.genfromtxt(clsres_file, skip_header=1)
    clsres['mdz'] = [vals[1:, 2], vals[1:, 3]]
    vals2 = np.copy(vals)
    vals2[0::2, 0] = vals[0::2, 2]   
    vals2[0::2, 1] = vals[0::2, 3] # The odd full modes should be corrected by the iso odd full modes
    clsres['norm'] = [vals2[:, 0], vals2[:, 1]]

    return clsres

def subiso_corr(averd, iso_bkg):
    averd_sic = {}

    for key in averd.keys():
        averd_sic[key] = [averd[key][0] - iso_bkg[key][0], np.sqrt(averd[key][1]**2 + iso_bkg[key][1]**2)]

    return averd_sic

# Averaging over vertices -> nevts should be a dictionary:
def av_over_zvtx(avcls, nevts):
    mean_zvtx = {}
    for key in ['norm', 'mdz']:
        mean_zvtx_k = np.average([np.abs(avcls[vtx][key][0]) for vtx in nevts.keys()], axis=0, weights=[nevts[vtx] for vtx in nevts.keys()])    
        std_zvtx = np.sqrt(np.sum([(nevts[vtx]*avcls[vtx][key][1])**2 for vtx in nevts.keys()], axis=0))
        sumw = np.sum([nevts[vtx] for vtx in nevts.keys()])
        mean_zvtx[key] = [mean_zvtx_k, std_zvtx/sumw]
    return mean_zvtx

# The blm coefficients from a function isotropic in theta
def blm(l, m, cut=0.9):
    if cut:
        qi, qf = 2.*np.arctan(np.exp(-np.array([cut, -cut])))
    else:
        qi, qf = 0., 2*np.pi

    b_lm = 4*np.pi/(np.cos(qi) - np.cos(qf))*np.sqrt((2.*l+1.)/(4*np.pi)*factorial(l - m)/factorial(l + m))*quad(lambda theta: np.sin(theta)*
        lpmv(m, l, np.cos(theta)), qi, qf)[0]

    return b_lm

# Calculates Cl analytically for certain alm coefficients until lsize
def cls_calc(lsize, alms, args=[]):
    cls_true = {'norm': np.zeros(lsize), 'mdz': np.zeros(lsize)}

    for l in range(lsize):
        cls_true['norm'][l] = np.abs(alms(l, 0, *args))**2
        c0s = np.copy(cls_true['norm'][l])
        for m in range(1, l+1):
            cls_true['norm'][l] += 2.*np.abs(alms(l, m, *args))**2
        
        cls_true['mdz'][l] = cls_true['norm'][l] - c0s
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
    C0 = averd['norm'][0][0]
    errC0 = averd['norm'][1][0]
    
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
