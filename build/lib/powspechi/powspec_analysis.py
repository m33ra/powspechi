import numpy as np
import healpy as hp
from powspechi.maps_manip import mapping, make_normmaps
from powspechi.powspec_calc import maps2cld, isobackground, subisocorr
import os
import powspechi.pserrors as pserr

class powspec_analysis():

    r"""Perform the full analysis and extract the final angular power spectrum 
    :math:`\langle S_{\ell}^{m \neq 0} \rangle`. 

    Parameters
    ----------
    nside : int, scalar
        Chosen map resolution.
    angs : float, array_like
        List or array of angular coordinates of all events. Each of said events 
        should be a 2-D array with shape (length, 2), where columns 0 and 1 stand 
        for the azimuthal :math:`\phi` and polar :math:`\theta` coordinates, 
        respectively.
    etacut : float, scalar, optional
        Limit imposed on pseudorapidity, i.e., :math:`|\eta|` < `etacut`. If there
        is no limit, just set it to *None*. Default: 0.9.
    detcorr : bool, optional
        Determines whether to divide the observed maps by their ensemble sum.
        Usually used when there is a need to correct for detector's non-uniform
        efficiency or to compare a heavy-ion model to the published angular power
        spectrum [1]_. Default: *False*.
    multcorr_norm : filenamestr, optional
        File containing the averaged spectrum which corrects for event multiplicity
        the normalized spectrum. Format: four columns respectively representing the
        full averaged spectrum, its mean error, the averaged spectrum for :math:`m\neq0`
        and its mean error. First line will be skipped. Also, it can only be activated
        if `detcorr` = *True*. Default: *None*. 
    multcorr_obs : filenamestr, optional
        File containing the averaged spectrum which corrects for event multiplicity
        the observed spectrum. Format is the same as in `multcorr_norm`. Default: *None*.

    Attributes
    ----------
    tmap : float, ndarray
        Sum of all maps within the event ensemble.
    avcl_obs : dict
        Observed angular power spectrum averaged over all events. Its keys indicate
        the full spectrum and its mean error as well as the :math:`m\neq0` spectrum
        and its mean error. 
    avcl_norm : dict, optional
        Normalized angular power spectrum averaged over all events. Its format is the
        same as `avcl_obs`. Activated when `detcorr` = *True*.
    avcl_normcorr : dict, optional
        Normalized angular power spectrum corrected by event multiplicity. Its format
        is the same as `avcl_obs`. Activated when both `detcorr` = *True* and a file
        is given as `multcorr_norm`.
    avcl_obscorr : dict, optional
        Observed angular power spectrum corrected by event multiplicity. Its format is
        the same as `avcl_obs`. Activated when a file is given as `multcorr_obs`.

    Raises
    ------
    SupmapError
        When one wants to divide a single map by itself.
    IsomapError
        When the desired `supmap_iso*.fits` file does not exist. See ``maps_manip.getsupmapiso``
        for more details.
    PowSpecError
        When one tries to correct a normalized spectrum when it does not exist.

    See Also
    --------    
    maps_manip.mapping : Makes HEALPix maps.
    maps_manip.make_normmaps : Divide maps by a single map, usually the ensemble sum.
    powspec_calc.maps2cld : Calculate the angular power spectrum from maps.
    powspec_calc.isobackground : Make a dictionary out of a spectrum file for correction.
    powspec_calc.subisocorr : Subtract a spectrum from another.

    References
    ----------
    .. [1] M. Machado, "Heavy ion anisotropies: a closer look at the angular power spectrum", arXiv:1907.00413 [hep-ph] (2019).

    """

    #The constructor for the class ``powspec_analysis``.
    def __init__(self, nside, angs, etacut=0.9, detcorr=False, multcorr_norm=None, multcorr_obs=None):
       
        if angs[0].ndim == 1: # if angs is alone
            angs = [angs]
            flagd = True
        
        npix = hp.nside2npix(nside)
        tmap = []
        Fthetaphi = np.zeros(npix)
        for angi in angs:
            tmap.append(mapping(nside, angi))
            Fthetaphi += tmap[-1] / npix
        Fthetaphi *= npix / np.sum(Fthetaphi)
        
        if 'flagd' in locals():
            Clobs = maps2cld(tmap)
        else:
            clds_obs, Clobs = maps2cld(tmap)
        
        self.tmap = Fthetaphi 
        self.avcl_obs = Clobs 
        
        # Detector efficiency corrections or simple division by g(theta)
        
        if detcorr and 'flagd' in locals():
            raise pserr.SupmapError('In the case of a single map, it makes no sense to divide it by itself.')
            
        elif detcorr and etacut:
            curr_dir = os.path.dirname(__file__)
            det_file = os.path.join(curr_dir, 'supmaps_iso/supmap_iso%s_ns%d.fits' %(etacut, nside))

            if os.path.isfile(det_file):
                supmapiso = hp.read_map(det_file, verbose=False)
                pixs = np.nonzero(supmapiso)
                supmap = np.copy(Fthetaphi)
                supmap[pixs] /= supmapiso[pixs]
                supmap *= npix / np.sum(supmap)
            
                # Making the f_bar maps:
                tmap_norm = make_normmaps(tmap, supmap, etacut)
                clds_norm, Clnorm = maps2cld(tmap_norm)
            
                del tmap
                del tmap_norm
            
                self.avcl_norm = Clnorm
                
            else:
                raise pserr.IsomapError('The desired supmap_iso*.fits file with nside = %d and |eta| < %s does not exist. Please refer to documentation.' %(nside, etacut))

        elif detcorr and etacut is None:
            # Making the f_bar maps:
            tmap_norm = make_normmaps(tmap, Fthetaphi, None)
            clds_norm, Clnorm = maps2cld(tmap_norm)
            
            del tmap
            del tmap_norm
            
            self.avcl_norm = Clnorm
                    
        # Average power spectrum correction by multiplicity 
        
        # If one wants to correct the modified average spectrum:
        if multcorr_norm and detcorr:
            Slnorm = subisocorr(Clnorm, isobackground(multcorr_norm))
            self.avcl_normcorr = Slnorm

        elif multcorr_norm:
            raise pserr.PowSpecError('The averaged normalized spectrum does not exist.')

        # If one wants to correct the original average spectrum:
        if multcorr_obs:
            Slobs = subisocorr(Clobs, isobackground(multcorr_obs))
            self.avcl_obscorr = Slobs # The code follows the notation of the paper "Heavy ion anisotropies: a closer look at the angular power spectrum", though not the attributes.