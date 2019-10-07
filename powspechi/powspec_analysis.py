import numpy as np
import healpy as hp
from powspechi.maps_manip import mapping, make_modfmaps
from powspechi.powspec_calc import maps2cld, isobackground, subisocorr
import os
import powspechi.pserrors as pserr

class powspec_analysis():

    r""" A class to perform the full analysis and extract the final :math:`\langle S_{\ell}^{m \neq 0} \rangle` """

    def __init__(self, nside, angs, etacut=0.9, multcorr_modf=None, multcorr_raw=None, detcorr=False):
        
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
            Clraw = maps2cld(tmap)
        else:
            clds_raw, Clraw = maps2cld(tmap)
        
        self.tmap = Fthetaphi 
        self.Clraw = Clraw 
        
        # Detector efficiency corrections or simple division by g(theta)
        
        if detcorr and 'flagd' in locals():
            raise pserr.SupmapError('In the case of a single map, it makes no sense to divide it by itself.')
            
        elif detcorr:
            curr_dir = os.path.dirname(__file__)
            det_file = os.path.join(curr_dir, 'supmaps_iso/supmap_iso%s_ns%d.fits' %(etacut, nside))
            if os.path.isfile(det_file):
                supmapiso = hp.read_map(det_file, verbose=False)
                pixs = np.nonzero(supmapiso)
                supmap = np.copy(Fthetaphi)
                supmap[pixs] /= supmapiso[pixs]
                supmap *= npix / np.sum(supmap)
            
                # Making the f_bar maps:
                tmap_modf = make_modfmaps(tmap, supmap, etacut)
                clds_modf, Clmodf = maps2cld(tmap_modf)
            
                del tmap
                del tmap_modf
            
                self.Clmodf = Clmodf
                
            else:
                raise pserr.IsomapError('The desired supmap_iso*.fits file with nside = %d and |eta| < %s does not exist. Please refer to documentation.' %(nside, etacut))
                    
        # Average power spectrum correction by multiplicity 
        
        # If one wants to correct the original average spectrum:
        if multcorr_raw:
            Slraw = subisocorr(Clraw, isobackground(multcorr_raw))
            self.Slraw = Slraw
        
        # If one wants to correct the modified average spectrum:
        if multcorr_modf and detcorr:
            Slmodf = subisocorr(Clmodf, isobackground(multcorr_modf))
            self.Slmodf = Slmodf
        elif multcorr_modf:
            raise pserr.PowSpecError('The averaged normalized spectrum does not exist.')