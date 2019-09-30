import numpy as np
import healpy as hp
from powspecHI.maps_manip import mapping, make_modf_maps
from powspecHI.powspec_calc import cld_from_maps, iso_background, subiso_corr
import os
import powspecHI.pserrors

class powspec_analysis():

    """ A class to perform the full analysis and extract the final <Sl^{m \= 0}> """

    def __init__(self, nside, angs, eta_cut=0.9, multcorr_modf=None, multcorr_raw=None, detcorr=False):
        if angs[0].ndim == 1: # if angs is alone
            angs = [angs]
            flagd = True
        
        npix = hp.nside2npix(nside)
        tmap = []
        Fthetaphi = np.zeros(npix)
        for angi in angs:
            tmap.append(mapping(nside, angi))
            Fthetaphi += tmap[-1] / npix
        #del angs
        Fthetaphi *= npix / np.sum(Fthetaphi)
        
        if 'flagd' in locals():
            Clraw = cld_from_maps(tmap)
        else:
            clds_raw, Clraw = cld_from_maps(tmap)
        
        self.tmap = Fthetaphi 
        self.Clraw = Clraw 
        
        """
            Detector efficiency corrections or simple division by g(theta)
        """
        
        if detcorr and 'flagd' in locals():
            raise SupmapError('In the case of a single map, it makes no sense to divide it by itself.')
            
        elif detcorr:
            det_file = 'supmaps_iso/supmap_iso%s_ns%d.fits' %(eta_cut, nside)
            if os.path.isfile(det_file):
                supmapiso = hp.read_map(det_file, verbose=False)
                pixs = np.nonzero(supmapiso)
                supmap = np.copy(Fthetaphi)
                supmap[pixs] /= supmapiso[pixs]
                supmap *= npix / np.sum(supmap)
            
                # Making the f_bar maps:
                tmap_modf = make_modf_maps3(tmap, supmap, eta_cut)
                clds_modf, Clmodf = cld_from_maps(tmap_modf)
            
                del tmap
                del tmap_modf
            
                self.Clmodf = Clmodf
                
            else:
                raise IsomapError('The desired supmap_iso file with nside = %d and |eta| < %s does not exist. Please refer to documentation.' %(nside, eta_cut))
                    
        """
            Average power spectrum correction by multiplicity 
        """
        
        # If one wants to correct the original average spectrum:
        if multcorr_raw:
            Slraw = subiso_corr(Clraw, iso_background(multcorr_raw))
            self.Slraw = Slraw
        
        # If one wants to correct the modified average spectrum:
        if multcorr_modf and detcorr:
            Slmodf = subiso_corr(Clmodf, iso_background(multcorr_modf))
            self.Slmodf = Slmodf
        elif multcorr_modf:
            raise PowSpecError('The averaged normalized spectrum does not exist.')