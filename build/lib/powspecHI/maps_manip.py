import numpy as np
import healpy as hp

def readfile_phi_theta(infile):
    
    data = []

    with open(infile, 'r') as f:
        f.readline()
        for line in f:
            data.append(line.split())

    return np.asarray(data, dtype=float)

def mapping(nside, angs):

    npix = hp.nside2npix(nside)
    maph = np.zeros(npix)

    pix = hp.ang2pix(nside, angs[:, 1], angs[:, 0])
    vals, times = np.unique(pix, return_counts=True)

    maph[vals] = times
    maph *= float(npix)/len(angs)

    return maph

# Make a supmap out of maps in a dictionary
def supmaps(dmaps, supmapiso=None):
	npix = hp.get_map_size(dmaps['0'])
	supmap = np.zeros(npix)

	for key in dmaps.keys():
		supmap += dmaps[key]/npix

	sumi = np.sum(supmap)
	supmap *= npix/sumi

	if np.any(supmapiso):
		pixs = np.nonzero(supmapiso)
		supmap[pixs] /= supmapiso[pixs]
		sumi = np.sum(supmap)
		supmap *= npix/sumi

	return supmap

# Make modf maps out of the given maps and a supmap
def make_modf_maps(maps, supmap, eta_cut=0.9):
    npix = hp.get_map_size(maps['0'])
    nside = hp.npix2nside(npix)
    
    mask = np.ones(npix)
    for i in range(npix):
        if np.abs(-np.log(np.tan(hp.pix2ang(nside, i)[0]/2.))) < eta_cut:
            mask[i] = 0.

    finmap = supmap/npix*(1.-mask)+mask
    pixs = np.where(finmap == 0.)
    finmap[pixs] = 1.
                    
    modf_maps = {}
    for key in maps.keys():
        modf_maps[key] = np.copy(maps[key])/npix
        modf_maps[key] /= finmap
        sumi = np.sum(modf_maps[key])
        modf_maps[key] *= npix/sumi
        
    return modf_maps