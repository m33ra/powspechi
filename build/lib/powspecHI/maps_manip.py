import numpy as np
import healpy as hp

def readevtfile(infile):
    
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
def supmaps(maps, supmapiso=None):
	if maps[0].ndim == 0:
		maps = np.reshape(maps, (1, len(maps)))

	npix = hp.get_map_size(maps[0])

	supmap = np.sum(maps, axis=0)
	supmap *= npix/np.sum(supmap)

	if np.any(supmapiso):
		pixs = np.nonzero(supmapiso)
		supmap[pixs] /= supmapiso[pixs]
		supmap *= npix/np.sum(supmap)

	return supmap

# Make modf maps out of the given maps and a supmap
def make_modf_maps(maps, supmap, eta_cut=0.9):
	if maps[0].ndim == 0:
		maps = np.reshape(maps, (1, len(maps)))

    npix = hp.get_map_size(maps[0])
    nside = hp.npix2nside(npix)
    
    qi, qf = 2.*np.arctan(np.exp(-np.array([eta_cut, -eta_cut])))
    mask = np.ones(npix)
    mask[hp.query_disc(nside, qi, qf)] = 0.

    finmap = supmap/npix*(1.-mask)+mask
    pixs = np.where(finmap == 0.)
    finmap[pixs] = 1.
                    
    modf_maps = maps / (npix*finmap)
    modf_maps *= npix / np.sum(modf_maps, axis=1)[:, None]
        
    return modf_maps