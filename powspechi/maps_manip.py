import numpy as np
import healpy as hp
import os
from powspechi.pserrors import IsomapError

# Read event file
def readevtfile(infile, skip_header=True):

	r"""Read an event file with at least two columns, where the first should
	correspond to the particles' azimuthal (:math:`\phi`) coordinates and the
	second to the polar (:math:`\theta`) coordinates.

	Parameters
	----------
	infile : filenamestr
		The event file name whose format is described above.
	skip_header : bool, optional
		Option to skip the first line of the file. Default: *True*.

	Returns
	-------
	angs : float, ndarray
		A 2-D array whose shape is *(mult, ncol)*, where *mult* is the
		event multiplicity and *ncol* the number of columns.

	"""
    
	data = []

	with open(infile, 'r') as f:
		if skip_header:
			f.readline()
		for line in f:
			data.append(line.split())

	angs = np.asarray(data, dtype=float)

	return angs

# Mapping angs of type 'phi theta'
def mapping(nside, angs):

	npix = hp.nside2npix(nside)
	maph = np.zeros(npix)

	pix = hp.ang2pix(nside, angs[:, 1], angs[:, 0])
	vals, times = np.unique(pix, return_counts=True)

	maph[vals] = times
	maph *= float(npix)/len(angs)

	return maph

# Get supmap_iso numpy array:
def get_supmap_iso(nside, eta_cut=0.9):

	curr_dir = os.path.dirname(__file__)
	det_file = os.path.join(curr_dir, 'supmaps_iso/supmap_iso%s_ns%d.fits' %(eta_cut, nside))

	if os.path.isfile(det_file):
		supmapiso = hp.read_map(det_file, verbose=False)
		return supmapiso

	else:
		raise IsomapError('The desired supmap_iso file with nside = %d and |eta| < %s does not exist. Please refer to documentation.' %(nside, eta_cut))

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
	mask[hp.query_strip(nside, qi, qf)] = 0.

	finmap = supmap/npix*(1.-mask)+mask
	pixs = np.where(finmap == 0.)
	finmap[pixs] = 1.

	modf_maps = maps / (npix*finmap)
	modf_maps *= npix / np.sum(modf_maps, axis=1)[:, None]

	return modf_maps