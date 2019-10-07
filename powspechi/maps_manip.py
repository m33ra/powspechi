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

	r"""Make a map with a chosen resolution out of particles angular 
	coordinates :math:`(\phi, \theta)`.

	Parameters
	----------
	nside : int, scalar
		Chosen map resolution.
	angs : float, ndarray
		A 2-D array whose first column corresponds to the :math:`\phi`
		coordinate while the second corresponds to the :math:`\theta`
		coordinate.

	Returns
	-------
	maph : float, ndarray
		A 1-D array whose size depends on the `nside` through the relation
		:math:`\mathrm{len(maph)} = 12 \cdot nside^2`.

	"""

	npix = hp.nside2npix(nside)
	maph = np.zeros(npix)

	pix = hp.ang2pix(nside, angs[:, 1], angs[:, 0])
	vals, times = np.unique(pix, return_counts=True)

	maph[vals] = times
	maph *= float(npix)/len(angs)

	return maph

# Get supmap_iso numpy array:
def getsupmapiso(nside, etacut=0.9):

	r"""Get the desired supmap_iso.

	Parameters
	----------
	nside : int, scalar
		Map resolution.
	etacut : float, scalar, optional
		The imposed limit to pseudorapidity, such that :math:`|\eta|` < `etacut`.
		Default: 0.9.

	Returns
	-------
	supmapiso : float, ndarray
		A 1-D array representing a HEALPix map with the specified resolution.

	Raises
	------
	IsomapError
		If the `supmap_iso*.fits` file does not exist.

	Notes
	-----
	The maps in the files `supmap_iso*.fits` are meant to correct for edge effects when
	there is a limit on :math:`\eta` (:math:`\theta`) and it is necessary to divide the
	event maps their ensemble sum. In the case of no :math:`\theta` limitations
	or no divisions, such maps are not necessary.

	"""

	curr_dir = os.path.dirname(__file__)
	det_file = os.path.join(curr_dir, 'supmaps_iso/supmap_iso%s_ns%d.fits' %(etacut, nside))

	if os.path.isfile(det_file):
		supmapiso = hp.read_map(det_file, verbose=False)
		return supmapiso

	else:
		raise IsomapError('The desired supmap_iso*.fits file with nside = %d and |eta| < %s does not exist. Please refer to documentation.' %(nside, etacut))

# Make a supmap out of maps in a dictionary
def supmaps(maps, supmapiso=None):

	r"""Sum an ensemble of maps.

	Parameters
	----------
	maps : float, array_like
		A map or a list/array of maps.
	supmapiso : float, ndarray, optional
		A map limited in :math:`\theta`, used to account for the pixel weights
		on map edges. Default: *None*.

	Returns
	-------
	supmap : float, ndarray
		A 1-D array resultant of the sum of the elements in `maps`. If `supmapiso`
		is given, weights are assigned to the pixels on the edges of `supmap`.

	"""
	
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
def make_modfmaps(maps, supmap, etacut=0.9):

	r"""Divide an ensemble of maps by a single map, preferably the sum of
	said ensemble.

	Parameters
	----------
	maps : float, array_like
		A single map or an ensemble of maps. They should be limited in
		pseudorapidity by the value in `etacut`.
	supmap : float, ndarray
		A 1-D array usually representing the sum of all elements in `maps`.
	etacut : float, scalar, optional
		The value of the pseudorapidity limit, :math:`|\eta|` < `etacut`.
		Default: 0.9.

	Returns
	-------
	modf_maps : float, array_like
		The result of dividing `maps` by `supmap`. Its shape will be the same
		as `maps`.

	"""

	if maps[0].ndim == 0:
		maps = np.reshape(maps, (1, len(maps)))

	npix = hp.get_map_size(maps[0])
	nside = hp.npix2nside(npix)

	qi, qf = 2.*np.arctan(np.exp(-np.array([etacut, -etacut])))
	mask = np.ones(npix)
	mask[hp.query_strip(nside, qi, qf)] = 0.

	finmap = supmap/npix*(1.-mask)+mask
	pixs = np.where(finmap == 0.)
	finmap[pixs] = 1.

	modf_maps = maps / (npix*finmap)
	modf_maps *= npix / np.sum(modf_maps, axis=1)[:, None]

	return modf_maps