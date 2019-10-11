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
	or no divisions, such maps are not necessary. Currently, ``powspechi`` supports such
	corrections for `nside`: 8, 16, 32, 64 and 128 and `etacut`: 0.8 and 0.9.

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

# Make norm maps out of the given maps and a supmap
def make_normmaps(maps, supmap, etacut=0.9):

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
		If there is no limit, set it to *None*. Default: 0.9.

	Returns
	-------
	norm_maps : float, array_like
		The result of dividing `maps` by `supmap`. Its shape will be the same
		as `maps`. 

	Notes
	-----
	In the power spectral analysis at hand [1]_ [2]_, `supmap` is the sum
	of all event maps and it is represented by :math:`F^{all}(\mathbf{n_p})`,
	where :math:`\mathbf{n_p}` is a pixel number. A normalized map is thus defined 
	by the following expression:

	.. math:: \bar{f}(\mathbf{n_p}) = \frac{f(\mathbf{n_p})}{F^{all}(\mathbf{n_p})},

	where :math:`f(\mathbf{n_p})` is a map from the original event ensemble, the latter 
	denoted by the `maps` parameter.

	References
	----------
	.. [1] M. Machado, P.H. Damgaard, J.J. Gaardhoeje, and C. Bourjau, "Angular power spectrum of heavy ion collisions", Phys. Rev. C **99**, 054910 (2019).
	.. [2] M. Machado, "Heavy ion anisotropies: a closer look at the angular power spectrum", arXiv:1907.00413 [hep-ph] (2019). 

	"""

	if maps[0].ndim == 0:
		maps = np.reshape(maps, (1, len(maps)))

	npix = hp.get_map_size(maps[0])
	nside = hp.npix2nside(npix)

	if etacut:
		qi, qf = 2.*np.arctan(np.exp(-np.array([etacut, -etacut])))
		mask = np.ones(npix)
		mask[hp.query_strip(nside, qi, qf)] = 0.
	else:
		qi, qf = 0., 2*np.pi
		mask = 0.

	finmap = supmap/npix*(1.-mask)+mask
	pixs = np.where(finmap == 0.)
	finmap[pixs] = 1.

	norm_maps = maps / (npix*finmap)
	norm_maps *= npix / np.sum(norm_maps, axis=1)[:, None]

	return norm_maps