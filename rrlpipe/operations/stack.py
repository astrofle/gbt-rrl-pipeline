"""
Stacking functions for GBT-RRL-pipe.
"""

import os
#import warnings
import numpy as np

#from astropy.io import fits
from astropy import units as u
from spectral_cube import SpectralCube
from spectral_cube.masks import BooleanArrayMask
#from spectral_cube.utils import SpectralCubeWarning
from astropy.convolution import Gaussian1DKernel, interpolate_replace_nans


fwhm_factor = np.sqrt(8*np.log(2))


def interpolate_nans_cube(cube, kernel):
    """
    """

    # Manipulate spectral axis.
    inaxis = cube.spectral_axis
    indiff = np.mean(np.diff(inaxis))
    # Reverse spectral axis if needed.
    specslice = slice(None, None, 1) if indiff >= 0 else slice(None, None, -1)
    inaxis = inaxis[specslice]
    indiff = np.mean(np.diff(inaxis))
    # Get cube data.    
    cubedata = cube.filled_data
    # Store interpolated results here.
    newcube = np.empty(cubedata.shape,
                       dtype=cubedata[:1, 0, 0].dtype)
    newmask = np.empty(cubedata.shape,
                       dtype=bool)

    # Loop over pixels interpolating the spectral axis.
    yy,xx = np.indices(cube.shape[1:])    
    for ix, iy in (zip(xx.flat, yy.flat)):
        mask = cube.mask.include(view=(specslice, iy, ix))
        if any(mask):
            newcube[specslice,iy,ix] = \
                interpolate_replace_nans(cubedata[specslice,iy,ix].value, kernel)
            newmask[:,iy,ix] = True
        else:
            newmask[:, iy, ix] = False
            newcube[:, iy, ix] = np.NaN

    # Update WCS.
    newwcs = cube.wcs.deepcopy()
    newwcs.wcs.crpix[2] = 1
    newwcs.wcs.crval[2] = inaxis[0].value if specslice.step > 0 \
                                          else inaxis[-1].value
    newwcs.wcs.cunit[2] = inaxis.unit.to_string('FITS')
    newwcs.wcs.cdelt[2] = indiff.value if specslice.step > 0 \
                                       else -indiff.value
    newwcs.wcs.set()

    newbmask = BooleanArrayMask(newmask, wcs=newwcs)

    newcube = cube._new_cube_with(data=newcube, wcs=newwcs, mask=newbmask,
                                  meta=cube.meta,
                                  fill_value=cube.fill_value)

    return newcube


def stack_cubes(cubes, vmin, vmax, dv, rms_vmin, rms_vmax, output, overwrite=False, 
                replace_nans=True, spectral_smooth=True):
    """
    This assumes that all the input cubes have the same 
    spatial WCS transformation and number of pixels.
    """

    vel_axis = np.arange(vmin, vmax+dv, dv) * u.m/u.s
    vgrid_cubes = []

    # Smooth and grid the spectral axis of the cubes 
    # to the same velocity axis.
    # During the smoothing interpolate NaN channels.
    for fnm in cubes:
        
        cube = SpectralCube.read(fnm)
        dv_cube = cube.header['CDELT3']
        dv_chan = dv/dv_cube
        kwidth = dv_chan/fwhm_factor
        kernel = Gaussian1DKernel(kwidth)
        smcube = cube.spectral_smooth(kernel, 
                                      nan_treatment='interpolate',
                                      preserve_nan=False)
        new_cube = smcube.spectral_interpolate(vel_axis, 
                                               suppress_smooth_warning=True)
        # Are there any masked channels left?
        if any(new_cube.mask.view().all(axis=(1,2))) \
                and replace_nans:
            new_cube = interpolate_nans_cube(new_cube, Gaussian1DKernel(8))
        fnm_out = f'{os.path.splitext(fnm)[0]}_vi.fits'
        new_cube.write(fnm_out, format='fits')
        vgrid_cubes.append(fnm_out)

    # Create an array to store the stack.
    stack = np.ma.empty((len(vgrid_cubes),)+new_cube.shape, dtype=float)
    rms = np.ma.empty((len(vgrid_cubes),)+new_cube.shape, dtype=float)

    # Loop over the new cubes and add them to the stack.
    for i,fnm in enumerate(vgrid_cubes):
        cube = SpectralCube.read(fnm)
        stack[i] = np.ma.masked_invalid(cube.unmasked_data[:,:,:].value)
        rms[i] = cube.spectral_slab(rms_vmin, 
                                    rms_vmax).std(axis=0).value


    stack = np.ma.average(stack, axis=0, weights=1./np.ma.power(rms, 2.))

    stack_cube = SpectralCube(data=stack.filled(np.nan), wcs=cube.wcs, header=cube.header)
    stack_cube.write(output, format='fits', overwrite=overwrite)