
import os
import subprocess
import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from astropy.convolution import convolve, Gaussian2DKernel
from reproject import reproject_interp

from rrlpipe import utils
from groundhog import sd_fits_io

def run(table, model_file, cleanup=True):
    """
    """

    
    file_tmp = 'temp.fits'
    try:
        os.remove(file_tmp)
    except FileNotFoundError:
        pass
    # Write table to grid.
    sd_fits_io.write_sdfits(file_tmp, table)

    # Grid the GBT data.
    #gbt_cube = utils.grid_map_data(sdfitsfile, nx, ny, scale, xcntr, ycntr)    
    cube_out = f"{os.path.splitext(file_tmp)[0]}"
    ch0 = int(table['DATA'].shape[1]*0.2)
    chf = int(table['DATA'].shape[1]*0.8)
    args = ['gbtgridder', '--noline', '--nocont', '--noweight',
            '-o', cube_out, '-c', f'{ch0}:{chf}', 
            file_tmp]
    subprocess.run(args)
    cube_file = cube_out + '_cube.fits'

    # Clean up.
    if cleanup:
        try:
            os.remove(file_tmp)
        except FileNotFoundError:
            pass
        try:
            os.remove(cube_out + '_weight.fits')
        except FileNotFoundError:
            pass

    # Load the gridded GBT data.
    hdu = fits.open(cube_file)
    head = hdu[0].header
    data = np.ma.masked_invalid(hdu[0].data)
    wcs = WCS(head)
    # Get the median of the GBT cube.
    cont_obs = np.ma.median(data[0], axis=0)

    # Load the continuum map.
    hdu_cont = fits.open(model_file)
    cont = np.ma.masked_invalid(hdu_cont[0].data)
    head_cont = hdu_cont[0].header
    wcs_cont = WCS(head_cont)

    # Define the convolution kernel to match the GBT observations.
    kwidth = 0.42466090014400953 * (np.sqrt(head['BMAJ']**2 - head_cont['BMAJ'])/head_cont['CDELT2'])
    kernel = Gaussian2DKernel(kwidth)

    # Convolve the model continuum map.
    cont_cnv = convolve(cont[0], kernel, boundary='fill')

    # Reproject the GBT continuum.
    cont_obs_rpj = reproject_interp((cont_obs, wcs.celestial), wcs_cont.celestial, 
                                    return_footprint=False, shape_out=cont_cnv.shape)
    
    # Define power and temperature vectors.
    # Avoid edge pixels since the model map is not big enough.
    tsou = cont_cnv[20:-20,20:-20].flatten()
    psou = cont_obs_rpj[20:-20,20:-20].flatten()

    # Fit a quadratic polynomial to the relation.
    # The first term is H, the second G and 
    # the third the system temperature.
    pfit = np.polyfit(psou, tsou, 2)
    print(f'H={pfit[0]}, G={pfit[1]}, Tsys={pfit[2]}')

    # Clean up.
    if cleanup:
        try:
            os.remove(cube_file)
        except FileNotFoundError:
            pass

    return pfit
