
import os
import subprocess
import numpy as np

from scipy.ndimage import binary_dilation

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
    #print(f'Target beam: {head["BMAJ"]*60}')
    #print(f'Model beam: {head_cont["BMAJ"]*60}')
    kwidth = 0.42466090014400953 * (np.sqrt(head['BMAJ']**2. - head_cont['BMAJ']**2.)/abs(head_cont['CDELT2']))
    #print(f'Kernel width: {kwidth}')
    kernel = Gaussian2DKernel(kwidth)

    # Convolve the model continuum map.
    cont_cnv = convolve(cont[0], kernel, boundary='fill')
    cont_cnv = np.ma.masked_invalid(cont_cnv)

    # Reproject the GBT continuum.
    cont_obs_rpj = reproject_interp((cont_obs, wcs.celestial), wcs_cont.celestial, 
                                    return_footprint=False, shape_out=cont_cnv.shape)
    cont_obs_rpj = np.ma.masked_invalid(cont_obs_rpj)
    cont_obs_rpj = np.ma.masked_where(cont_obs_rpj == 0, cont_obs_rpj)

    # Combine and dilate masks.
    mask = cont_cnv.mask | cont_obs_rpj.mask
    mask = binary_dilation(mask, iterations=2)
    cont_cnv = np.ma.masked_where(mask, cont_cnv)
    cont_obs_rpj = np.ma.masked_where(mask, cont_obs_rpj)

    # Define power and temperature vectors.
    # Avoid edge pixels since the model map is not big enough.
    tsou = cont_cnv[20:-20,20:-20].compressed()
    psou = cont_obs_rpj[20:-20,20:-20].compressed()

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
