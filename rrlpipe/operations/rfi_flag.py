
import os
import subprocess
import numpy as np

import fitsio

from astropy.io import fits
from astropy.convolution import Box1DKernel

from rrlpipe import utils
from groundhog import sd_fits_io, spectral_axis


def extend_flags(flags, threshold=1.1):
    """
    """

    eflags = np.ma.copy(flags)

    # Determine the percentage of flagged data for each channel
    # and its median value.
    fper = flags.sum(axis=0)/flags.shape[0]
    fper_med = np.median(fper)
    # Flag any channels which have more than 10% of their data
    # flagged relative to the median of all the channels.
    mask = (fper_med*threshold < fper)
    # Extend the flags to the adjacent channels.
    mask = repeat_or(mask, n=3)
    # Center the extended flags on the originally flagged channel.
    mask = np.roll(mask, -1)
    # Update the flags.
    eflags[:,mask] = True

    return eflags


def prepare_table(table, output, header):
    """
    """

    aof_shape = (len(table),1,1,1,table['DATA'].shape[1])

    # Define the columns of a new table.
    ao_dtype = np.dtype([('OBJECT', '<U32'),
                         ('DATE-OBS', '<U22'), ('TIME', '>f8'), ('EXPOSURE', '>f8'),
                         ('RESTFREQ', '>f8'), ('FREQRES', '>f8'), ('BANDWID', '>f8'),
                         ('CRPIX1', '>f4'), ('CRVAL1', '>f8'), ('CDELT1', '>f8'),
                         ('CRVAL3', '>f8'), ('CRVAL4', '>f8'), ('IFNUM', '>i2'),
                         ('DATA', '>f4', aof_shape[1:]), ('FLAGGED', 'u1', aof_shape[1:]),
                         ('SCAN', '>i4'), ('PLNUM', '>i2'), ('FDNUM', '>i2')])
    new_table = np.empty(len(table), dtype=ao_dtype)

    freq = spectral_axis.compute_spectral_axis(table)

    # Loop over rows normalizing the data column.
    new_data = np.empty(aof_shape, dtype=float)
    for i in range(len(table)):
        row = table['DATA'][i]
        pfit = np.polyfit(freq[0].value - freq[0].value.mean(), row, 1)
        pval = np.poly1d(pfit)(freq[0].value - freq[0].value.mean())
        norm_row = (row - pval)
        norm_row /= norm_row.std()
        new_data[i,0,0,0,:] = norm_row

    # Update the data in the new table.
    new_table['DATA'] = new_data
    new_table['DATE-OBS'] = table['DATE-OBS'][0].split('T')[0]
    new_table['EXPOSURE'] = table['EXPOSURE']
    new_table['RESTFREQ'] = table['RESTFREQ']
    new_table['FREQRES'] = table['FREQRES']
    new_table['BANDWID'] = (freq[0,0] - freq[0,-1]).to('Hz').value
    new_table['CRPIX1'] = table['CRPIX1']
    new_table['CRVAL1'] = table['CRVAL1']
    new_table['CDELT1'] = table['CDELT1']
    new_table['TIME'] = table['LST']
    new_table['SCAN'] = table['SCAN']

    # Write the new table to an SDFITS file.
    try:
        os.remove(output)
    except FileNotFoundError:
        pass
    sd_fits_io.write_sdfits(output, new_table, add_header=False)

    # Update additional table keywords.
    hkeys = {'CTYPE1': 'FREQ',
             'CTYPE2': 'STOKES',
             'CRPIX2': 1.0,
             'CDELT2': -1.0,
             'CRVAL2': -5.0,
             'CTYPE3': 'RA',
             'CRPIX3': 1.0,
             'CDELT3': -1.0,
             'CTYPE4': 'DEC',
             'CRPIX4': 1.0,
             'CDELT4': 1.0,
             }
    for k,v in hkeys.items():
        fits.setval(output, k, value=v, ext=1)

    for key in ['TELESCOP', 'PROJID', 'SITELONG', 'SITELAT', 'SITEELEV']:
        fits.setval(output, key, value=header[key], ext=1)

    # Add units.
    kunits = {'EXPOSURE': 's', 'TIME': 's', 'RESTFREQ': 'Hz', 'FREQRES': 'Hz', 'BANDWID': 'Hz', 'CRVAL1': 'Hz',
              'CRVAL3': 'deg', 'CRVAL4': 'deg'}
    for k,v in kunits.items():
        for i,key in enumerate(ao_dtype.names):
            if k == key:
                fits.setval(output, f'TUNIT{i+1}', value=v, ext=1)


def repeat_or(mask, n=3):
    """
    Copied from:
    https://stackoverflow.com/questions/32706135/extend-numpy-mask-by-n-cells-to-the-right-for-each-bad-value-efficiently
    """
    m = np.copy(mask)
    k = m.copy()

    # lenM and lenK say for each mask how many
    # subsequent Trues there are at least.
    lenM, lenK = 1, 1

    # We run until a combination of both masks will give us n or more
    # subsequent Trues.
    while lenM+lenK < n:
        # Append what we have in k to the end of what we have in m.
        m[lenM:] |= k[:-lenM]

        # Swap so that m is again the small one.
        m, k = k, m

        # Update the lengths.
        lenM, lenK = lenK, lenM+lenK

    # See how much m has to be shifted in order to append the missing Trues.
    k[n-lenM:] |= m[:-n+lenM]

    return k


def run(table, strategy, path, header, interpolate_nans=False):
    """
    """

    file_norm = f"{path}_norm.fits"
    file_out = f"{path}_rfi.fits"

    prepare_table(table, file_norm, header)

    # Call AOFlagger on the new SDFITS file.
    args = ['/home/apps/aoflagger/bin/aoflagger',
            '-strategy', strategy,
            file_norm]
    subprocess.run(args)

    # Now, transfer the flags to the un-normalized data.
    # Load the flags.
    hdu = fitsio.FITS(file_norm, 'r')
    flags = hdu[1]['DATA'][:][:,0,0,0,:] == 1e20
    # Extend the flags.
    flags = extend_flags(flags, threshold=1.1)
    nchanflag = (flags.sum(axis=0)/len(flags) == 1).sum()
    print(f"Number of channels totally flagged {nchanflag}/{flags.shape[1]}")
    # Apply flags to the data.
    masked_data = np.ma.masked_where(flags, table['DATA'])
    table['DATA'] = masked_data.filled(np.nan)
    
    if interpolate_nans:
        # Interpolate NaN channels.
        kernel = Box1DKernel(nchanflag)
        table = utils.interpolate_flagged_table(table, kernel)

    # Write the flagged data.
    try:
        os.remove(file_out)
    except FileNotFoundError:
        pass
    sd_fits_io.write_sdfits(file_out, table)


    return table

