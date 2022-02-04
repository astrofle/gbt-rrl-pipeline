
import os
import shutil
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import numpy.lib.recfunctions as rfn

import fitsio
import pandas as pd

from astropy.io import fits
from lmfit.models import PolynomialModel

from crrlpy import crrls
from scripts import cube2vel
from groundhog import sd_fits, sd_fits_io, spectral_axis


def find_rrls(freq, line, vel_range, z=0):
    """
    """

    n, f = crrls.find_lines_sb(np.sort(freq.to('MHz').value),
                               line, z=z)

    n = list(map(int, n))
    lines = dict.fromkeys(n)

    for i,(n_,f_) in enumerate(zip(n,f)):
        df = crrls.dv2df(f_, vel_range)
        ch0 = np.argmin(abs(freq.to('MHz').value - (f_-df)))
        chf = np.argmin(abs(freq.to('MHz').value - (f_+df)))
        ch0,chf = np.sort([ch0,chf])
        lines[n_] = [ch0, chf]

    return lines


def flag_rfi(sdfitsfile, strategy):
    """
    """

    file_norm = f"{os.path.splitext(sdfitsfile)[0]}_norm.fits"
    file_out = f"{os.path.splitext(sdfitsfile)[0]}_rfi.fits"

    # Load the SDFITS file.
    sdfits = sd_fits.SDFITS()
    sdfits.load(sdfitsfile)
    table = sdfits.hdu[1][:]
    head = sdfits.hdu[1].read_header()    
    freq = spectral_axis.compute_freq_axis(table, chstart=1, chstop=-1, apply_doppler=True)

    # Shape of the data array to be compatible with AOFlagger.
    aof_shape = (head['NAXIS2'],1,1,1,freq.shape[1])

    # Define the columns of a new table.
    ao_dtype = np.dtype([('OBJECT', '<U32'), 
                         ('DATE-OBS', '<U22'), ('TIME', '>f8'), ('EXPOSURE', '>f8'),
                         ('RESTFREQ', '>f8'), ('FREQRES', '>f8'), ('BANDWID', '>f8'),
                         ('CRPIX1', '>f4'), ('CRVAL1', '>f8'), ('CDELT1', '>f8'), 
                         ('CRVAL3', '>f8'), ('CRVAL4', '>f8'), ('IFNUM', '>i2'),
                         ('DATA', '>f4', aof_shape[1:]), ('FLAGGED', 'u1', aof_shape[1:]),
                         ('SCAN', '>i4'), ('PLNUM', '>i2'), ('FDNUM', '>i2')])
    new_table = np.empty(head['NAXIS2'], dtype=ao_dtype)

    # Loop over rows normalizing the data column.
    new_data = np.empty(aof_shape, dtype=float)
    for i in range(len(table)):    
        row = table['DATA'][i]
        pfit = np.polyfit(freq[0].value, row, 1)
        pval = np.poly1d(pfit)(freq[0].value)
        norm_row = (row - pval)
        norm_row /= norm_row.std()
        new_data[i,0,0,0,:] = norm_row
    
    # Update the data in the new table.
    new_table['DATA'] = new_data
    new_table['DATE-OBS'] = table['DATE-OBS'][0].split('T')[0]
    new_table['EXPOSURE'] = table['EXPOSURE']
    new_table['RESTFREQ'] = table['RESTFREQ']
    new_table['FREQRES'] = table['FREQRES']
    new_table['BANDWID'] = (freq[0,0] - freq[0,-1]).to('Hz').value #table['BANDWID']
    new_table['CRPIX1'] = table['CRPIX1']
    new_table['CRVAL1'] = table['CRVAL1']
    new_table['CDELT1'] = table['CDELT1']
    new_table['TIME'] = table['LST']
    new_table['SCAN'] = table['SCAN']

    # Write the new table to an SDFITS file.
    try:
        os.remove(file_norm)
    except FileNotFoundError:
        pass
    sd_fits_io.write_sdfits(file_norm, new_table, add_header=False)

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
        fits.setval(file_norm, k, value=v, ext=1)

    head = sdfits.hdu[1].read_header()
    for key in ['TELESCOP', 'PROJID', 'SITELONG', 'SITELAT', 'SITEELEV']:
        fits.setval(file_norm, key, value=head[key], ext=1)
        
    # Add units.
    kunits = {'EXPOSURE': 's', 'TIME': 's', 'RESTFREQ': 'Hz', 'FREQRES': 'Hz', 'BANDWID': 'Hz', 'CRVAL1': 'Hz',
              'CRVAL3': 'deg', 'CRVAL4': 'deg'}
    for k,v in kunits.items():
        for i,key in enumerate(ao_dtype.names):
            if k == key:
                fits.setval(file_norm, f'TUNIT{i+1}', value=v, ext=1)

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
    fper = flags.sum(axis=0)/flags.shape[0]
    mask = (fper >= 0.01)
    print(f"Number of channels totally flagged {mask.sum()}/{len(mask)}")
    flags[:,mask] = True
    # Apply flags to the data.
    masked_data = np.ma.masked_where(flags, sdfits.hdu[1]['DATA'][:])
    table['DATA'] = masked_data.filled(np.nan)
    # Write the flagged data.
    try:
        os.remove(file_out)
    except FileNotFoundError:
        pass
    sd_fits_io.write_sdfits(file_out, table)

    return file_out


def flatten_bandpass(sdfitsfile, v_line, dv_line, poly_order):
    """
    """

    fileout = f"{os.path.splitext(sdfitsfile)[0]}_bpcorr.fits"

    sdfits = sd_fits.SDFITS()
    sdfits.load(sdfitsfile)
    table = sdfits.hdu[1][:]
    freq = spectral_axis.compute_freq_axis(table, 
                                           chstart=1, chstop=-1, 
                                           apply_doppler=True)

    # Define channel range with line emission/absorption
    # to blank during baseline fitting.
    nu_center = np.mean(freq[0].to('MHz').value)
    dnu = crrls.dv2df(nu_center, dv_line)
    nu0 = crrls.vel2freq(nu_center, v_line)
    ch0 = np.argmin(abs(freq[0].to('MHz').value - (nu0-dnu)))
    chf = np.argmin(abs(freq[0].to('MHz').value - (nu0+dnu)))
    ch0,chf = np.sort([ch0,chf])

    # Select and blank data.
    data = np.ma.masked_invalid(table['DATA'])
    data[:,ch0:chf+1].mask = True

    # Fit a polynomial to the data.
    y = data.mean(axis=0)
    x = np.ma.masked_where(y.mask, freq[0].to('MHz').value)
    # Use the mean subtracted frequency to avoid rank errors.
    pfit = np.polyfit(x.compressed() - np.mean(x.compressed()), y.compressed(), poly_order)
    pval = np.poly1d(pfit)(freq[0].to('MHz').value - np.mean(freq[0].to('MHz').value))
    
    # Subtract the fitted polynomial from the data.
    new_data = np.ma.empty(table['DATA'][:].shape, dtype=np.float32)
    bp_arr = np.tile(pval, (table['DATA'][:].shape[0],1))
    bp_arr = bp_arr/np.mean(pval)*np.nanmean(table['DATA'], axis=1)[:,np.newaxis]
    new_data[:,:] = np.ma.masked_invalid(table['DATA'][:] - bp_arr)
    
    # Write the bandpass flattened data.
    table['DATA'] = new_data.filled(np.nan)
    try:
        os.remove(fileout)
    except:
        pass
    sd_fits_io.write_sdfits(fileout, table)

    return fileout
  

def freq2vel(cubefile, line, z=0):
    """
    """

    fileout = f"{os.path.splitext(cubefile)[0]}_vel.fits"
    shutil.copyfile(cubefile, fileout)
    cube2vel.cube2vel(fileout, transition=line, z=z, f_col=3, v_col=3)

    return fileout


def grid_map_data(sdfitsfile, nx, ny, scale):
    """
    """

    fileout = f"{os.path.splitext(sdfitsfile)[0]}"

    args = ['gbtgridder', '--noline', '--nocont',
            '-o', fileout, '--clobber', 
            '--size', f'{nx}', f'{ny}', '--pixelwidth', f'{scale}',
            '--kernel', 'gaussbessel',
            sdfitsfile]
    subprocess.run(args)

    cubefile = f"{os.path.splitext(sdfitsfile)[0]}_cube.fits"

    return cubefile


def make_line_list(sdfitsfiles, line_list_file, rms_vmin, rms_vmax, line_vmin, line_vmax):
    """
    """

    # Fields for the line list.
    dtype = [('spw', int), ('pol', int), ('n', int), 
             ('rms', float), ('use', bool), ('file', '<U256')]
    line_list = np.zeros(len(sdfitsfiles), dtype=dtype)

    for i,f in enumerate(sdfitsfiles):
        spw = int(f.split('_')[1])
        pol = int(f.split('_')[3])
        n = int(f.split('_')[5])
        sdfits = sd_fits.SDFITS()
        sdfits.load(f)
        table = sdfits.hdu[1][:]
        freq = spectral_axis.compute_freq_axis(table, chstart=1, chstop=-1, apply_doppler=True)
        nu_min = crrls.vel2freq(freq[0].to('MHz').value.mean(), rms_vmin)
        nu_max = crrls.vel2freq(freq[0].to('MHz').value.mean(), rms_vmax)
        ch_min = np.argmin(abs(freq[0].to('MHz').value - nu_min))
        ch_max = np.argmin(abs(freq[0].to('MHz').value - nu_max))
        ch_min,ch_max = np.sort([ch_min,ch_max])
        avg = np.nanmean(table['DATA'], axis=0)
        rms = avg[ch_min:ch_max+1].std()
        line_list[i] = (spw, pol, n, rms, 0, f)

    # Compute statistics across all files.
    rms_avg = np.nanmean(line_list['rms'])
    rms_med = np.nanmedian(line_list['rms'])
    rms_std = np.nanstd(line_list['rms'])

    # Lines with a high rms will not be used.
    line_list['use'] = (line_list['rms'] < rms_med + rms_std)

    # Now loop over the lines again and find 
    # the order of the best fit polynomial.
    # If the order is greater than zero,
    # then do not use the line.
    for i,row in enumerate(line_list):
        if row[4]:
            
            sdfits = sd_fits.SDFITS()
            sdfits.load(row[5])
            table = sdfits.hdu[1][:]
            freq = spectral_axis.compute_freq_axis(table, chstart=1, chstop=-1, apply_doppler=True)
            nu_min = crrls.vel2freq(freq[0].to('MHz').value.mean(), line_vmin)
            nu_max = crrls.vel2freq(freq[0].to('MHz').value.mean(), line_vmax)
            ch_min = np.argmin(abs(freq[0].to('MHz').value - nu_min))
            ch_max = np.argmin(abs(freq[0].to('MHz').value - nu_max))
            ch_min,ch_max = np.sort([ch_min,ch_max])
            avg = np.nanmean(table['DATA'], axis=0)
            x = freq[0].to('MHz').value
            x -= x.mean()
            y = avg
            # Remove NaNs.
            x = x[~np.isnan(y)]
            y = y[~np.isnan(y)]
            # Blank line.
            ym = np.hstack([y[:ch_min],y[ch_max:]])
            xm = np.hstack([x[:ch_min],x[ch_max:]])
            bic = np.zeros(8, dtype=float)
            aic = np.zeros(8, dtype=float)
            # Fit polynomials with degrees [0,7].
            for deg in range(0,8):
                mod = PolynomialModel(degree=deg)
                pars = mod.guess(y, x=x)
                fit = mod.fit(ym, pars, x=xm)
                bic[deg] = fit.bic
                aic[deg] = fit.aic
            if np.argmin(aic) > 0 and np.argmin(bic) > 0:
                line_list[i][4] = False
    
    # Save as a text file for later use.
    df = pd.DataFrame(line_list)
    df.to_csv(line_list_file, sep='\t', index=False, na_rep='NaN', escapechar='#')

    return line_list


def make_report(sdfitsfile):
    """
    """

    fileout = f"{os.path.splitext(sdfitsfile)[0]}_report.pdf"

    sdfits = sd_fits.SDFITS()
    sdfits.load(sdfitsfile)
    sdfits.make_summary()

    table = sdfits.hdu[1][:]
    freq = spectral_axis.compute_freq_axis(table, chstart=1, chstop=-1, apply_doppler=True)

    fig = plt.figure(dpi=150, frameon=False)
    ax1 = fig.add_subplot(211)
    ax1.plot(freq[0].to('MHz'), np.nanmean(table['DATA'], axis=0))
    ax1.set_ylabel("Antenna temperature (K)")
    ax1.minorticks_on()
    ax1.tick_params('both', direction='in', which='both',
                    bottom=True, top=True, left=True, right=True)
    ax2 = fig.add_subplot(212)
    ax2.plot(freq[0].to('MHz'), np.ma.masked_invalid(table['DATA']).mask.sum(axis=0)/len(freq))
    ax2.set_ylabel("Fraction of flagged data")
    ax2.set_xlabel("Frequency (MHz)")
    ax2.minorticks_on()
    ax2.tick_params('both', direction='in', which='both',
                    bottom=True, top=True, left=True, right=True)    
    plt.savefig(fileout, bbox_inches='tight', pad_inches=0.06)
    plt.close(fig)


def split_rrls(sdfitsfile, line, vel_range, z=0):
    """
    """

    fileout = os.path.splitext(sdfitsfile)[0]

    sdfits = sd_fits.SDFITS()
    sdfits.load(sdfitsfile)
    sdfits.make_summary()

    freq = spectral_axis.compute_freq_axis(sdfits.hdu[1][:], 
                                           chstart=1, chstop=-1, 
                                           apply_doppler=True)

    n, f = crrls.find_lines_sb(np.sort(freq[0].to('MHz').value), 
                               line, z=z)

    lines = dict.fromkeys(n)

    # Loop over lines, 
    # select a velocity range centered on the lines, 
    # and write new SDFITS for each one.
    for i,(n_,f_) in enumerate(zip(n,f)):
        df = crrls.dv2df(f_, vel_range)
        ch0 = np.argmin(abs(freq[0].to('MHz').value - (f_-df)))
        chf = np.argmin(abs(freq[0].to('MHz').value - (f_+df)))
        ch0,chf = np.sort([ch0,chf])
        file_n = f"{fileout}_n_{int(n_)}.fits"
        lines[n_] = [ch0, chf, file_n]
        sub_table = sdfits.get_channels(ch0, chf, dch=1, extname='SINGLE_DISH')
        try:
            os.remove(file_n)
        except FileNotFoundError:
            pass
        sd_fits_io.write_sdfits(file_n, sub_table)

    return lines


def split_channel_range(table, ch0, chf, dch=1):
    """
    """

    # Find the number of channels and 
    # define how many channels the selection will have.
    nrow, nchan = table['DATA'].shape
    fslice = slice(ch0, chf, dch)
    chan_slice = fslice.indices(nchan)
    nchan_sel = (chan_slice[1] - chan_slice[0])//chan_slice[2]

    # Remove the DATA column from the table.
    nodata_table = rfn.drop_fields(table, 'DATA')
    # Copy the column definitions as a list.
    nodata_table_dt = nodata_table.dtype.descr
    # Concatenate the column definitions with the new data shape.
    new_dt = np.dtype(nodata_table_dt[:6] + [('DATA', '>f4', (nchan_sel,))] + nodata_table_dt[6:])
    # Create a new table with the same number of rows.
    new_table = np.empty(nrow, dtype=new_dt)

    # Fill the new table with the old contents, 
    # and the DATA selection.
    for n in nodata_table.dtype.names:
        new_table[n] = nodata_table[n]
    new_table['DATA'] = table['DATA'][:,fslice]
    # Update the frequency axis and bandwidth.
    new_table['CRPIX1'] -= ch0
    new_table['BANDWID'] = new_table['FREQRES'] * nchan_sel


    return new_table    
