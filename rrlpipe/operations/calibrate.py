
import numpy as np

from astropy.convolution import convolve

from groundhog import sd_fits_utils, spectral_axis

from rrlpipe import utils


def get_continuum(freq, data, poly_order, blanks=None):
    """
    """
    y = np.ma.copy(data)
    x = np.copy(freq.to('MHz').value)
    
    if blanks is not None:
        for b in blanks:
            y.mask[b[0]:b[1]] = True
    x = np.ma.masked_where(y.mask, x)
        
    pfit = np.polyfit((x - x.mean()).compressed(), y.compressed(), poly_order)
    pval = np.poly1d(pfit)(x.data - x.mean())
    
    return pval


def remove_continuum(freq, data, poly_order, blanks=None):
    """
    """
    
    pval = get_continuum(freq, data, poly_order, blanks=blanks)
    
    return (data - pval)


def cal_no_ref(table, poly_order, lines, tcal):
    """
    """

    # Find lines that will be blanked.
    freq = spectral_axis.compute_freq_axis(table)
    blanks = []
    for line,(dv,z) in lines.items():
        l = utils.find_rrls(freq[0], line, dv, z=z)
        for k,v in l.items():
            blanks.append(l[k])

    # Find rows with the noise diode on and off.
    mask_on = sd_fits_utils.get_table_mask(table, cal='T')
    mask_off = sd_fits_utils.get_table_mask(table, cal='F')
    assert(mask_on.sum() == mask_off.sum())

    # Compute the difference between the rows with the noise
    # diode on and off: Pon-Poff = G Tcal
    # and fit a polynomial to it.
    pon = np.ma.masked_invalid(np.nanmean(table['DATA'][mask_on], axis=0))
    pof = np.ma.masked_invalid(np.nanmean(table['DATA'][mask_off], axis=0))
    x = freq[0].to('MHz').value
    gtcal = pon - pof
    pfit = np.polyfit(x - x.mean(), gtcal, 1)
    gtcal_val = np.poly1d(pfit)(x - x.mean())

    # Remove blanked rows.
    mask = np.ma.masked_invalid(table['DATA']).mask.any(axis=1)
    print(f"Percentage of blanked data: {mask.sum()/mask.shape[0]*100:.2f} %")
    table = table[~mask]

    # Calibrate all rows.
    cal_data = np.ma.empty(table['DATA'].shape, dtype=table['DATA'].dtype)
    for i in range(len(table)):
        tl = remove_continuum(freq[i],
                              np.ma.masked_invalid(table['DATA'][i]),
                              poly_order,
                              blanks)
        if tcal is None:
            cal_data[i] = tl/gtcal_val*table['TCAL'][i]
        else:
            cal_data[i] = tl/gtcal_val*tcal

    # Update the table data column.
    table['DATA'] = cal_data

    return table


def cal_refsmo(table, ref_table, tcal, kernel):
    """
    """


    mask_on = sd_fits_utils.get_table_mask(ref_table, cal='T')
    mask_off = sd_fits_utils.get_table_mask(ref_table, cal='F')

    # Compute system temperature at reference position.
    tsys = tcal/(ref_table['DATA'][mask_on]/ref_table['DATA'][mask_off] - 1.)

    tint_on = ref_table['EXPOSURE'][mask_on][:,np.newaxis]
    tint_off = ref_table['EXPOSURE'][mask_off][:,np.newaxis]

    # Average the system temperature in time.
    tsys_avg = np.ma.average(tsys, axis=0, weights=(tint_on+tint_off)*np.power(tsys, -2.))
    # Average the system temperature over the inner 80% of the band.
    tsys_avg = tsys_avg[int(0.1*len(tsys_avg)):int(0.9*len(tsys_avg))].mean()

    # Average the counts at the reference position with the noise diode on and off.
    p_ref_on = np.ma.average(ref_table['DATA'][mask_on], axis=0, weights=tint_on*np.power(tsys, -2.))
    p_ref_off = np.ma.average(ref_table['DATA'][mask_on], axis=0, weights=tint_off*np.power(tsys, -2.))
    
    # Smooth the reference spectra.
    if kernel is not None:
        p_ref_on = convolve(p_ref_on, kernel)
        p_ref_off = convolve(p_ref_off, kernel)
        
    # Calibrate the data.
    mask_on = sd_fits_utils.get_table_mask(table, cal='T')
    cal_data_on = (table['DATA'][mask_on] - p_ref_on[np.newaxis,:])/p_ref_on[np.newaxis,:] * (tsys_avg + tcal) - tcal
    mask_off = sd_fits_utils.get_table_mask(table, cal='F')
    cal_data_off = (table['DATA'][mask_off] - p_ref_off[np.newaxis,:])/p_ref_off[np.newaxis,:] * tsys_avg

    # Update the data and system temperature.
    table['DATA'][mask_on] = cal_data_on
    table['DATA'][mask_off] = cal_data_off   
    table['TSYS'] = tsys_avg

    return table 


def run(args, mode='no-ref'):
    """
    """

    modes = {'no-ref': cal_no_ref,
             'refsmo': cal_refsmo}

    table = modes[mode](*args)

    return table
    
    
