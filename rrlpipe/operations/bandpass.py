

import numpy as np

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

    return (data - pval) + pval.mean()


def run(table, poly_order, lines, tcal=None):
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

    # Flatten the bandpass response for all rows.
    cal_data = np.ma.empty(table['DATA'].shape, dtype=table['DATA'].dtype)
    for i in range(len(table)):
        tl = remove_continuum(freq[i],
                              np.ma.masked_invalid(table['DATA'][i]),
                              poly_order,
                              blanks)

        cal_data[i] = tl

    # Update the table data column.
    table['DATA'] = cal_data


    return table

