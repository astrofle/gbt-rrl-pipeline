"""
This script should be run after selecting the lines that will go into a stack.
It will, 
   * interpolate the data to a common velocity axis
   * stack the selected lines
   * change to Galactic coordinates if needed.
"""

import argparse
import numpy as np
import numpy.lib.recfunctions as rfn

import pandas as pd

from rrlpipe import utils
from groundhog import sd_fits, spectral_axis, sd_fits_io
from rrlpipe.operations import interpolate2vel, eq2gal


def parse_args():
    """
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--logfile",
                        type=str,
                        help="Input file.",
                        required=True)
    parser.add_argument("-o", "--output",
                        type=str,
                        help="Output file with updated flags.",
                        required=True)
    args = parser.parse_args()

    return args


def update_table(table, new_data, new_weights):
    """
    """

    nrow = len(table["DATA"])
    nchan = new_data.shape[1]

    # Remove the DATA column from the table.
    nodata_table = rfn.drop_fields(table, "DATA")
    # Copy the column definitions as a list.
    nodata_table_dt = nodata_table.dtype.descr
    # Concatenate the column definitions with the new data shape.
    new_dt = np.dtype(nodata_table_dt[:6] + [("DATA", ">f4", (nchan,))] + nodata_table_dt[6:])
    # Create a new table with the same number of rows.
    new_table = np.empty(nrow, dtype=new_dt)

    # Fill the new table with the old contents, 
    # and the DATA selection.
    for n in nodata_table.dtype.names:
        new_table[n] = nodata_table[n]
    new_table["DATA"] = new_data

    return new_table


def main(line_list_file, settings, stack_filename):
    """
    """

    log = pd.read_csv(line_list_file, delimiter='\t')
    files = log["file"][log["use"]].to_numpy()

    print(f"Stacking {len(files)} out of {len(log)}.")

    # Set stack size.
    nchan = int(abs(settings.vmin - settings.vmax)/settings.dv)
    sdfits = sd_fits.SDFITS()
    sdfits.load(files[0])
    ntime = sdfits.hdu[1].read_header()["NAXIS2"] 
    
    # Arrays for the stack.
    num = np.zeros((ntime, nchan), dtype=float)
    den = np.zeros((ntime, nchan), dtype=float)
    cnt = np.zeros((ntime, nchan), dtype=int)

    # Loop over files interpolating and adding to the stack.
    for i,f in enumerate(files):
    
        sdfits = sd_fits.SDFITS()
        sdfits.load(f)
        table = sdfits.hdu[1][:]
        freq = spectral_axis.compute_spectral_axis(table)
        lines = utils.find_rrls(freq[0],
                                settings.line, settings.vel_range,
                                z=settings.z)
        n = list(lines.keys())[0]
        new_table = interpolate2vel.run(freq, table,
                                        settings.vmin, settings.vmax, settings.dv,
                                        n, line=settings.line, applied_doppler=True)

        #rms = np.nanstd(new_table["DATA"], axis=1)
        nanloc = np.isnan(new_table["DATA"])
        data = new_table["DATA"]
        data[nanloc] = 0
        
        #weight = 1./rms[:,None]**2
        weight = (new_table["EXPOSURE"]/new_table["TSYS"]**2)[:,None]
        
        num += new_table["DATA"]*weight
        den += weight*~nanloc
        cnt += ~nanloc
    
    avg = num/den
    wei = den

    stack_table = update_table(new_table, avg, wei)
    stack_table["TSYS"] = 1.
    stack_table["EXPOSURE"] = np.nanmedian(wei, axis=1)

    # Convert to Galactic coordinates if needed.
    stack_table = eq2gal.eq2gal(stack_table)

    sd_fits_io.write_sdfits(stack_filename, stack_table)    


if __name__ == "__main__":

    import settings_800_CygX as settings
    #line_list_file = "/home/scratch/psalas/projects/CygnusX/data/target/hrrl-pipe-v5/line_list_man_02_Ra.txt"
    #output_file = "/home/scratch/psalas/projects/CygnusX/data/target/hrrl-pipe-v5/02_Ra_stack_texptsys.fits"
    #line_list_file = "/home/scratch/psalas/projects/CygnusX/data/target/hrrl-pipe-v5/02/Ra/line_list.txt"
    #output_file = "/home/scratch/psalas/projects/CygnusX/data/target/hrrl-pipe-v5/02_Ra_stack_texptsys_oldlist.fits"
    args = parse_args()
    main(args.logfile, settings, args.output)
