
import numpy as np

from astropy import units as u

from groundhog import sd_fits, spectral_axis

from rrlpipe import utils
from rrlpipe.operations import calibrate, rfi_flag, stack


rod_dict = {'Ra': 'RALongMap',
            'Dec': 'DecLatMap'}


def pipeline(filein, outdir, ifnums, plnums, rod, vbank, settings):
    """
    """

    # Go through the steps.

    # Load the raw data.
    print(f"Loading {filein}")
    sdfits = sd_fits.SDFITS()
    sdfits.load(filein)
    sdfits.make_summary()
    header = sdfits.hdu[1].read_header()

    # Select the mapping scans for a given direction.
    print(f"Selecting mapping scans.")
    mask = (np.asarray(sdfits._summary.Proc) == rod_dict[rod])
    if mask.sum() == 0:
        print("No records found for the mapping type.")
        return
    sscans = np.asarray(sdfits._summary.Scan)
    map_scans = np.unique(sscans[mask])
    print(f"Will process the following scans: {map_scans}")

    # Load the user computed Tcal values.
    if settings.tcal_file is not None:
        print("Loading Tcal values.")
        tcal_arr = np.load(settings.tcal_file)

    proc_files = []
    # Loop over spectral window and polarization.
    for ifnum in ifnums:
        for plnum in plnums:

            print(f"ifnum: {ifnum} - plnum: {plnum}")

            # Select table rows.
            table = sdfits.get_rows(1, map_scans, ifnum=ifnum, plnum=plnum)
            # Get frequency axis.
            freq = spectral_axis.compute_freq_axis(table)
            # Find RRLs.
            print("Finding RRLs.")
            lines = utils.find_rrls(freq[0], settings.line, settings.vel_range, z=settings.z)

            tcal = tcal_arr[ifnum,plnum]

            # Loop over lines.
            for n,chans in lines.items():
                print(f"Processing line: {settings.line} n={n}")
                table_rrl = utils.split_channel_range(table, chans[0], chans[1], dch=1)
                print("Calibrating scans.")
                table_rrl = calibrate.run(table_rrl, settings.cal_poly_order, settings.blank_lines, tcal=tcal)
                print("Flagging RFI.")
                rfi_file_path = f"{outdir}/spw_{ifnum}_pol_{plnum}_n_{n}"
                rfi_file = f"{rfi_file_path}_rfi.fits"
                table_rfi = rfi_flag.run(table_rrl, settings.lua_strategy, rfi_file_path, header)
                
                print("Producing report.")
                utils.make_report(rfi_file)
                proc_files.append(rfi_file)
                print("Gridding.")
                print(f"Will grid: {rfi_file}")
                cubefile = utils.grid_map_data(rfi_file, 
                                               settings.npix_x, 
                                               settings.npix_y, 
                                               settings.pix_scale,
                                               settings.x_cntr,
                                               settings.y_cntr)
                cubefile = utils.freq2vel(cubefile, line=settings.line, z=0, qnidx=0)
                print(f"Gridded data is in: {cubefile}")

    
    ## Make a report for the processed lines.
    #line_vmin = settings.v_center - settings.dv
    #line_vmax = settings.v_center + settings.dv
    #line_list_file = f"{outdir}/line_list_{vbank}.txt"
    #line_list = utils.make_line_list(proc_files, line_list_file, settings.rms_vmin, settings.rms_vmax, line_vmin, line_vmax)


if __name__ == "__main__":
    
    import settings_800_01 as settings

    project = 'AGBT22A_437'

    for session in ['01']:
        for rod in ['Dec']:
            for vbank in ['A', 'B']:
                ifnums = settings.vbanks[vbank]
                plnums = [0,1]
                projid = f'{project}_{session}'
                filein = f'/home/sdfits/{projid}/{projid}.raw.vegas/{projid}.raw.vegas.{vbank}.fits'
                outdir = f'/home/scratch/psalas/projects/GDIGS-Low/data/target-v2/{session}/{rod}/'
                pipeline(filein, outdir, ifnums, plnums, rod, vbank, settings)

            print('Stacking.')
            stack.run(outdir, f'{outdir}/line_list.txt', -300e3*u.m/u.s, 300e3*u.m/u.s, 500*u.m/u.s,
                      settings.rms_vmin*u.m/u.s, settings.rms_vmax*u.m/u.s, 
                      (settings.v_center - settings.dv)*u.m/u.s, 
                      (settings.v_center + settings.dv)*u.m/u.s, 
                      output='stack.fits')
