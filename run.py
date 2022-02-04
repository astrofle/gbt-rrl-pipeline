
import numpy as np

from groundhog import sd_fits, spectral_axis

#import settings
from rrlpipe import utils
from rrlpipe.operations import calibrate, rfi_flag


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

            # Loop over lines.
            for n,chans in lines.items():
                print(f"Processing line: {settings.line} n={n}")
                table_rrl = utils.split_channel_range(table, chans[0], chans[1], dch=1)
                print("Calibrating scans.")
                table_rrl = calibrate.run(table_rrl, settings.cal_poly_order, settings.blank_lines)
                print("Flagging RFI.")
                rfi_file_path = f"{outdir}/spw_{ifnum}_pol_{plnum}_n_{n}"
                rfi_file = f"{rfi_file_path}_rfi.fits"
                table_rfi = rfi_flag.run(table_rrl, settings.lua_strategy, rfi_file_path, header)
                
                #try:
                #    print("Correcting bandpass.")
                #    rrlfile = utils.flatten_bandpass(rfi_file, v_line=settings.v_center, dv_line=settings.dv, poly_order=settings.bp_poly_order)
                #except TypeError:
                #    print("All channels flagged.")
                #    continue
                print("Producing report.")
                utils.make_report(rfi_file)
                proc_files.append(rfi_file)
                print("Gridding.")
                print(f"Will grid: {rfi_file}")
                cubefile = utils.grid_map_data(rfi_file, settings.npix_x, settings.npix_y, settings.pix_scale)
                cubefile = utils.freq2vel(cubefile, line=settings.line, z=settings.z)
                print(f"Gridded data is in: {cubefile}")

    
    # Make a report for the processed lines.
    line_vmin = settings.v_center - settings.dv
    line_vmax = settings.v_center + settings.dv
    line_list_file = f"{outdir}/line_list_{vbank}.txt"
    line_list = utils.make_line_list(proc_files, line_list_file, settings.rms_vmin, settings.rms_vmax, line_vmin, line_vmax)


if __name__ == "__main__":
    
    import settings_800 as settings

    for session in ['02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13']:
        for vbank in ['A', 'B']:
            for rod in ['Ra', 'Dec']:
                ifnums = settings.vbanks[vbank]
                plnums = [0,1]
                #rod = 'Dec'
                project = 'AGBT21A_292'
                projid = f'{project}_{session}'
                filein = f'/home/sdfits/{projid}/{projid}.raw.vegas/{projid}.raw.vegas.{vbank}.fits'
                outdir = f'/home/scratch/psalas/projects/CygnusX/data/tcal/hrrl-pipe-v2/{session}/{rod}/'
                pipeline(filein, outdir, ifnums, plnums, rod, vbank, settings)
