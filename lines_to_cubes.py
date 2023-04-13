
import os
import subprocess
import numpy as np
import pandas as pd

from rrlpipe import utils

#def grid_map_data(sdfitsfile, nx, ny, scale, xcntr, ycntr):
#    """
#    """
#
#    fileout = f"{os.path.splitext(sdfitsfile)[0]}"
#
#    args = ['gbtgridder', '--noline', '--nocont',
#            '-o', fileout, '--clobber',
#            '--size', f'{nx}', f'{ny}', '--pixelwidth', f'{scale}',
#            '--mapcenter', f'{xcntr}', f'{ycntr}',
#            '--kernel', 'gaussbessel',
#            sdfitsfile]
#    subprocess.run(args)
#
#    cubefile = f"{os.path.splitext(sdfitsfile)[0]}_cube.fits"
#
#    return cubefile


if __name__ == "__main__":

    path = "/home/scratch/psalas/projects/CygnusX/"
    line_list_file = f"{path}/data/target/hrrl-pipe-v5/line_list_man_23_Dec.txt"

    log = pd.read_csv(line_list_file, delimiter='\t')
    files = log["file"][log["use"]].to_numpy()

    import settings_340 as settings
    nx = settings.npix_x
    ny = settings.npix_y
    scale = settings.pix_scale
    xcntr = settings.x_cntr
    ycntr = settings.y_cntr

    for f in files:
        utils.grid_map_data(f, nx, ny, scale, xcntr, ycntr)
