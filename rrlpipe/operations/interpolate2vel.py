
import numpy as np
import numpy.lib.recfunctions as rfn

from astropy import units as u

from crrlpy import crrls


def update_table(table, new_data, vaxis, applied_doppler=True):
    """
    """

    nrow = len(table["DATA"])
    nchan = len(vaxis)

    # Remove the DATA column from the table.
    nodata_table = rfn.drop_fields(table, "DATA")
    # Copy the column definitions as a list.
    nodata_table_dt = nodata_table.dtype.descr
    # Concatenate the column definitions with the new data shape.
    new_dt = np.dtype(nodata_table_dt[:6] + [("DATA", ">f4", (nchan,))] + nodata_table_dt[6:] + [("CUNIT1", "<U8")])
    # Create a new table with the same number of rows.
    new_table = np.empty(nrow, dtype=new_dt)

    # Fill the new table with the old contents, 
    # and the DATA selection.
    for n in nodata_table.dtype.names:
        new_table[n] = nodata_table[n]
    new_table["DATA"] = new_data
    # Update the frequency axis and bandwidth.
    new_table["CTYPE1"] = "VELO-LSR"
    new_table["CRPIX1"] = 0
    new_table["CRVAL1"] = vaxis[0]
    new_table["CDELT1"] = abs(vaxis[0] - vaxis[1])
    new_table["BANDWID"] = abs(vaxis[0] - vaxis[-1])
    new_table["CUNIT1"] = vaxis.unit.to_string()
    if applied_doppler:
        new_table['VFRAME'] = 0
    
    return new_table


def run(freq, table, vmin, vmax, dv, n, 
        line="RRL_HIalpha", applied_doppler=True,
        output_units='km/s'):
    """
    """

    # Find line frequency.
    fc = crrls.n2f(n, line) # MHz
    # Compute velocities.
    velo = crrls.freq2vel(fc, freq.to("MHz").value) # m/s
    # Axis to interpolate the velocities to.
    vel_axis = np.arange(vmin, vmax, dv)

    # Extract data.
    data = table["DATA"]
    nints = data.shape[0]
    nchan = data.shape[1]

    # New data array.
    interp_data = np.empty((nints, len(vel_axis)), dtype=data.dtype)

    # Loop over rows interpolating to the new velocity axis.
    r = 1
    for i,row in enumerate(data):

        if table["CDELT1"][i] > 0:
            r = -1
        else:
            r = 1

        # Average the data along the spectral axis if necessary.
        if np.mean(abs(np.diff(velo[i])))/dv >= 2:
            if len(velo[i])%2 == 0:
                a = 2
            elif len(velo[i])%3 == 0:
                a = 3
        else:
            a = 1

        x = velo[i,::r].reshape(-1,a).mean(axis=1)
        y = row[::r].reshape(-1,a).mean(axis=1)
        interp_data[i] = np.interp(vel_axis, x, y)

    vel_axis *= u.m/u.s
    vel_axis = vel_axis.to(output_units)

    # Update the table contents with the interpolated data
    # and velocity axis information.
    interp_table = update_table(table, interp_data, vel_axis, applied_doppler)

    return interp_table
