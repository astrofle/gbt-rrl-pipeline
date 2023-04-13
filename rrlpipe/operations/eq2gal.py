
from astropy import units as u
from astropy.coordinates import SkyCoord


def eq2gal(table):
    """
    """

    mask = (table["CTYPE2"] == "RA  ") & (table["CTYPE3"] == "DEC ")
    coo = SkyCoord(table["CRVAL2"][mask]*u.deg, table["CRVAL3"][mask]*u.deg, frame="fk5")

    glon = coo.galactic.l.deg
    glat = coo.galactic.b.deg

    table["CTYPE2"][mask] = "GLON"
    table["CRVAL2"][mask] = glon

    table["CTYPE3"][mask] = "GLAT"
    table["CRVAL3"][mask] = glat

    return table
