
# VEGAS settings.
vbanks = {'A':[0,1,2,3,4,5,6,7],
          'B':[8,9,10,11,12,13,14]
          }


# Line settings.
# Velocities in meters per second.
line = 'RRL_HIalpha'
vel_range = 500e3 # m/s
z = 0
v_center = 0.e3 # m/s
dv = 50e3 # m/s

# Calibration settings.
cal_poly_order = 11
blank_lines = {'RRL_HIalpha':[dv,z],
               'RRL_CIalpha':[dv,z]}
tcal_file = '/home/scratch/psalas/projects/CygnusX/data/tcal/tcal_800.npy'

# RFI flagging settings.
lua_strategy = '/home/scratch/psalas/projects/GDIGS-Low/gbt-rrl-pipeline/strategies/gbt-800.lua'

# Baseline correction settings.
bp_poly_order = 7

# Velocity range to compute RMS.
rms_vmin = -300e3
rms_vmax = -150e3

# Gridding settings.
npix_x = 237
npix_y = 198
pix_scale = 186
x_cntr = '307.47745833'
y_cntr = '41.49563889'
