
# VEGAS settings.
vbanks = {'A':[0,1,2,3,4,5,6,7],
          'B':[8,9,10,11,12,13,14]
          }

# Calibration settings.
cal_poly_order = 11
blank_lines = {'RRL_HIalpha':[50e3,0],
               'RRL_CIalpha':[50e3,0]}

# Line settings.
# Velocities in meters per second.
line = 'RRL_HIalpha'
vel_range = 500e3 # m/s
z = 0.
v_center = 0 # m/s
dv = 50e3 # m/s

# RFI flagging settings.
lua_strategy = '/home/scratch/psalas/projects/CygnusX/hrrl-pipeline/strategies/gbt-800.lua'

# Baseline correction settings.
bp_poly_order = 7

# Velocity range to compute RMS.
rms_vmin = -300e3
rms_vmax = -150e3

# Gridding settings.
npix_x = 237
npix_y = 198
pix_scale = 186

