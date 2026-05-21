MAX_ALLOWED_MOVEMENT = 40
PIXEL_LENGTH   = 4.651162790697675     # in um
INNER_DIAMETER = 4000/PIXEL_LENGTH     # pixels --> 4 mm = 4000 um / PIXEL_LENGTH
OUTER_DIAMETER = 8000/PIXEL_LENGTH     # pixels --> 8 mm = 8000 um / PIXEL_LENGTH
INNER_RADIUS   = int(INNER_DIAMETER/2) # pixels
OUTER_RADIUS   = int(OUTER_DIAMETER/2) # pixels

fovea_center_dict = {'L_06': (918, 1104), 'L_15': (1006, 987), 'L_26': (942, 937), 'L_30': (946, 946), 'L_42': (1016, 1008), 
 'L_48': (930, 958), 'L_63': (1078, 973), 'L_78': (949, 951), 'S_08': (977, 982), 'S_46': (920, 1123)}

