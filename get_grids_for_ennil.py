import numpy as np
R1 = 0.141564
R2 = 1.13845
Rs = np.linspace(R1,R2,319)
# for r in Rs:
#     print('%.6f'%r,end=',')
lon1 = 1
lon2 = 359
lons = np.linspace(lon1,lon2,180)
# for lon in lons:
#     print('%.1f'%lon,end=',')
lat1 = -59
lat2 = 59
lats = np.linspace(lat1,lat2-2,59)
for lat in lats:
    print('%.1f'%lat,end=',')