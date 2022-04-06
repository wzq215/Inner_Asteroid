import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from moviepy.editor import VideoClip
import imageio
from PIL import Image
import time
from Brightness_K_Corona import get_Brightness_of_K_Corona
from Brightness_F_Corona import get_Brightness_of_F_Corona
from Brightness_Asteroids import get_Brightness_of_Asteroid
import multiprocessing
import sys
import itertools
from Asteroid_positions import get_aster_pos, get_body_pos


# ============Asteroid==============
'''I_aster单位: W/m**2'''
fullname_str = '163693 Atira (2003 CP20)'#'       (1998 DK36)' #'       (2020 AV2)' # 163693 Atira (2003 CP20)'
start_time = '2021-10-01'
stop_time = '2022-02-01'
[lon_aster,lat_aster,I_aster,deg2_aster] = get_Brightness_of_Asteroid(fullname_str,start_time,stop_time)
steps = len(I_aster)

print('==========Asteroid==========')
print('Target: '+fullname_str)
print('Time Range: '+start_time+'_'+stop_time)
print('Steps: '+str(steps))
print('============================')

# =============Corona==============
'''corona单位：W/(m**2*sr)'''
'''STEREO 分辨率：HI1 70''(0.0195deg) HI2 4 (0.067deg)' '''
resolution = [1, 1]
lon_corona = np.arange(-60, 60, resolution[0])
lat_corona = np.arange(-50, 50, resolution[0])
print('Resolution: '+str(resolution[0])+'*'+str(resolution[1]))
print('Lontitude Range: '+str(lon_corona[0])+'_'+str(lon_corona[-1]))
print('Latitude Range: '+str(lat_corona[0])+'_'+str(lat_corona[-1]))

# I_F = list(map(get_Brightness_of_F_Corona,itertools.product(lon_corona,lat_corona)))
# I_K = list(map(get_Brightness_of_K_Corona,itertools.product(lon_corona,lat_corona)))
# I_F = np.array(I_F).reshape(len(lon_corona),len(lat_corona))
# I_K = np.array(I_K).reshape(len(lon_corona),len(lat_corona))
# np.save('I_F('+str(resolution[0])+')',I_F)
# np.save('I_K('+str(resolution[0])+')',I_K)

B_F = np.load('I_F(' + str(resolution[0]) + ').npy')
# B_K = np.load('I_K('+str(resolution[0])+').npy')
B_K = np.load('I_Kcorona_1deg.npy')
B_F[np.isinf(B_F)]= np.nan
B_K[np.isinf(B_K)] = np.nan
AU = 1.49e11  # distance from sun to earth
Rs = 6.97e8  # solar radii

# ===============画图合成================
lon_1 = lon_corona[0]
lat_1 = lat_corona[0]
# pixel_s = resolution[0]*resolution[1]*(np.pi/180)**2
pixel_size = [0.005,0.005]
print('Pixel Size: '+str(pixel_size[0])+'*'+str(pixel_size[1]))
pixel_s = pixel_size[0]*pixel_size[1]*(np.pi/180)**2
I_total = (B_F+B_K) * pixel_s
I_F = B_F * pixel_s # W/m^2
I_K = B_K * pixel_s # W/m^2

Magnitude_F = 2.5*np.log10(1368/I_F)-26.74
Magnitude_K = 2.5*np.log10(1368/I_K)-26.74
Magnitude_A = 2.5*np.log10(1368/I_aster)-26.74

print('====================')
print('Max&Min of Magnitude_F')
print(np.nanmax(Magnitude_F))
print(np.nanmin(Magnitude_F))
print('Max&Min of Magnitude_K')
print(np.nanmax(Magnitude_K))
print(np.nanmin(Magnitude_K))
print('Max&Min of Magnitude_Aster')
print(np.nanmax(Magnitude_A))
print(np.nanmin(Magnitude_A))

df = pd.read_csv('data/sbdb_query_results.csv')
aster_df = df[df['full_name']==fullname_str]
# print(aster_df)
spkid = aster_df.loc[:,'spkid'].values[0]
aster_positions = get_aster_pos(spkid, start_time, stop_time, observer='SUN', frame='HCI')
earth_positions = get_body_pos('EARTH', start_time, stop_time, observer='SUN', frame='HCI')
fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection='3d')
im=ax.scatter(aster_positions[0], aster_positions[1], aster_positions[2],c=Magnitude_A,cmap='gist_rainbow',s=50*np.linspace(0,steps,steps)/steps)
plt.colorbar(im)
# ax.colorbar(mappable=plt.cm.ScalarMappable(norm=Magnitude_A[100:140],cmap='rainbow'),ax=ax)
ax.scatter(earth_positions[0], earth_positions[1], earth_positions[2], c='blue',s=50*np.linspace(0,steps,steps)/steps)
# im=ax.scatter(aster_positions[0,100:140], aster_positions[1,100:140], aster_positions[2,100:140],c=Magnitude_A[100:140],cmap='rainbow')
# plt.colorbar(im)
# # ax.colorbar(mappable=plt.cm.ScalarMappable(norm=Magnitude_A[100:140],cmap='rainbow'),ax=ax)
# ax.scatter(earth_positions[0,100:140], earth_positions[1,100:140], earth_positions[2,100:140], c='blue')
# ax.scatter(aster_positions[0,100], aster_positions[1,100], aster_positions[2,100], s=100,c='red')
# ax.scatter(earth_positions[0,100], earth_positions[1,100], earth_positions[2,100], s=100,c='blue')
# for i in range(1):
#     index = i+120
#     ax.plot([earth_positions[0,index],aster_positions[0,index]],[earth_positions[1,index],aster_positions[1,index]],[earth_positions[2,index],aster_positions[2,index]],'k--')
# ax.colorbar()
ax.scatter(0, 0, 0, c='red')
ax.set_xlabel('X (AU)')
ax.set_ylabel('Y (AU)')
ax.set_zlabel('Z (AU)')
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])
plt.title('SPKID=' + str(spkid) + '(' + start_time + '_' + stop_time + ')')
plt.show()