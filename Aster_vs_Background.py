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


# ============Asteroid==============
'''I_aster单位: W/m**2'''
fullname_str = '       (1998 DK36)' #'       (2020 AV2)' # 163693 Atira (2003 CP20)
start_time = '2021-08-15'
stop_time = '2022-08-15'
[lon_aster,lat_aster,I_aster,deg2_aster] = get_Brightness_of_Asteroid(fullname_str,start_time,stop_time)
steps = len(I_aster)

# =============Corona==============
'''corona单位：W/(m**2*sr)'''
'''STEREO 分辨率：HI1 70''(0.0195deg) HI2 4 (0.067deg)' '''
pixel_size = [2,2]
lon_corona = np.arange(-60,60,pixel_size[0])
lat_corona = np.arange(-50,50,pixel_size[0])

B_F = np.load('I_F('+str(pixel_size[0])+').npy')
# B_K = np.load('I_K('+str(resolution[0])+').npy')
B_K = np.load('I_Kcorona.npy')
B_F[np.isinf(B_F)]= np.nan
B_K[np.isinf(B_K)] = np.nan
AU = 1.49e11  # distance from sun to earth
Rs = 6.97e8  # solar radii

# ===============画图合成================
lon_1 = lon_corona[0]
lat_1 = lat_corona[0]
# pixel_s = resolution[0]*resolution[1]*(np.pi/180)**2
pixel_s = 0.01*0.01*(np.pi/180)**2
I_total = (B_F+B_K) * pixel_s
I_F = B_F * pixel_s
I_K = B_K * pixel_s
print('Max&Min of I_F')
print(np.log10(np.nanmax(I_F)))
print(np.log10(np.nanmin(I_F)))
print('Max&Min of I_K')
print(np.log10(np.nanmax(I_K)))
print(np.log10(np.nanmin(I_K)))
print('Max&Min of I_aster')
print(np.log10(np.nanmax(I_aster)))
print(np.log10(np.nanmin(I_aster)))

'''统一到W/m**2，corona亮度要在一个像素单元里积分，小天体不考虑跨像素的话应该暂时不用考虑面积的问题。'''
def array2image(A,rgb,alpha):
    # A = np.uint8(A)
    A_imarray = np.uint8(np.zeros([np.shape(A)[0],np.shape(A)[1],4]))
    A_imarray[:,:,0] = A*rgb[0]
    A_imarray[:,:,1] = A*rgb[1]
    A_imarray[:,:,2] = A*rgb[2]
    A_imarray[:,:,3] = A*0+alpha

    A_im = Image.fromarray(A_imarray,'RGBA')
    return A_im

def normalize_image(A,max,min):
    A_new = 255*(A-min)/(max-min)
    A_new[A>max] = 255.
    A_new[A<min] = 0.
    A_new = np.uint8(A_new)
    return A_new

I_aster_vof = I_total*np.nan
Aster_F_ratio = I_aster*0
Aster_K_ratio = I_aster*0
start_timestamp = time.mktime(time.strptime(start_time,'%Y-%m-%d')) #'2021-08-15'
stop_timestamp = time.mktime(time.strptime(stop_time,'%Y-%m-%d'))

for i_tmp in range(steps):
    time_tmp = start_timestamp+(stop_timestamp-start_timestamp)*i_tmp/steps
    time_str = time.strftime('%Y-%m-%d',time.localtime(time_tmp))
    # 找到小天体在哪个像素里（假设不跨像素）
    Aster_m = int(np.floor((lon_aster[i_tmp]-lon_1)/pixel_size[0]))
    Aster_n = int(np.floor((lat_aster[i_tmp]-lat_1)/pixel_size[1]))
    if Aster_m < 0 or Aster_m > len(lon_corona):
        Aster_m = np.nan
    if Aster_n < 0 or Aster_n > len(lat_corona):
        Aster_n = np.nan

    # 单独小天体的视场图像
    # I_aster_vof = I_total*np.nan
    if ~np.isnan(Aster_n) and ~np.isnan(Aster_m):
        # print(Aster_n,Aster_m)
        I_aster_vof[Aster_m, Aster_n] = np.nanmax([I_aster[i_tmp],I_aster_vof[Aster_m,Aster_n]])
        Aster_F_ratio[i_tmp] = I_aster[i_tmp]/I_F[Aster_m,Aster_n]
        Aster_K_ratio[i_tmp] = I_aster[i_tmp]/I_K[Aster_m,Aster_n]

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(Aster_F_ratio,label='I_aster/I_Fcorona')
ax1.plot(Aster_K_ratio,label='I_aster/I_Kcorona')
ax1.legend(loc=2)
ax1.set_ylabel('Radiation Intensity Ratio')
ax1.set_xlabel('Time')
ax2 = ax1.twinx() # this is the important function
ax2.plot(deg2_aster/(np.pi/180)**2/1e-4,'k-',label = "Area of Asteroid")
ax2.legend(loc=1)
ax2.set_ylabel('Size (1e-4 deg^2)')
plt.title(fullname_str+'('+time_str+')')
plt.show()