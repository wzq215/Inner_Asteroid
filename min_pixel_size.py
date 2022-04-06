import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoClip
import imageio
from PIL import Image
from Brightness_K_Corona import get_Brightness_of_K_Corona
from Brightness_F_Corona import get_Brightness_of_F_Corona
from Brightness_Asteroids import get_Brightness_of_Asteroid
import multiprocessing
import sys
import itertools


# ============Asteroid==============
'''I_aster单位: W/m**2'''
fullname_str = '163693 Atira (2003 CP20)' #'       (2020 AV2)' # 163693 Atira (2003 CP20)'
start_time = '2021-10-10'
stop_time = '2022-10-10'
[lon_aster,lat_aster,I_aster] = get_Brightness_of_Asteroid(fullname_str,start_time,stop_time)
steps = len(I_aster)


# =============Corona==============
'''corona单位：W/(m**2*sr)'''
pixel_size = [0.1,0.1]
lon_corona = np.arange(-60,60,pixel_size[0])
lat_corona = np.arange(-50,50,pixel_size[0])

# I_F = list(map(get_Brightness_of_F_Corona,itertools.product(lon_corona,lat_corona)))
# I_K = list(map(get_Brightness_of_K_Corona,itertools.product(lon_corona,lat_corona)))
# I_F = np.array(I_F.reshape(len(lon_corona),len(lat_corona)))
# I_K = np.array(I_K.reshape(len(lon_corona),len(lat_corona)))
# np.save('I_F('+str(resolution[0])+')',I_F)
# np.save('I_K('+str(resolution[1])+')',I_K)

B_F = np.load('I_F('+str(pixel_size[0])+').npy')
B_K = np.load('I_K('+str(pixel_size[0])+').npy')
B_F[np.isinf(B_F)]= np.nan
B_K[np.isinf(B_K)] = np.nan
AU = 1.49e11  # distance from sun to earth
Rs = 6.97e8  # solar radii


# ===============画图合成================
lon_1 = lon_corona[0]
lat_1 = lat_corona[0]
pixel_s = pixel_size[0]*pixel_size[1]
I_total = (B_F+B_K) * pixel_s
I_F = B_F * pixel_s
I_K = B_K * pixel_s
print(np.nanmax(I_F))
print(np.nanmax(I_K))
print(np.nanmax(I_aster))

print(np.nanmin(I_F))
print(np.nanmin(I_K))
print(np.nanmin(I_aster))


# I_max = np.nanmax([np.nanmax(I_F),np.nanmax(I_K),np.nanmax(I_aster)])
# I_min = np.nanmin([np.nanmin(I_F),np.nanmin(I_K),np.nanmin(I_aster)])
# print(I_max)
# print(I_min)
'''统一到W/m**2，corona亮度要在一个像素单元里积分，小天体不考虑跨像素的话应该暂时不用考虑面积的问题。'''
min_size_K = np.zeros((steps,1))
min_size_F = np.zeros((steps,1))
for i_tmp in range(steps):
    # 找到小天体在哪个像素里（假设不跨像素）
    Aster_m = int(np.floor((lon_aster[i_tmp]-lon_1)/pixel_size[0]))
    Aster_n = int(np.floor((lat_aster[i_tmp]-lat_1)/pixel_size[1]))

    # 单独小天体的视场图像
    I_aster_vof = I_total*np.nan
    I_aster_vof[Aster_m,Aster_n] = I_aster[i_tmp]

    # 总的辐射强度累加
    I_total_tmp = I_total
    I_total_tmp[Aster_m,Aster_n] = I_total[Aster_m,Aster_n] + I_aster[i_tmp]

    ratio = 1
    min_size_K[i_tmp] = np.sqrt(I_aster[i_tmp] * ratio / B_K[Aster_m,Aster_n])
    min_size_F[i_tmp] = np.sqrt(I_aster[i_tmp] * ratio / B_F[Aster_m,Aster_n])

print('max(min_size_K):')
print(np.max(min_size_K))
print('max(min_size_F):')
print(np.max(min_size_F))


