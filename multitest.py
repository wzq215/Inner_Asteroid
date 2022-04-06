#!/usr/local/anaconda3/bin/python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoClip
import imageio
from PIL import Image
from Brightness_K_Corona import get_Brightness_of_K_Corona
from Brightness_F_Corona import get_Brightness_of_F_Corona
# from Brightness_Asteroids import get_Brightness_of_Asteroid
from multiprocessing import Pool
import itertools
import sys


# ============Asteroid==============
# '''I_aster单位: W/m**2'''
# fullname_str = '163693 Atira (2003 CP20)' #'       (2020 AV2)' # 163693 Atira (2003 CP20)'
# start_time = '2021-10-10'
# stop_time = '2022-10-10'
# [lon_aster,lat_aster,I_aster] = get_Brightness_of_Asteroid(fullname_str,start_time,stop_time)
# steps = len(I_aster)


# =============Corona==============
'''corona单位：W/(m**2*sr)'''
pixel_size = [1,1]
lon_corona = np.arange(-60,60,pixel_size[0])
lat_corona = np.arange(-50,50,pixel_size[0])
grid = np.meshgrid(lon_corona,lat_corona)
## 跑得太慢，把数据存下来
# multiprocess

# cores = multiprocessing.cpu_count()
# pool = multiprocessing.Pool(processes=cores)

# cnt = 0
# loc_iter = []
# for i in range(0, len(lon_corona)):  # unit: degree.
#     for j in range(0, len(lat_corona)):
#         loc_iter.append(tuple(lon_corona[i],lat_corona[j]))

# loc_iter = itertools.product(lon_corona,lat_corona)
# print(loc_iter)
if __name__ == '__main__':
    with Pool(5) as p:
        I_F = []
        I_K = []
        I_F = p.map(get_Brightness_of_F_Corona, itertools.product(lon_corona,lat_corona))
        I_K = p.map(get_Brightness_of_K_Corona, itertools.product(lon_corona,lat_corona))
# I_F = list(map(get_Brightness_of_F_Corona,itertools.product(lon_corona,lat_corona)))
# I_K = list(map(get_Brightness_of_K_Corona,itertools.product(lon_corona,lat_corona)))
        I_F = np.array(I_F).reshape(len(lon_corona),len(lat_corona))
        I_K = np.array(I_K).reshape(len(lon_corona),len(lat_corona))


        np.save('I_F('+str(pixel_size[0])+')',I_F)
        np.save('I_K('+str(pixel_size[1])+')',I_K)
        B_F = np.load('I_F('+str(pixel_size[0])+').npy')
        B_K = np.load('I_K('+str(pixel_size[0])+').npy')

        plt.figure()
        B_Corona = np.array(B_F+B_K)
        plt.pcolormesh(lon_corona, lat_corona, np.log10(np.transpose(B_Corona)))
        plt.colorbar()
        plt.axis('scaled')
        plt.xlabel('view_field_theta (Degree, in ecliptic plane)')
        plt.ylabel('view_field_theta (Degree, perp to ecliptic plane)')
        plt.title('log10(B_corona)')
        plt.colorbar
        plt.show()
