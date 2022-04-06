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
start_time = '2021-08-15'
stop_time = '2022-08-15'
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
aster_positions = get_aster_pos(spkid, start_time, stop_time, observer='SUN', frame='HEEQ')
earth_positions = get_body_pos('EARTH', start_time, stop_time, observer='SUN', frame='HEEQ')
fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(aster_positions[0], aster_positions[1], aster_positions[2],c=Magnitude_A)
ax.scatter(earth_positions[0], earth_positions[1], earth_positions[2], c='blue')
ax.plot([earth_positions[0,0],aster_positions[0,0]],[earth_positions[1,0],aster_positions[1,0]],[earth_positions[2,0],aster_positions[2,0]],'k--')
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
    A_new[A_new<0] = 0.
    A_new[A_new>255] = 255.
    A_new = np.uint8(A_new)
    return A_new

Magnitude_aster_vof = I_total*np.nan
I_aster_vof =  I_total*np.nan
Aster_F_ratio = I_aster*0
Aster_K_ratio = I_aster*0
start_timestamp = time.mktime(time.strptime(start_time,'%Y-%m-%d')) #'2021-08-15'
stop_timestamp = time.mktime(time.strptime(stop_time,'%Y-%m-%d'))

for i_tmp in range(steps):
    time_tmp = start_timestamp+(stop_timestamp-start_timestamp)*i_tmp/steps
    time_str = time.strftime('%Y-%m-%d',time.localtime(time_tmp))
    # 找到小天体在哪个像素里（假设不跨像素）
    Aster_m = int(np.floor((lon_aster[i_tmp]-lon_1) / resolution[0]))
    Aster_n = int(np.floor((lat_aster[i_tmp]-lat_1) / resolution[1]))
    if Aster_m < 0 or Aster_m > len(lon_corona):
        Aster_m = np.nan
    if Aster_n < 0 or Aster_n > len(lat_corona):
        Aster_n = np.nan

    # 单独小天体的视场图像
    # I_aster_vof = I_total*np.nan
    if ~np.isnan(Aster_n) and ~np.isnan(Aster_m):
        # print(Aster_n,Aster_m)
        Magnitude_aster_vof[Aster_m, Aster_n] = np.nanmax([Magnitude_A[i_tmp],Magnitude_aster_vof[Aster_m,Aster_n]])
        I_aster_vof[Aster_m, Aster_n] = np.nanmax([I_aster[i_tmp],I_aster_vof[Aster_m,Aster_n]])
        Aster_F_ratio[i_tmp] = I_aster[i_tmp]/I_F[Aster_m,Aster_n]
        Aster_K_ratio[i_tmp] = I_aster[i_tmp]/I_K[Aster_m,Aster_n]

    I_max = 15
    I_min = 20
    norm_F = normalize_image(np.transpose(Magnitude_F),I_max,I_min)
    norm_K = normalize_image(np.transpose(Magnitude_K),I_max,I_min)
    norm_A = normalize_image(np.transpose(Magnitude_aster_vof),I_max,I_min)

    # =========分别保存成三张图，透明度合成==========
    alpha = 255//2
    R_F_im = array2image(norm_F,[1,0,0],alpha)
    G_K_im = array2image(norm_K,[0,0,1],alpha)
    B_A_im = array2image(norm_A,[0,1,0],alpha)

    composited = Image.new('RGBA',(len(lon_corona),len(lat_corona)))
    composited = Image.alpha_composite(R_F_im,G_K_im)
    composited = Image.alpha_composite(composited,B_A_im) 
    composited = composited.resize((960,800))
    plt.figure()
    plt.imshow(composited)
    plt.xticks([0,480,960],[-60,0,60])
    plt.yticks([0,400,800],[-50,0,50])
    plt.title(fullname_str+'('+time_str+')')
    plt.savefig('frames/'+fullname_str+str(i_tmp)+'.png')
    plt.close()


    # I_total = np.nan_to_num(I_F)+np.nan_to_num(I_K)+np.nan_to_num(I_aster_vof)
    I_total = np.nan_to_num(I_K)+np.nan_to_num(I_aster_vof)
    Magnitude_total = 2.5*np.log10(1368/I_total)-26.74
    # print(np.nanmax(Magnitude_total))
    grayscale = normalize_image(np.transpose(Magnitude_total),15,20)
    gray_im = array2image(grayscale,[1,1,1],255)
    gray_im = gray_im.resize((960,800))

    plt.figure()
    plt.imshow(gray_im)
    plt.xticks([0,480,960],[-60,0,60])
    plt.yticks([0,400,800],[-50,0,50])
    plt.title(fullname_str+'('+time_str+')')
    plt.savefig('frames/gray'+fullname_str+str(i_tmp)+'.png')
    plt.close()

    # composited.save('frames/'+str(i_tmp)+'.png')

# 画成动画
frames=[]
for i_tmp in range(steps):
    frames.append(imageio.imread('frames/'+fullname_str+str(i_tmp)+'.png'))
imageio.mimsave(fullname_str+'('+start_time+'-'+stop_time+')'+'.mp4',frames,fps=12)

frames=[]
for i_tmp in range(steps):
    frames.append(imageio.imread('frames/gray'+fullname_str+str(i_tmp)+'.png'))
imageio.mimsave('gray'+fullname_str+'('+start_time+'-'+stop_time+')'+'.mp4',frames,fps=12)

# plt.figure()
# plt.plot(Aster_F_ratio,label='I_aster/I_Fcorona')
# plt.plot(Aster_K_ratio,label='I_aster/I_Kcorona')
# plt.legend()
# plt.title('Pixel size = 0.01deg * 0.01deg')
# plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(Aster_F_ratio,label='I_aster/I_Fcorona')
ax1.plot(Aster_K_ratio,label='I_aster/I_Kcorona')
ax1.legend()
ax1.set_ylabel('Radiation Intensity Ratio')
ax1.set_xlabel('Time')
ax2 = ax1.twinx() # this is the important function
ax2.plot(deg2_aster/(np.pi/180)**2/1e-4,'k-',label = "Area of Asteroid")
ax2.legend()
ax2.set_ylabel('Size (1e-4 deg^2)')
plt.show()