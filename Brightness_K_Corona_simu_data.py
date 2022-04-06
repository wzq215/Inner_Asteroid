#!/usr/local/anaconda3/bin/python
"""
林荣 20211110
"""
import pandas as pd
import numpy as np
from scipy import interpolate

sin = np.sin
cos = np.cos
sqrt = np.sqrt
arctan = np.arctan
pi = np.pi
AU_unity = 1
deg2rad = pi / 180
# apply by '*' but not '/'

# 1. load simulation data (get Ne)
print('LOAD simulation_data')
names = ['r','lon','lat','N*r^2']
data = pd.read_csv(
    r'WSA_Enlil_Data/pointdata_693555030377_20210829_000140.txt',
    sep='\s+')
# '/' is for mac users, if Windows, use '\'
data['N'] = data['N*r^2']/data['r']/data['r']
print(data)
# Unit of N: cm^-3
# Coordinate: HEEQ (earth_lon = 180 deg.)

r_list = sorted(set(data['r']))
lon_list = sorted(set(data['lon']))
lat_list = sorted(set(data['lat']))
N_data = np.array(data['N'])
reshaped_N_3D = N_data.reshape([len(lat_list),len(lon_list),len(r_list)])

print("reshaped_N_3D.shape =",reshaped_N_3D.shape)
reshaped_N_3D = reshaped_N_3D[:, :, 1:]  # eliminate NaN at r=r_min
r_list = r_list[1:] # eliminate NaN at r=r_min
print("reshaped_N_3D.shape =",reshaped_N_3D.shape)
reshaped_N_3D = np.concatenate([reshaped_N_3D[:,-2:-1,:],reshaped_N_3D,reshaped_N_3D[:,0:1,:]],axis=1)
print("reshaped_N_3D.shape =",reshaped_N_3D.shape)
lon_list.append(361.0)
lon_list.insert(0, -1.0)
print("len(lon_list) =",len(lon_list))

meshed_lon,meshed_lat, meshed_r = np.meshgrid( lon_list,lat_list, r_list)
points = np.array([meshed_lat.T, meshed_lon.T, meshed_r.T])
print("interpolation points.shape =",points.shape)
points = points.T
print("points.shape =",points.shape)
points_flatten = points.reshape(-1,3)
print("points_flatten.shape =",points_flatten.shape)
reshaped_N_3D_flatten = reshaped_N_3D.reshape(-1)
print("reshaped_N_3D_flatten.shape =",reshaped_N_3D_flatten.shape)

# 2. calculate K corona brightness
I_0 = 1361 * 215**2. / np.pi  # pi
sigma_e = 7.95e-30
u = 0.63
AU = 1.49e11
Rs = 6.97e8

def obtain_Thomson_scattered_white_light_intersity(view_field_theta, view_field_phi, z_max_new):
    cos_psi = cos(view_field_theta)*cos(view_field_phi)
    sin_psi = sqrt(1-cos_psi**2)
    delta_z = Rs
    z = np.arange(delta_z, z_max_new, delta_z)
    
    z_proj = z*cos(view_field_phi)
    r2 = AU * np.sin(view_field_theta)
    Z1 = AU * np.cos(view_field_theta) - z_proj
    Z2_proj = np.sqrt(r2**2 + Z1**2)
    Z2 = np.sqrt(Z2_proj**2+(z*sin(view_field_phi))**2)
    HEEQ_lat_deg = np.arccos(Z2_proj/Z2)* 180 / np.pi
    
    chi = np.arccos((Z2_proj**2 + z_proj**2 - AU**2) / (2. * Z2_proj * z_proj))
    if view_field_theta > 0:
        HEEQ_lon_deg = (2 * np.pi - chi - view_field_theta) * 180 / np.pi
        # 如果地球的位置不在180，需要修改此处的2*pi.
    else:
        HEEQ_lon_deg = (chi - view_field_theta) * 180 / np.pi
    Omega = np.arccos(np.sqrt(Z2**2 - Rs**2) / Z2)
    sin_Omega = np.sin(Omega)
    cos_Omega = np.cos(Omega)

    Z2_in_AU = Z2 / AU
    interp_points = np.array([HEEQ_lat_deg, HEEQ_lon_deg, Z2_in_AU])
    interp_points = interp_points.T
    Ne_r = interpolate.griddata(points_flatten, reshaped_N_3D_flatten,
                                interp_points,'nearest')
    Ne_r *= 1e6  # from cm^-3 to m^-3

    A = cos_Omega * sin_Omega**2
    B = -1 / 8 * (1 - 3 * sin_Omega**2 - cos_Omega**2 / sin_Omega *
                  (1 + 3 * sin_Omega**2) * np.log((1 + sin_Omega / cos_Omega)))
    C = 4 / 3 - cos_Omega - cos_Omega**3 / 3
    D = 1 / 8 * (5 + sin_Omega**2 - cos_Omega**2 / sin_Omega *
                 (5 - sin_Omega**2) * np.log((1 + sin_Omega) / cos_Omega))

    I_tot_coeff = I_0 * np.pi * sigma_e / 2 / z**2
    first_item = 2 * I_tot_coeff * ((1 - u) * C + u * D)
    second_item = -I_tot_coeff * np.sin(chi)**2 * ((1 - u) * A + u * B)
    I_tot = first_item + second_item
    I_at_per_z = Ne_r * z**2 * I_tot * delta_z
    I = np.nansum(I_at_per_z)
    return I


def alpha(theta, phi, z):
    temp = z * abs(sin(phi))
    temp = temp / sqrt(AU**2 +
                       (z * cos(phi))**2 - 2 * AU * z * cos(phi) * cos(theta))
    temp = arctan(temp)
    return temp

if __name__=='__main__':
    from matplotlib import pyplot as plt
    import time

    thetas = np.arange(-60,60,1)*deg2rad
    phis = np.arange(-50,50,1)*deg2rad
    z_max_matrix = np.zeros((len(thetas),len(phis)))

    for i,theta in enumerate(thetas):
        for j,phi in enumerate(phis):
            cos_psi = cos(phi) * cos(theta)
            sin_psi = sqrt(1-cos_psi**2)
            z_max = AU * cos_psi + np.sqrt((1.14*AU)**2 - (AU*sin_psi)**2)
            if z_max < AU / cos_psi:
                alpha_critical = alpha(theta, phi, z_max)
            else:
                alpha_critical = alpha(theta, phi, AU / cos_psi)
            if alpha_critical < 59*deg2rad:
                pass
            else:
                z_max = np.nan
            z_max_matrix[i,j] = z_max
    
    fig = plt.figure()
    _ = plt.pcolormesh(thetas/deg2rad,phis/deg2rad,z_max_matrix.T/AU)
    _ = plt.colorbar()
    _ = plt.title('z_max (AU)')
    _ = plt.ylabel(r'$\phi$ (Lon. from Earth)')
    _ = plt.xlabel(r'$\theta$ (Lat. from Earth)')
    _ = plt.savefig('available_z')

    I_matrix = np.zeros((len(thetas),len(phis)))
    start = time.time()
    print(f'x / {len(thetas)}:')
    for i,theta in enumerate(thetas):
        for j,phi in enumerate(phis):
            print(j, end=' ')
            if np.isnan(z_max_matrix[i,j]):
                I = np.nan
            else:
                I = obtain_Thomson_scattered_white_light_intersity(theta,phi,z_max_matrix[i,j]) 
            I_matrix[i,j] = I
        print('\n'+str(i)+'/')
    end = time.time()

    print("it costs %.2f second(s) to calculate."%(end - start)) # about 90min.

    fig = plt.figure()
    pclr = plt.pcolormesh(thetas/deg2rad,phis/deg2rad,np.log10(I_matrix.T))
    cbar = plt.colorbar(pclr)
    _ = plt.title('log I (log Wm^-2sr^-1)')
    _ = plt.ylabel(r'$\phi$ (Lon. from Earth)')
    _ = plt.xlabel(r'$\theta$ (Lat. from Earth)')
    _ = plt.savefig('I_Kcorona')

    np.save('I_Kcorona_1deg.npy',I_matrix)

"""
output case
(venv) (base)  ✘ linrong@x86_64-apple-darwin13  ~/Desktop/内日球层小天体  python Brightness_K_Corona_simu_data.py
LOAD simulation_data
         r  lon   lat   N*r^2           N
0  0.14156  1.0 -59.0     NaN         NaN
1  0.14470  1.0 -59.0  10.625  507.448387
2  0.14783  1.0 -59.0  10.526  481.657372
3  0.15097  1.0 -59.0  10.385  455.643512
4  0.15410  1.0 -59.0  10.274  432.647585
reshaped_N_3D.shape = (60, 180, 319)
reshaped_N_3D.shape = (60, 180, 318)
reshaped_N_3D.shape = (60, 182, 318)
len(lon_list) = 182
interpolation points.shape = (3, 318, 182, 60)
points.shape = (60, 182, 318, 3)
points_flatten.shape = (3472560, 3)
reshaped_N_3D_flatten.shape = (3472560,)
Brightness_K_Corona_simu_data.py:109: RuntimeWarning: invalid value encountered in sqrt
  (z * cos(phi))**2 - 2 * AU * z * cos(phi) * cos(theta))
Brightness_K_Corona_simu_data.py:109: RuntimeWarning: divide by zero encountered in double_scalars
  (z * cos(phi))**2 - 2 * AU * z * cos(phi) * cos(theta))
Brightness_K_Corona_simu_data.py:109: RuntimeWarning: invalid value encountered in double_scalars
  (z * cos(phi))**2 - 2 * AU * z * cos(phi) * cos(theta))
x / 60:
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
0/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
1/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
2/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
3/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
4/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
5/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
6/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
7/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
8/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
9/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
10/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
11/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
12/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
13/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
14/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
15/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
16/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
17/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
18/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
19/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
20/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
21/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
22/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
23/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
24/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
25/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
26/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
27/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
28/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
29/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
30/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
31/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
32/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
33/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
34/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
35/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
36/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
37/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
38/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
39/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
40/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
41/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
42/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
43/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
44/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
45/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
46/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
47/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
48/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
49/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
50/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
51/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
52/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
53/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
54/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
55/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
56/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
57/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
58/
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
59/
it costs 2192.37 second(s) to calculate.
"""