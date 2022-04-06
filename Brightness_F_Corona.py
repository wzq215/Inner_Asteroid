import numpy as np
import matplotlib.pyplot as plt
from scipy import special as sp

'''
PROBLEMS:
'''

def obtain_VSF0(a, Nmr0, scatter_angle):
    wavelength = 550e-9  # m
    alpha = 2 * np.pi * a / wavelength
    albedo = 0.25  # bond albedo
    sigma = a ** 2 * abs(sp.jv(1, alpha * np.sin(np.deg2rad(scatter_angle)))) ** 2 / \
            abs(np.sin(np.deg2rad(scatter_angle))) ** 2 + albedo * a ** 2 / 4  # m^2
    delta_alpha = abs(np.mean(np.diff(alpha)))
    # vsf0 = np.nansum(sigma[1:] * (np.diff(Nmr0) / np.diff(alpha)) * delta_alpha)  # m^-1
    vsf0 = np.nansum((sigma[1:]+sigma[:-1])/2 * abs(np.diff(Nmr0)/np.diff(alpha)) * delta_alpha) # m^-1

    return vsf0

def get_Brightness_of_F_Corona(loc_angle):

    # Constants & Geometry
    I_0 = 1361  # W/m^2 Solar Irradiance
    AU = 1.49e11  # distance from sun to earth
    Rs = 6.96e8  # solar radii
    lon_angle = loc_angle[0]  # 经度方向上的方位角
    lat_angle = loc_angle[1]  # 纬度方向上的方位角
    elongation = np.rad2deg(np.arccos(np.cos(np.deg2rad(lat_angle))*np.cos(np.deg2rad(lon_angle)))) # deg
    scatter_angle = np.arange(elongation, 180, 1)
    l = np.cos(np.deg2rad(lat_angle))*np.cos(np.deg2rad(lon_angle)) - (1/np.tan(np.deg2rad(scatter_angle)))*np.sqrt(1-np.cos(np.deg2rad(lat_angle))**2*np.sin(np.deg2rad(lon_angle))**2) # AU along LOS
    cos2betas = 1-l**2*np.sin(np.deg2rad(lat_angle))**2/(1-2*l*np.cos(np.deg2rad(lat_angle))*np.cos(np.deg2rad(lon_angle))+l**2) # 相对于太阳中心参考系的纬度
    # r = AU * np.sin(np.deg2rad(lon_angle)) / np.sin(np.deg2rad(scatter_angle))

    # Size & Spatial Distribution of Dust
    a = np.arange(1e-9, 1e-6, 1e-8)  # Dust size (m)
    rho = 2.5e6  # g/m^3
    m = (4. / 3) * np.pi * a ** 3 * rho  # g
    c1 = 2.2e3; c2 = 15.; c3 = 1.3e-9; c4 = 1e11; c5 = 1e27; c6 = 1.3e-16; c7 = 1e6
    g1 = 0.306; g2 = -4.38; g3 = 2.; g4 = 4.; g5 = -0.36; g6 = 2.; g7 = -0.85
    Fmr0 = (c1 * m ** g1 + c2) ** g2 + c3 * (m + c4 * m ** g3 + c5 * m ** g4) ** g5 + c6 * (
                m + c7 * m ** g6) ** g7  # m^-2s^-1
    v0 = 20e3  # m/s
    Nmr0 = 4. * Fmr0 / v0  # m^-3
    print('Nmr0:',Nmr0)

    # VSF
    VSF0 = []
    for i in range(0, len(scatter_angle)):
        VSF0.append(obtain_VSF0(a, Nmr0, scatter_angle[i])) # m^-1
    # plt.axes(yscale = 'log')
    # plt.plot(scatter_angle,VSF0)
    # plt.xlabel('Scatter Angle')
    # plt.ylabel('VSF(r0,theta)')
    # plt.show()
    # LOS Integral
    nu = 1.3
    delta_angle = abs(np.mean(np.diff(scatter_angle)))
    I = I_0 * AU * (1 * np.sin(np.deg2rad(elongation))) ** (-nu - 1) * \
        np.nansum(np.sin(np.deg2rad(scatter_angle))**nu * (0.15 + 0.85*cos2betas**14) * np.array(VSF0) * np.deg2rad(delta_angle))
    return I


lon = np.arange(1,90,1)
lat = np.arange(1,30,1)
I_F = np.zeros((len(lon),len(lat)))
for i in range(0, len(lon)):  # unit: degree.
    for j in range(0, len(lat)):
        I_tmp = get_Brightness_of_F_Corona([lon[i],lat[j]])
        I_F[i,j] = I_tmp

[lonv,latv] = np.meshgrid(lon,lat)
I_F = np.array(I_F)
plt.pcolormesh(lon,lat, np.log10(np.transpose(I_F)))
plt.axis('scaled')
plt.xlabel('view_field_theta (Degree, in ecliptic plane)')
plt.ylabel('view_field_theta (Degree, perp to ecliptic plane)')
plt.title('log10(I_F)')
plt.show()

I_F = np.zeros((len(lon),1))
for i in range(0, len(lon)):
    I_tmp = get_Brightness_of_F_Corona([lon[i],0])
    I_F[i] = I_tmp
print(I_F[19])
plt.plot(lon, I_F)
plt.yscale('log')
plt.xlabel('view_field_theta (Degree, in ecliptic plane)')
plt.ylabel('White Light Intensity of F corona (W m^-2 sr^-1)')
plt.show()