import numpy as np
import matplotlib.pyplot as plt

'''
PROBLEMS:
(1) 需要改进电子分布
'''


def transform_to_xyz(R, theta, phi):
    x = R * np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi))
    y = R * np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi))
    z = R * np.cos(np.deg2rad(theta))
    return x, y, z


def transform_to_rtf(x, y, z):
    R = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(np.deg2rad(z / R))
    xy1 = np.intersect1d(np.where(x > 0), np.where(y > 0))
    xy2 = np.intersect1d(np.where(x < 0), np.where(y > 0))
    xy3 = np.intersect1d(np.where(x < 0), np.where(y < 0))
    xy4 = np.intersect1d(np.where(x > 0), np.where(y < 0))
    phi1 = np.arcsin(np.rad2deg(y / np.sqrt(x ** 2 + y ** 2)))
    phi2 = np.arccos(np.rad2deg(x / np.sqrt(x ** 2 + y ** 2)))
    phi3 = 180 - np.arcsin(np.deg2rad(y / np.sqrt(x ** 2 + y ** 2)))
    phi4 = 360 + np.arcsin(np.deg2rad(y / np.sqrt(x ** 2 + y ** 2)))
    phi = phi1 * 0
    phi[xy1] = phi1[xy1]
    phi[xy2] = phi2[xy2]
    phi[xy3] = phi3[xy3]
    phi[xy4] = phi4[xy4]
    return R, theta, phi


def obtain_grid(z, view_field_theta, view_field_phi):
    earth_R0 = 1
    earth_lon = 0
    earth_lat = 0
    earth_x0, earth_y0, earth_z0 = transform_to_xyz(earth_R0, earth_lon, 90 - earth_lat)
    volume_element_x = earth_x0 - z * np.cos(np.deg2rad(view_field_theta))
    volume_element_y = earth_y0 + z * np.sin(np.deg2rad(view_field_theta))
    volume_element_z = earth_z0 + z * np.tan(np.deg2rad(view_field_phi))
    volume_element_R, volume_element_theta, volume_element_phi = transform_to_rtf(volume_element_x, volume_element_y,
                                                                                  volume_element_z)
    volume_element_lat = 90 - volume_element_theta
    volume_element_lon = volume_element_phi
    return volume_element_R, volume_element_lon, volume_element_lat


def get_Brightness_of_K_Corona(loc_angle):
    view_field_theta = loc_angle[0]
    view_field_phi = loc_angle[1]
    I_0 = 1368 * 215 ** 2 / 2 / np.pi  # W/m^2/sr^1
    sigma_e = 7.95e-30
    u = 0.63
    AU = 1.49e11  # distance from sun to earth
    Rs = 6.97e8  # solar radii
    z = np.arange(0, 1e3, 0.5) * Rs
    # 根据lon_angle,lat_angle转换到theta_new
    edge1 = z * np.cos(np.deg2rad(view_field_phi)) * np.cos(np.deg2rad(view_field_theta))
    edge2 = z
    edge3 = np.sqrt((z * np.cos(np.deg2rad(view_field_phi)) * np.sin(np.deg2rad(view_field_theta))) ** 2 + (
                z * np.sin(np.deg2rad(view_field_phi))) ** 2)
    up = edge1 ** 2 + edge2 ** 2 - edge3 ** 2
    down = 2 * edge1 * edge2
    view_field_theta_new_in_3D = np.arccos(np.deg2rad(up / down))

    r2 = AU * np.sin(np.deg2rad(view_field_theta_new_in_3D))
    Z1 = abs(np.sqrt(AU ** 2 - r2 ** 2) - z)
    Z2 = np.sqrt(r2 ** 2 + Z1 ** 2)
    chi = np.arccos((Z2 ** 2 + z ** 2 - AU ** 2) / (2 * Z2 * z))
    Omega = np.arccos(np.sqrt(Z2 ** 2 - Rs ** 2) / Z2)

    # %生成视线上每小段的R,lon,lat(日心)，注意修改下面函数obtain_grid()中的地球位置。
    volume_element_R, volume_element_lon, volume_element_lat = obtain_grid(z, view_field_theta, view_field_phi)
    # %需要[volume_element_R,volume_element_lon,volume_element_lat]的密度数据

    c1 = 1.36e12; d1 = 2.14
    c2 = 1.68e14; d2 = 6.13
    Ne_r = c1 * (Z2 / Rs)**(-d1) + c2*(Z2/Rs)**(-d2) # choose a suitable radial distribution！

    delta_z = np.mean(np.diff(z))
    A = np.cos(Omega) * np.sin(Omega) ** 2
    B = -1 / 8. * (1 - 3 * np.sin(Omega) ** 2 - np.cos(Omega) ** 2 / np.sin(Omega) * (
            1 + 3 * np.sin(Omega) ** 2) * np.log((1 + np.sin(Omega) / np.cos(Omega))))
    C = 4 / 3 - np.cos(Omega) - np.cos(Omega) ** 2 / 3
    D = 1 / 8 * (5 + np.sin(Omega) ** 3 - np.cos(Omega) ** 2 / np.sin(Omega) * (5 - np.sin(Omega) ** 2) * np.log(
        (1 + np.sin(Omega)) / np.cos(Omega)))

    I_tot_coeff = I_0 * np.pi * sigma_e / 2 / z ** 2

    first_item = 2 * I_tot_coeff * ((1 - u) * C + u * D)
    second_item = - I_tot_coeff * np.sin(chi) ** 2 * ((1 - u) * A + u * B)
    I_tot = first_item + second_item

    I_at_per_z = Ne_r * z ** 2 * I_tot * delta_z
    I = np.nansum(I_at_per_z)

    return I

# lon = np.arange(1,90,1)
# lat = np.arange(1,30,1)
# I_F = np.zeros((len(lon),len(lat)))
# for i in range(0, len(lon)):  # unit: degree.
#     for j in range(0, len(lat)):
#         I_tmp = get_Brightness_of_K_Corona(lon[i],lat[j])
#         I_F[i,j] = I_tmp
#
# [lonv,latv] = np.meshgrid(lon,lat)
# I_F = np.array(I_F)
# plt.pcolormesh(lon,lat, np.log10(np.transpose(I_F)))
# plt.axis('scaled')
# plt.xlabel('view_field_theta (Degree, in ecliptic plane)')
# plt.ylabel('view_field_theta (Degree, perp to ecliptic plane)')
# plt.title('log10(I_K)')
plt.show()
# I_K = []
# lon_angle = np.arange(5, 50, 1)
# for theta in lon_angle:  # unit: degree.
#     I_K.append(obtain_Thomson_scattered_white_light_intersity(theta))
#
# print(I_K)
# print(lon_angle)
# plt.scatter(lon_angle, I_K);
# plt.xlabel('View\_field [\circ][angle between sight line and Sun-Earth line]')
# plt.ylabel('Thomson-scattering integral along LOSs')
# plt.show()
