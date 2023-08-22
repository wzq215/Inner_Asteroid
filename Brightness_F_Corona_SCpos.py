import numpy as np
import matplotlib.pyplot as plt
from scipy import special as sp

I_0 = 1361  # W/m^2 Solar Irradiance
AU = 1.49e11  # distance from sun to earth
Rs = 6.96e8  # solar radii


def obtain_VSF0(a, Nmr0, theta_rad):
    wavelength = 550e-9  # m
    alpha = 2 * np.pi * a / wavelength
    albedo = 0.25  # bond albedo
    sigma = a ** 2 * abs(sp.jv(1, alpha * np.sin(theta_rad))) ** 2 / \
            abs(np.sin(theta_rad)) ** 2 + albedo * a ** 2 / 4  # m^2
    delta_alpha = abs(np.mean(np.diff(alpha)))
    # vsf0 = np.nansum(sigma[1:] * (np.diff(Nmr0) / np.diff(alpha)) * delta_alpha)  # m^-1
    vsf0 = np.nansum((sigma[1:] + sigma[:-1]) / 2 * abs(np.diff(Nmr0) / np.diff(alpha)) * delta_alpha)  # m^-1

    return vsf0


def get_Brightness_of_F_Corona(fov_angle, SC_pos_carr):
    # Constants & Geometry
    beta_rad = fov_angle[0]  # 经度方向上的方位角
    gamma_rad = fov_angle[1]  # 纬度方向上的方位角

    cos_elongation = (1 + np.tan(beta_rad) ** 2 + np.tan(gamma_rad) ** 2) ** (-1 / 2)
    elongation_rad = np.arccos(cos_elongation)

    theta_rad = np.linspace(elongation_rad, np.pi, 180)

    x_sc_carr, y_sc_carr, z_sc_carr = SC_pos_carr[0], SC_pos_carr[1], SC_pos_carr[2]
    R_sc = np.sqrt(x_sc_carr ** 2 + y_sc_carr ** 2 + z_sc_carr ** 2)

    l_los = R_sc * np.sin(theta_rad - elongation_rad) / np.sin(theta_rad)

    carr2fov_arr = np.array(
        [[-x_sc_carr * z_sc_carr / R_sc ** 2, -y_sc_carr * z_sc_carr / R_sc ** 2, 1 - z_sc_carr ** 2 / R_sc ** 2],
         [y_sc_carr / R_sc, -x_sc_carr / R_sc, 0],
         [x_sc_carr / R_sc, y_sc_carr / R_sc, z_sc_carr / R_sc]])
    # print(carr2fov_arr)
    if x_sc_carr == 0 and y_sc_carr == 0:
        fov2carr_arr = np.array([[1,0,0],[0,1,0],[0,0,1]])
    else:
        fov2carr_arr = np.linalg.inv(carr2fov_arr)

    xp_fov = l_los * np.tan(beta_rad) * cos_elongation
    yp_fov = l_los * np.tan(gamma_rad) * cos_elongation
    zp_fov = - l_los * cos_elongation

    sc_p_carr = np.dot(fov2carr_arr, np.vstack((xp_fov, yp_fov, zp_fov)))

    xp_carr = sc_p_carr[0] + x_sc_carr
    yp_carr = sc_p_carr[1] + y_sc_carr
    zp_carr = sc_p_carr[2] + z_sc_carr

    cos2lat = 1 - zp_carr ** 2 / (xp_carr ** 2 + yp_carr ** 2 + zp_carr ** 2)

    # Size & Spatial Distribution of Dust
    a = np.arange(1e-9, 1e-6, 1e-8)  # Dust size (m)
    rho = 2.5e6  # g/m^3
    m = (4. / 3) * np.pi * a ** 3 * rho  # g
    c1 = 2.2e3
    c2 = 15.
    c3 = 1.3e-9
    c4 = 1e11
    c5 = 1e27
    c6 = 1.3e-16
    c7 = 1e6
    g1 = 0.306
    g2 = -4.38
    g3 = 2.
    g4 = 4.
    g5 = -0.36
    g6 = 2.
    g7 = -0.85
    Fmr0 = (c1 * m ** g1 + c2) ** g2 \
           + c3 * (m + c4 * m ** g3 + c5 * m ** g4) ** g5 \
           + c6 * (m + c7 * m ** g6) ** g7  # m^-2s^-1
    v0 = 20e3  # m/s
    Nmr0 = 4. * Fmr0 / v0  # m^-3

    # VSF
    VSF0 = []
    for i in range(0, len(theta_rad)):
        VSF0.append(obtain_VSF0(a, Nmr0, theta_rad[i]))  # m^-1
    # plt.axes(yscale = 'log')
    # plt.plot(scatter_angle,VSF0)
    # plt.xlabel('Scatter Angle')
    # plt.ylabel('VSF(r0,theta)')
    # plt.show()
    # LOS Integral
    nu = 1.3
    delta_rad = abs(np.mean(np.diff(theta_rad)))
    I = I_0 * AU * (1 * np.sin(elongation_rad)) ** (-nu - 1) * \
        np.nansum(np.sin(theta_rad) ** nu * (0.15 + 0.85 * cos2lat ** 14) * np.array(VSF0) * delta_rad)
    return I


if __name__ == '__main__':
    # %%
    lon_rad = np.linspace(0, np.pi / 3, 60)
    lat_rad = np.linspace(0, np.pi / 3, 60)
    I_F = np.zeros((len(lon_rad), len(lat_rad)))
    for i in range(0, len(lon_rad)):  # unit: degree.
        for j in range(0, len(lat_rad)):
            I_tmp = get_Brightness_of_F_Corona([lon_rad[i], lat_rad[j]], [0, AU*np.cos(np.pi/3*2), AU*np.sin(np.pi/3*2)])
            I_F[i, j] = I_tmp

    [lonv, latv] = np.meshgrid(lon_rad, lat_rad)
    I_F = np.array(I_F)
    # %%
    plt.pcolormesh(np.rad2deg(lon_rad), np.rad2deg(lat_rad), np.log10(np.transpose(I_F)))
    plt.axis('scaled')
    plt.xlabel('beta_fov (Degree)')
    plt.ylabel('gamma_fov (Degree)')
    plt.title('log10(I_F)')
    plt.colorbar()
    plt.clim([-6,-3])
    plt.show()


    # I_F = np.zeros((len(lon),1))
    # for i in range(0, len(lon)):
    #     I_tmp = get_Brightness_of_F_Corona([lon[i],0])
    #     I_F[i] = I_tmp
    # print(I_F[19])
    # plt.plot(lon, I_F)
    # plt.yscale('log')
    # plt.xlabel('view_field_theta (Degree, in ecliptic plane)')
    # plt.ylabel('White Light Intensity of F corona (W m^-2 sr^-1)')
    # plt.show()
