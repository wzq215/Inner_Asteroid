import itertools
import os
from functools import partial
from pathos.multiprocessing import ProcessingPool as Pool

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import spiceypy as spice
from scipy import special as sp

import furnsh_kernels

I_0 = 1361  # W/m^2 Solar Irradiance
AU_km = 1.496e8  # km, distance from sun to earth
Rs_km = 696300  # km, solar radii

Sun_sr = np.pi * Rs_km ** 2 / AU_km ** 2  # sr

F_0 = I_0 / Sun_sr


def inst2sun(obs_str: str, inst_str: str,
             obs_dt: datetime,
             to_frame='IAU_SUN'):
    """ Obtain transformation matrix from instrument to heliocentric frame.

    :param obs_str: str, 'SPP'
    :param inst_str: str, 'SPP_WISPR_INNER'
    :param obs_dt: datetime.datetime
    :param to_frame: str, 'IAU_SUN'
    :return:
    """
    obs_et = spice.datetime2et(obs_dt)
    obs_pos, _ = spice.spkpos(obs_str, obs_et, to_frame, 'NONE', 'SUN')  # km
    obs_pos = obs_pos.T
    inst2carr_arr = spice.sxform(inst_str, to_frame, obs_et)[0:3, 0:3]
    return obs_pos, inst2carr_arr


def fov2inst(obs_str, inst_str):
    """ Obtain transformation matrix from FOV to instrument. In WISPR, X_WISPR = LON_FOV, Y_WISPR = - LAT_FOV

    :param obs_str:
    :param inst_str:
    :return:
    """
    return np.eye(3)


def los_in_fov_z_base(beta_rad, gamma_rad,
                      z_rng=[0, 200], z_resolution=1.):
    z_fov = np.arange(z_rng[0], z_rng[1], z_resolution)
    x_fov = z_fov * np.tan(gamma_rad)
    y_fov = z_fov * np.tan(beta_rad)
    xyz_fov = np.vstack([x_fov, y_fov, z_fov])  # xyz[0] = x, etc.
    return xyz_fov


# TODO: TO ADD another los generating method, based on scattering angles.
# def los_in_fov_theta_base(beta_rad, gamma_rad,
#                           theta_rng=[0,np.pi],theta_resolution=np.deg2rad(1.)):
#


def sun_los(obs_str, inst_str, obs_dt,
            beta_rad, gamma_rad):
    obs_et = spice.datetime2et(obs_dt)
    sun_pos, _ = spice.spkpos('SUN', obs_et, inst_str, 'NONE', obs_str)  # sun pos in inst frame (km)
    los_vec_fov = np.array([np.tan(beta_rad), np.tan(gamma_rad), 1])

    fov2inst_arr = fov2inst(obs_str, inst_str)
    los_vec_inst = np.dot(fov2inst_arr, los_vec_fov)

    coselongation = np.dot(sun_pos / Rs_km, los_vec_inst) / np.linalg.norm(sun_pos / Rs_km) / np.linalg.norm(
        los_vec_inst)
    elongation_rad = np.arccos(coselongation)

    r_sun_inst_km = np.linalg.norm(sun_pos)
    return elongation_rad, r_sun_inst_km


def scattering_func_single_particle(los_theta_rad, los_sun_dist_au,
                                    los_cumulative_spatial_density,  # m^-3
                                    particle_radius_m, wavelength_m=550e-9,
                                    sigma_power_law=1.25):
    # CITE: Mann, A&A (1992)
    albedo = 0.25  # albedo

    particle_radius_m_arr, los_theta_rad_arr = np.meshgrid(particle_radius_m, los_theta_rad)  # 2D arr (N_size, N_theta)

    alpha_arr = 2 * np.pi * particle_radius_m_arr / wavelength_m

    sigma_r0_arr = (particle_radius_m_arr ** 2 * abs(sp.jv(1, alpha_arr * np.sin(los_theta_rad_arr))) ** 2
                    / abs(np.sin(los_theta_rad_arr)) ** 2 + albedo * particle_radius_m_arr ** 2 / 4)  # m^2
    sigma_arr = (sigma_r0_arr.T * los_sun_dist_au ** (-sigma_power_law)).T

    scattering_func = np.nansum(sigma_arr
                                * abs(np.diff(los_cumulative_spatial_density, axis=1, prepend=0.)), axis=1)  # m^-1
    return scattering_func


def cumulative_density_of_dust_1AU(particle_radius_m):
    # Spherical grains
    particle_density = 2.5e6  # g*m^-3
    mass = (4 / 3) * np.pi * particle_density * particle_radius_m ** 3  # g

    # Cite: Mann. et al. Dust Near The Sun. Space Science Reviews 110, 269–305 (2004).
    #       https://doi.org/10.1023/B:SPAC.0000023440.82735.ba
    c1 = 2.2e3
    c2 = 15
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

    # The cumulative flux of dust with masses > m (gram) at r0 = 1AU
    cumulative_flux_1AU = (c1 * mass ** g1 + c2) ** g2 \
                          + c3 * (mass + c4 * mass ** g3 + c5 * mass ** g4) ** g5 \
                          + c6 * (mass + c7 * mass ** g6) ** g7  # m^-2s^-1

    impact_velocity_1AU = 20.  # km/s
    cumulative_spatial_density_1AU = cumulative_flux_1AU / (1. / 4. * impact_velocity_1AU) * 1e-3  # m^-3

    return cumulative_spatial_density_1AU


def geometric_parameter_out_of_ecliptic(cos2lat_los, type='cosine'):
    # CITE: Lamy et al., SSR (2022)
    if type == 'cosine':
        k = 0.2
        nu_c = 44
        return k + (1 - k) * (cos2lat_los) ** (-nu_c / 2)
    elif type == 'ellipsoid':
        gamma_E = 4.5
        nu_E = -0.65
        return (1 + gamma_E ** 2 * (1 - cos2lat_los)) ** nu_E
    elif type == 'fan':
        gamma_F = 2.6
        nu_F = 1.
        return np.exp(-gamma_F * np.sqrt(1 - cos2lat_los) ** nu_F)
    else:
        return None


def geometric_parameter_radial(los_sun_dist, type='DFZ'):
    if type == None:
        return los_sun_dist * 0 + 1
    elif type == 'DFZ':
        r_in = 3.
        r_out = 19.
        los_sun_dist_Rs = los_sun_dist / Rs_km
        lambda_r = (los_sun_dist_Rs - r_in) / (r_out - r_in)
        lambda_r[lambda_r < 0] = lambda_r[lambda_r < 0] * 0.
        lambda_r[lambda_r > 1] = lambda_r[lambda_r > 1] * 0. + 1.
        return lambda_r
    else:
        return None


def F_corona_in_inst_fov(obs_str, inst_str, obs_dt,
                         fov_range, fov_resolution):
    '''

    :param obs_str:
    :param inst_str:
    :param obs_dt:
    :param fov_range: np.array([[gamma_min, gamma_max],[beta_min, beta_max]])
    :param fov_resolution:
    :return:
    '''

    # Transformation matrix from instrument frame to target frame (Heliocentric Carrington)
    obs_pos, inst2sun_arr = inst2sun(obs_str, inst_str, obs_dt)
    # Transformation matrix from FOV frame to instrument frame
    fov2inst_arr = fov2inst(obs_str, inst_str)

    # Sampling LOSs in FOV
    beta_rad_list = np.deg2rad(np.arange(fov_range[0][0], fov_range[0][1], fov_resolution))
    gamma_rad_list = np.deg2rad(np.arange(fov_range[1][0], fov_range[1][1], fov_resolution))

    los_angle_rad_list = itertools.product(beta_rad_list, gamma_rad_list)

    particle_radius_m = np.arange(1., 100., 1) * 1e-6  # m
    cumulative_spatial_density_1AU = cumulative_density_of_dust_1AU(
        particle_radius_m)  # m^-3, ndarray(len(particle_radius))

    I_matrix = []
    for los_angle_rad in los_angle_rad_list:
        beta_rad = los_angle_rad[0]
        gamma_rad = los_angle_rad[1]

        # Solving Observer-ScatteringPoint(P)-Sun Triangle
        elongation_rad, R_sun_inst_km = sun_los(obs_str, inst_str, obs_dt, beta_rad, gamma_rad)

        # Sampling along Line-of-sight
        los_xyz_fov = los_in_fov_z_base(beta_rad, gamma_rad)

        # Calculate properties along Los
        los_length = np.linalg.norm(los_xyz_fov, axis=0)  # Length of Obs-P
        los_dist_sun = np.sqrt(los_length ** 2 + R_sun_inst_km ** 2
                               - 2 * los_length * R_sun_inst_km * np.cos(
            elongation_rad))  # Heliocentric Distance Sun-P

        sin_los_scatter_angle_rad = R_sun_inst_km * np.sin(
            elongation_rad) / los_dist_sun  # Scatter Angle <Sun-P, P-Obs>
        los_scatter_angle_rad = np.arcsin(sin_los_scatter_angle_rad)
        los_scatter_angle_rad[los_length
                              > R_sun_inst_km * np.cos(elongation_rad)] \
            = np.pi - los_scatter_angle_rad[los_length
                                            > R_sun_inst_km * np.cos(
            elongation_rad)]  # Scatter Angle \in (elongation, pi)

        # Transform points on LoS from FOV to target frame
        los_xyz_inst = np.dot(fov2inst_arr, los_xyz_fov)
        los_xyz_sun = (np.dot(inst2sun_arr, los_xyz_inst).T + np.array(obs_pos)).T

        # Calculate heliocentric latitude along LoS
        # TODO: Assuming the Asymmetric Plane of Zodiacal Cloud to be Heliocentric latitude = 0. SHOULD BE CHANGEABLE.
        los_cos2lat_sun = 1 - los_xyz_sun[2] ** 2 / np.linalg.norm(los_xyz_sun,axis=0)  ** 2

        # cumulative flux to cumulative spatial density
        cumulative_spatial_density_1AU_arr, los_dist_sun_arr = np.meshgrid(cumulative_spatial_density_1AU, los_dist_sun)

        # cumulative_spatial_density = cumulative_spatial_density_1AU_arr / los_dist_sun_arr  # m^-3 N(particle_radius, r)
        sigma_power_law = 1.25
        scattering_func = scattering_func_single_particle(los_scatter_angle_rad, los_dist_sun / AU_km,
                                                          cumulative_spatial_density_1AU_arr,
                                                          particle_radius_m,
                                                          sigma_power_law=sigma_power_law)  # m^-1

        los_delta_scatter_angle_rad = np.diff(los_scatter_angle_rad, prepend=elongation_rad)
        # FIXME: SOMETHING IS WRONG. EXPECTED TO ELONGATE ALONG ECLIPTIC.
        para_out_of_ecliptic = geometric_parameter_out_of_ecliptic(los_cos2lat_sun,type='ellipsoid')
        para_radial = geometric_parameter_radial(los_dist_sun,type='DFZ')
        I = F_0 * AU_km * 1e3 / (  # Wm^-2 * m
                R_sun_inst_km / AU_km * np.sin(elongation_rad)) ** (sigma_power_law + 1.) \
            * np.nansum(scattering_func  # m^-1
                        * para_out_of_ecliptic
                        # * para_radial
                        * np.sin(los_scatter_angle_rad) ** sigma_power_law
                        * los_delta_scatter_angle_rad)

        I_matrix.append(I)  # Wm^-2
    I_matrix = np.array(I_matrix).reshape(len(beta_rad_list), len(gamma_rad_list)).T
    return I_matrix


if __name__ == '__main__':
    # %%
    I_matrix = F_corona_in_inst_fov('SPP', 'SPP_WISPR_INNER', datetime(2018, 11, 6), [[-20, 20], [-20, 20]], 1.)
    # plt.pcolormesh(I_matrix)
    plt.contourf(np.log10(I_matrix / F_0))
    plt.colorbar()
    plt.show()
