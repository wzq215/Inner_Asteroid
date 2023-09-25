import itertools
import os
from functools import partial
from pathos.multiprocessing import ProcessingPool as Pool

import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import spiceypy as spice
from scipy import special as sp

import sunpy.coordinates.sun as sun
from sunpy.coordinates import get_body_heliographic_stonyhurst,HeliocentricInertial,HeliographicCarrington
from astropy.coordinates import SkyCoord

import furnsh_kernels

I_0 = 1361  # W/m^2 Solar Irradiance
AU_km = 1.496e8  # km, distance from sun to earth
Rs_km = 696300  # km, solar radii

Sun_sr = np.pi * Rs_km ** 2 / AU_km ** 2  # sr

F_0 = I_0 / Sun_sr

def polar_orbit(epoch,coord='HG'):

    T_orbit = timedelta(days=360)
    T_sun =(sun.carrington_rotation_time(2001)-sun.carrington_rotation_time(2000)).value
    T_sun = timedelta(days=T_sun)

    R = AU_km/Rs_km
    # t0 = epoch[0]
    t0 = datetime(2022,1,1)

    earth = get_body_heliographic_stonyhurst('earth', t0, include_velocity=False)
    earth_hci = SkyCoord(earth).transform_to(HeliocentricInertial())
    earth_carrington = SkyCoord(earth).transform_to(HeliographicCarrington(observer='earth'))

    earth_lon_hci = earth_hci.lon.to('rad').value
    earth_lon_carrington =earth_carrington.lon.to('rad').value

    t_orbit = np.array((epoch-t0)/T_orbit,dtype=float)
    t_sun = np.array((epoch-t0)/T_sun,dtype=float)

    if coord == 'HCI':
        r = R+np.zeros_like(t_orbit)
        lon = earth_lon_hci
        lat = (2*np.pi*t_orbit)

        x = r * np.cos(lat) * np.cos(lon)
        y = r * np.cos(lat) * np.sin(lon)
        z = r * np.sin(lat)

        lon = lon % (2 * np.pi)
        lat = np.arcsin(np.sin(lat))

    elif coord == 'HG':
        r = R+np.zeros_like(t_orbit)
        lon = (earth_lon_carrington - 2*np.pi*t_sun)
        lat = (2*np.pi*t_orbit)
        print('2*np.pi*t_orbit',lat)
        x = r * np.cos(lat) * np.cos(lon)
        y = r * np.cos(lat) * np.sin(lon)
        z = r * np.sin(lat)
        lon = lon % (2 * np.pi)
        lat = np.arcsin(np.sin(lat))
        print('np.arcsin(np.sin(lat))',lat)
    else:
        return None

    return {'r_Rs':r, 'lon':lon, 'lat':lat, 'x_Rs':x, 'y_Rs':y, 'z_Rs':z,'epoch':epoch, 'coord':coord}


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
    if obs_str == 'SPO':
        if to_frame == 'IAU_SUN':
            spo_pos = polar_orbit(obs_dt,coord='HG')
            spo_x_km = spo_pos['x_Rs'] * Rs_km
            spo_y_km = spo_pos['y_Rs'] * Rs_km
            spo_z_km = spo_pos['z_Rs'] * Rs_km
            spo_r_km = spo_pos['r_Rs'] * Rs_km
            obs_pos = np.array([spo_x_km,spo_y_km,spo_z_km])
            sun2inst_arr = np.array([[-spo_x_km * spo_z_km / spo_r_km ** 2,
                                      -spo_y_km * spo_z_km / spo_r_km ** 2,
                                      1 - spo_z_km ** 2 / spo_r_km ** 2],
                                     [spo_y_km / spo_r_km, -spo_x_km / spo_r_km, 0],
                                     [spo_x_km / spo_r_km, spo_y_km / spo_r_km, spo_z_km / spo_r_km]])
            if spo_x_km == 0 and spo_y_km == 0:
                inst2carr_arr = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            else:
                inst2carr_arr = np.linalg.inv(sun2inst_arr)
        else:
            return None

    elif obs_str == 'SPP':
        obs_et = spice.datetime2et(obs_dt)
        obs_pos, _ = spice.spkpos(obs_str, obs_et, to_frame, 'NONE', 'SUN')  # km
        obs_pos = obs_pos.T
        inst2carr_arr = spice.sxform(inst_str, to_frame, obs_et)[0:3, 0:3]
    else:
        return None

    return obs_pos, inst2carr_arr


def fov2inst(obs_str, inst_str):
    """ Obtain transformation matrix from FOV to instrument. In WISPR, X_WISPR = LON_FOV, Y_WISPR = - LAT_FOV

    :param obs_str:
    :param inst_str:
    :return:
    """
    if obs_str == 'SPP':
        # SPP_WISPR_INNER/OUTER (in HG):
        # SUN  (+X) <--- x (+Z)
        #                |
        #                v (+Y)
        # FOV (in HG):
        #           ^ +Y
        #           |
        # SUN       . ---> +X
        #           +Z
        fov2inst_arr = np.array([[-1,0,0],[0,-1,0],[0,0,-1]])
    elif obs_str == 'SPO':
        # SPO_INST (in HG):
        # e_z = e_R
        # e_y = e_R x e_Z
        # e_x = e_y x e_z
        fov2inst_arr = np.array([[0,1,0],[-1,0,0],[0,0,1]])
    else:
        return None
    return fov2inst_arr


def los_in_fov_z_base(beta_rad, gamma_rad,
                      z_rng=[0, 100], z_resolution=0.5):
    print(z_rng,z_resolution)
    z_fov = np.arange(z_rng[0], z_rng[1], z_resolution)*Rs_km
    x_fov = z_fov * np.tan(beta_rad)
    y_fov = z_fov * np.tan(gamma_rad)
    xyz_fov = np.vstack([x_fov, y_fov, -z_fov])  # xyz[0] = x, etc.
    # Use -z_fov to maintain right-hand coord.
    # ^ +Y
    # |
    # . ---> +X
    # +Z
    return xyz_fov


# TODO: TO ADD another los generating method, based on scattering angles.
# def los_in_fov_theta_base(beta_rad, gamma_rad,
#                           theta_rng=[0,np.pi],theta_resolution=np.deg2rad(1.)):
#


def sun_los(obs_str, inst_str, obs_dt,
            beta_rad, gamma_rad):
    if obs_str == 'SPP':
        obs_et = spice.datetime2et(obs_dt)
        sun_pos, _ = spice.spkpos('SUN', obs_et, inst_str, 'NONE', obs_str)  # sun pos in inst frame (km)
    elif obs_str == 'SPO':
        spo_pos_sun, inst2sun_arr = inst2sun(obs_str,inst_str,obs_dt,to_frame='IAU_SUN')
        sun2inst_arr = np.linalg.inv(inst2sun_arr)
        sun_pos = np.dot(sun2inst_arr,-spo_pos_sun)
    else:
        return None
    los_vec_fov = np.array([np.tan(beta_rad), np.tan(gamma_rad), 1])

    fov2inst_arr = fov2inst(obs_str, inst_str)
    los_vec_inst = np.dot(fov2inst_arr, los_vec_fov)

    coselongation = np.dot(sun_pos / Rs_km, los_vec_inst) / np.linalg.norm(sun_pos / Rs_km) / np.linalg.norm(
        los_vec_inst)
    elongation_rad = np.arccos(coselongation)

    r_sun_inst_km = np.linalg.norm(sun_pos)

    return elongation_rad, r_sun_inst_km


def scattering_func_single_particle(los_theta_rad, los_sun_dist_au,
                                    los_cumulative_spatial_density_1AU_arr,  # m^-3
                                    particle_radius_m, wavelength_m=550e-9,
                                    sigma_power_law=1.25,
                                    ):
    # CITE: Mann, A&A (1992)
    albedo = 0.25  # albedo
    particle_radius_m_arr, los_theta_rad_arr = np.meshgrid(particle_radius_m, los_theta_rad)  # 2D arr (N_size, N_theta)

    alpha_arr = 2 * np.pi * particle_radius_m_arr / wavelength_m

    sigma_r0_arr = (particle_radius_m_arr ** 2 * abs(sp.jv(1, alpha_arr * np.sin(los_theta_rad_arr))) ** 2
                    / abs(np.sin(los_theta_rad_arr)) ** 2 + albedo * particle_radius_m_arr ** 2 / 4)  # m^2
    sigma_arr = (sigma_r0_arr.T * los_sun_dist_au ** (-sigma_power_law)).T

    scattering_func = np.nansum(sigma_arr
                                * abs(np.diff(los_cumulative_spatial_density_1AU_arr, axis=1, prepend=0.)), axis=1)  # m^-1
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


def geometric_parameter_out_of_ecliptic_(cos2lat_los, type='cosine'):
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

def geometric_parameter_out_of_ecliptic(sinlat_los, type='cosine'):
    # CITE: Lamy et al., SSR (2022)
    if type == 'cosine':
        k = 0.2
        nu_c = 44
        return k + (1 - k) * np.sqrt(1-sinlat_los**2) ** (-nu_c)
    elif type == 'ellipsoid':
        gamma_E = 4.5
        nu_E = -0.65
        return (1 + (gamma_E*sinlat_los)**2) ** nu_E
    elif type == 'fan':
        gamma_F = 2.6
        nu_F = 1.
        return np.exp(-gamma_F * abs(sinlat_los) ** nu_F)
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
                         fov_range, fov_resolution,
                         z_rng_Rs=[0,200],z_resolution_Rs=1.,
                         check_los=False,
                         model_out_of_ecliptic='fan',
                         model_radial=None):
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

    if check_los:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(obs_pos[0] / Rs_km, obs_pos[1] / Rs_km, obs_pos[2] / Rs_km)
        ax.scatter(0, 0, 0, color='red')

    I_matrix = []
    for los_angle_rad in los_angle_rad_list:
        beta_rad = los_angle_rad[0]
        gamma_rad = los_angle_rad[1]

        # Solving Observer-ScatteringPoint(P)-Sun Triangle
        elongation_rad, R_sun_inst_km = sun_los(obs_str, inst_str, obs_dt, beta_rad, gamma_rad)

        # Sampling along Line-of-sight
        los_xyz_fov = los_in_fov_z_base(beta_rad, gamma_rad,
                                        z_rng=z_rng_Rs,z_resolution=z_resolution_Rs)  # km

        # Calculate properties along Los
        los_length = np.linalg.norm(los_xyz_fov, axis=0)  # Length of Obs-P [km]
        los_dist_sun = np.sqrt(los_length ** 2 + R_sun_inst_km ** 2
                               - 2 * los_length * R_sun_inst_km * np.cos(
            elongation_rad))  # Heliocentric Distance Sun-P [km]

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
        # los_cos2lat_sun = los_xyz_sun[2] ** 2 / np.linalg.norm(los_xyz_sun,axis=0) ** 2
        los_sinlat_sun = los_xyz_sun[2] / np.linalg.norm(los_xyz_sun,axis=0)

        # cumulative flux to cumulative spatial density
        cumulative_spatial_density_1AU_arr, los_dist_sun_arr = np.meshgrid(cumulative_spatial_density_1AU, los_dist_sun)

        # cumulative_spatial_density = cumulative_spatial_density_1AU_arr / los_dist_sun_arr  # m^-3 N(particle_radius, r)
        sigma_power_law = 1.25
        scattering_func = scattering_func_single_particle(los_scatter_angle_rad, los_dist_sun / AU_km,
                                                          cumulative_spatial_density_1AU_arr,
                                                          particle_radius_m,
                                                          sigma_power_law=sigma_power_law)  # m^-1

        los_delta_scatter_angle_rad = np.diff(los_scatter_angle_rad, prepend=elongation_rad)

        if model_out_of_ecliptic:
            para_out_of_ecliptic = geometric_parameter_out_of_ecliptic(los_sinlat_sun,type=model_out_of_ecliptic)
            scattering_func = scattering_func * para_out_of_ecliptic
        if model_radial:
            para_radial = geometric_parameter_radial(los_dist_sun,type=model_radial)
            scattering_func = scattering_func * para_radial

        # FIXME: Wrong Magnitude?
        I = F_0 * AU_km * 1e3 / (  # Wm^-2 * m
                R_sun_inst_km / AU_km * np.sin(elongation_rad)) ** (sigma_power_law + 1.) \
            * np.nansum(scattering_func  # m^-1
                        * np.sin(los_scatter_angle_rad) ** sigma_power_law
                        * los_delta_scatter_angle_rad)

        if check_los:
            # plot geometry
            scatter_ = ax.scatter(los_xyz_sun[0] / Rs_km, los_xyz_sun[1] / Rs_km, los_xyz_sun[2] / Rs_km,
                                  c=np.log10(scattering_func  # m^-1
                                    * np.sin(los_scatter_angle_rad) ** sigma_power_law
                                    * los_delta_scatter_angle_rad))
            # plt.colorbar(scatter_,)
            scatter_.set_clim(-22, -20)

        I_matrix.append(I)  # Wm^-2
    if check_los:
        plt.show()
    I_matrix = np.array(I_matrix).reshape(len(beta_rad_list), len(gamma_rad_list)).T
    return I_matrix


if __name__ == '__main__':
    # %%
    fov_x_rng_deg = [-20,20]
    fov_y_rng_deg = [-20,20]
    fov_resolution = 1.
    fov_x = np.arange(fov_x_rng_deg[0],fov_x_rng_deg[1],fov_resolution)
    fov_y = np.arange(fov_y_rng_deg[0],fov_y_rng_deg[1],fov_resolution)
    FOV_X, FOV_Y = np.meshgrid(fov_x,fov_y)

    I_matrix = F_corona_in_inst_fov('SPO', '', datetime(2018, 11, 6),
                                    [fov_x_rng_deg, fov_y_rng_deg], fov_resolution,
                                    z_rng_Rs=[0.,400.],z_resolution_Rs=1.,
                                    model_out_of_ecliptic='fan',
                                    model_radial='DFZ',check_los=False,
                                    )

    # %%
    plt.figure()
    plt.pcolormesh(FOV_X,FOV_Y,np.log10(I_matrix / F_0),cmap='jet')
    plt.colorbar()
    plt.xlabel('FOV_X (deg)')
    plt.ylabel('FOV_Y (deg)')
    plt.title('MSB (B_F/B_sun_1au)')
    # plt.clim([-6.5,-5.5])
    # plt.clim([-10.,-6.])
    plt.show()
