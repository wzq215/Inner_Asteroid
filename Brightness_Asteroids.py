import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Asteroid_positions import get_aster_pos

'''
ATTENTION: 目前输出的是W/m**2, 小天体面积半径数据来自spacereference.org.
'''
# def apparent_magnitude_2_brightness(V):
#     F_ratio = 10**((-26.74-V)/2.5)
#     I = 1368 * F_ratio # 这里的1368是不是应该除掉太阳对地球所张的视角面积？
#     return I

def get_Brightness_of_Asteroid(name_str,start_time,stop_time):
    AU = 1.49e8
    df = pd.read_csv('data/sbdb_query_results.csv')
    aster_df = df[df['full_name']==name_str]
    # print(aster_df)
    spkid = aster_df.loc[:,'spkid'].values[0]
    Radii_aster = aster_df.loc[:,'diameter'].values[0]/AU/2 #AU
    # Radii_aster = 1.903/AU # for Atira
    # print(Radii_aster)
    # print(spkid)

    # Physical properties
    H = np.array(aster_df['H'])
    G = 0.24 # default

    # Geometry
    # !!!
    positions = get_aster_pos(spkid, start_time, stop_time)
    r = np.sqrt((positions[0]-1)**2+positions[1]**2+positions[2]**2)
    Delta = np.sqrt(positions[0]**2+positions[1]**2+positions[2]**2)

    # Delta = np.sqrt(1.+r**2-2*r*np.cos(theta)) # distance between Earth and Asteroid
    alpha = np.arccos((r**2+Delta**2-1.**2)/(2*r*Delta)) # angle between Sun-Aster and Aster-Earth

    # phase function, currently base on H-G system
    A1 = 3.33;A2 = 1.87;B1 = 0.63;B2 = 1.22
    phi1 = np.exp(-A1*(np.tan(alpha/2)**(B1)))
    phi2 = np.exp(-A2*(np.tan(alpha/2)**(B2)))
    V = H-2.5*np.log10((1.-G)*phi1+G*phi2)+5*np.log10(r*Delta)
    Ha = H-2.5*np.log10((1.-G)*phi1+G*phi2) # absolute Magnitude

    # magnitude to brightness
    F_ratio = 10**((-26.74-V)/2.5)
    deg2_aster = (np.pi*Radii_aster**2/Delta) # Unit: sr
    I_aster = 1368 * F_ratio # /deg2_aster

    lon_angle = -np.rad2deg(np.arcsin(positions[1]/Delta))#180-np.rad2deg(alpha)-np.rad2deg(theta)
    lat_angle = np.rad2deg(np.arcsin(positions[2]/Delta))
    return lon_angle,lat_angle,I_aster,deg2_aster

# [lon,lat, I_aster] = obtain_Apparent_Magnitude_of_Asteroids('163693 Atira (2003 CP20)')
# plt.scatter(lon,lat,c=I_aster)
# plt.show()