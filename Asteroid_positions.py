import numpy as np
import matplotlib.pyplot as plt
from horizons_spk import get_spk_from_horizons
from mpl_toolkits.mplot3d import Axes3D
import spiceypy as spice
from datetime import datetime
import pandas as pd


def get_aster_pos(spkid, start_time, stop_time, observer ='Earth BARYCENTER', frame ='GSE'):
    # Print out the toolkit version
    spice.tkvrsn("TOOLKIT")
    # Needed for leap seconds
    spice.furnsh('kernels/naif0012.tls')
    # Needed for Earth
    spice.furnsh('kernels/earth_200101_990628_predict.bpc')
    spice.furnsh('kernels/GSE.tf')
    spice.furnsh('kernels/HEEQ.tf')
    spice.furnsh('kernels/HCI.tf')
    spice.furnsh('kernels/de430.bsp')
    spice.furnsh('kernels/pck00010.tpc')
    # Needed for Asteroids
    spkfileid = 1000320
    # spkfileid = get_spk_from_horizons(spkid, start_time, stop_time)
    spice.furnsh('kernels/' + str(spkfileid) + '(' + start_time + '_' + stop_time + ')' + '.bsp')

    # UTC2ET
    start_dt = datetime.strptime(start_time, '%Y-%m-%d')
    stop_dt = datetime.strptime(stop_time, '%Y-%m-%d')
    utc = [start_dt.strftime('%b %d, %Y'), stop_dt.strftime('%b %d, %Y')]
    etOne = spice.str2et(utc[0])
    etTwo = spice.str2et(utc[1])

    # Epochs
    step = 365
    times = [x * (etTwo - etOne) / step + etOne for x in range(step)]

    # Get positions
    positions, LightTimes = spice.spkpos(str(spkfileid), times, frame, 'NONE', observer)

    # Clear
    spice.kclear()

    AU = 1.49e8  # distance from sun to earth
    positions = positions.T  # positions is shaped (4000, 3), let's transpose to (3, 4000) for easier indexing
    positions = positions / AU
    # fig = plt.figure(figsize=(9, 9))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(positions[0], positions[1], positions[2])
    # ax.scatter(1, 0, 0, c='red')
    # ax.set_xlabel('X (AU)')
    # ax.set_ylabel('Y (AU)')
    # ax.set_zlabel('Z (AU)')
    # plt.title('SPKID=' + str(spkid) + '(' + start_time + '_' + stop_time + ')')
    # plt.show()
    return positions

def get_body_pos(body, start_time, stop_time, observer ='Earth BARYCENTER', frame ='GSE'):
    # Print out the toolkit version
    spice.tkvrsn("TOOLKIT")
    # Needed for leap seconds
    spice.furnsh('kernels/naif0012.tls')
    # Needed for Earth
    spice.furnsh('kernels/earth_200101_990628_predict.bpc')
    spice.furnsh('kernels/GSE.tf')
    spice.furnsh('kernels/HEEQ.tf')
    spice.furnsh('kernels/HCI.tf')
    spice.furnsh('kernels/de430.bsp')
    spice.furnsh('kernels/pck00010.tpc')

    # UTC2ET
    start_dt = datetime.strptime(start_time, '%Y-%m-%d')
    stop_dt = datetime.strptime(stop_time, '%Y-%m-%d')
    utc = [start_dt.strftime('%b %d, %Y'), stop_dt.strftime('%b %d, %Y')]
    etOne = spice.str2et(utc[0])
    etTwo = spice.str2et(utc[1])

    # Epochs
    step = 365
    times = [x * (etTwo - etOne) / step + etOne for x in range(step)]

    # Get positions
    positions, LightTimes = spice.spkpos(body, times, frame, 'NONE', observer)

    # Clear
    spice.kclear()

    AU = 1.49e8  # distance from sun to earth
    positions = positions.T  # positions is shaped (4000, 3), let's transpose to (3, 4000) for easier indexing
    positions = positions / AU
    return positions
