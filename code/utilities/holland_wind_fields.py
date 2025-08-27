import numpy as np
from dataclasses import dataclass
from datetime import datetime

GEO_PARAM_UNIT = 78
KAPPA_UNIT = 42
DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi
TRACKING_TOLERANCE = 1e-10
omega = 2.0 * np.pi / 86164.2
coriolis_forcing = True
coordinate_system = 2
atmos_boundary_layer = 0.9
SAMPLING_TIME = 0.88
earth_radius = 6367500.0

def calc_polar_coordinate(x, y, sloc):
    r = spherical_distance(x, y, sloc[0], sloc[1])
    theta = np.arctan2(np.deg2rad(y-sloc[1]), np.deg2rad(x-sloc[0]))
    return r, theta

def spherical_distance(x, y, x2, y2):
    dx = np.deg2rad(x2 - x)
    dy = np.deg2rad(y2 - y)
    earth_radius = 6367500.0

    distance = earth_radius * 2.0 * np.arcsin(np.sqrt(np.sin(0.5*dy)**2
                                                    + np.cos(np.deg2rad(y))
                                                    *np.cos(np.deg2rad(y2))
                                                    * np.sin(0.5*dx)**2))
    # print(distance)
    return distance


def get_pressure_difference(Pc):
    ambient_pressure = 101300.

    dp = ambient_pressure - Pc
    if (dp < 100.0):
        dp = 100.0

    return dp


def calculate_holland_param(mx, my, xlower, ylower,
                            dx, dy, t, wind_index, pressure_index, storm):
    maux = 3
    mbc = 0
    # get interpolated storm data
    sloc, tv, mwr, mws, Pc, radius = get_storm_data(t, storm)
    storm = calc_storm_speed(storm)

    # remove translational component
    mod_mws, tv = adjust_max_wind(tv, mws, convert_height=True)

    dp = get_pressure_difference(Pc)
    B = get_holland_b(mod_mws, dp)

    aux = np.zeros(shape=(maux, int(mx+mbc), int(my+mbc)))
    # make aux array the size of whatever field I'm plotting
    for j in range(int(my+mbc)):
        y = ylower + (j-0.5)*dy # degrees latitude
        f = calc_coriolis(y, 2)
        for i in range(int(mx+mbc)):
            x = xlower + (i-0.5)*dx # degrees longitude

            r, theta = calc_polar_coordinate(x, y, sloc)

            # set pressure field
            aux[pressure_index, i, j] = Pc + dp * np.exp(-(mwr / r) **B)

            # speed of wind at this point
            wind = np.sqrt((mwr / r) ** B \
                   * np.exp(1.0 - (mwr / r)**B) \
            * mod_mws**2.0 + (r * f)**2.0 / 4.0) - r * f / 2.0


            aux = post_process_wind_estimate(maux, mbc, mx, my, i, j, wind, aux,
                                       wind_index, pressure_index, r, radius,
                                       tv, mod_mws, theta, convert_height=True)

    return aux

def get_holland_b(mod_mws, dp):
    rho_air = 1.15


    B = rho_air * np.exp(1.0) * (mod_mws**2) / dp
    if (B < 1.0):
        B = 1.0
    elif B > 2.5:
        B = 2.5

    return B


def get_storm_data(t, storm):
    i = storm_index(t, storm)
    last_storm_index = i
    if i == storm.num_casts + 1:
        i = i - 1
        # At last forecast, use last data for storm strength parameters
        # and velocity, location uses last velocity and constant motion forward

        # Convert coordinates temporarily to meters so that we can use
        # the pre-calculated m/s velocities from before.
        x = latlon2xy(storm.track[i,1:3], storm.track[i,1:3])
        x = x + (t - storm.track[i,0]) * storm.velocity[i,:]

        fn = [xy2latlon(x), storm.velocity[i,:], storm.max_wind_radius[i],
            storm.max_wind_speed[i], storm.central_pressure[i], storm.radius[i]]

    else:
        # In between two forecast time points (the function storm_index) ensures that we
        # are not before the first data point, i.e. i > 1
        tn = storm.track[i, 0]
        tnm = storm.track[i-1, 0]
        weight = (t - tnm) / (tn - tnm)
        fn = np.array([storm.track[i, 1:3], storm.velocity[i, :],
                      storm.max_wind_radius[i], storm.max_wind_speed[i],
                      storm.central_pressure[i], storm.radius[i]], dtype='object')
        fnm = np.array([storm.track[i - 1, 1:3], storm.velocity[i - 1, :],
                        storm.max_wind_radius[i - 1], storm.max_wind_speed[i - 1],
                        storm.central_pressure[i - 1], storm.radius[i - 1]], dtype='object')
        fn = weight * (fn - fnm) + fnm

    # set output variables
    location = fn[0]
    velocity = fn[1]
    max_wind_radius = fn[2]
    max_wind_speed = fn[3]
    central_pressure = fn[4]
    radius = fn[5]
    return location, velocity, max_wind_radius, max_wind_speed, central_pressure, radius


def adjust_max_wind(tv, mws, convert_height=True):
    atmos_boundary_layer = 0.9
    trans_speed = np.sqrt(tv[0]**2 + tv[1]**2)
    mod_mws = mws - trans_speed

    # Bound this at 0
    if mod_mws < 0:
        trans_mod = mws / trans_speed
        tv = tv * trans_mod
        mod_mws = 0

    if convert_height:
        mod_mws = mod_mws / atmos_boundary_layer

    return mod_mws, tv


def post_process_wind_estimate(maux, mbc, mx, my, i, j, wind, aux,
                               wind_index, pressure_index, r, radius,
                               tv, mod_mws, theta, convert_height):
    RAMP_WIDTH = 100.0
    if mod_mws > 0:
        trans_speed_x = (abs(wind) / mod_mws) * tv[0]
        trans_speed_y = (abs(wind) / mod_mws) * tv[1]
    else:
        trans_speed_x = 0
        trans_speed_y = 0

    if convert_height:
        wind = wind * atmos_boundary_layer
    wind *= SAMPLING_TIME
    # velocity components of storm (assumes perfect vortex shape)
    # including addition of translation speed
    # print(wind, np.cos(theta) , trans_speed_y)
    aux[wind_index, i, j] = -wind * np.sin(theta) + trans_speed_x
    aux[wind_index + 1, i, j] = wind * np.cos(theta) + trans_speed_y

    # Apply distance ramp down(up) to fields to limit scope
    Pa = 101300.0
    ramp = 0.5 * (1.0 - np.tanh((r-radius)/ RAMP_WIDTH))
    aux[pressure_index, i, j] = Pa + (aux[pressure_index,i, j] - Pa) * ramp
    aux[wind_index:wind_index+1,i,j] *= ramp

    return aux


def latlon2xy(coords, projection_center):

    x[0] = np.deg2rad * earth_radius * \
           (coords[0] - projection_center(0))\
           * np.cos(np.deg2rad * projection_center[1])
    x[1] = np.deg2rad * earth_radius * coords[1]

    return x


def xy2latlon(coords, projection_center):

    coords[0] = projection_center[0] + x[0]/\
                (np.deg2rad * earth_radius * np.cos(np.deg2rad * projection_center[1]))


    coords[1] = x[1] / (np.deg2rad * earth_radius)

    return coords


def storm_index(t, storm):
    time = list(storm.track[:,0])
    index = time.index(t)
    return index
    # if last_storm_index == storm.num_casts - 1:
    #     index = storm.num_casts + 1
    # else:
    #     t0 = storm.track[0, last_storm_index - 1]
    #     t1 = storm.track[0, last_storm_index]
    #     if (abs(t0-t1) < TRACKING_TOLERANCE) or \
    #         (abs(t1 - t) < TRACKING_TOLERANCE) or \
    #         (t0 < t) and (t < t1):
    #         index = last_storm_index
    # return index


def calc_coriolis(y, coordinate_system):
    if coordinate_system == 1:
        theta = y / 111.0 * DEG2RAD + theta_0
        coriolis = 2.0 * omega * (np.sin(theta_0) + (theta - theta_0) * np.cos(theta_0))


    elif coordinate_system == 2:
        coriolis = 2.0 * omega * np.sin(y * DEG2RAD)
    else:
        coriolis = 0.0

    return coriolis



@dataclass
class StormData:
    num_casts: int
    landfall: datetime
    # time: np.array
    track: np.array
    max_wind_speed: np.array
    max_wind_radius: np.array
    central_pressure: np.array
    radius: np.array
    velocity: np.array

def load_storm_data(storm_data_path) -> StormData:
    print(f"Reading storm data file {storm_data_path}")
    with open(storm_data_path) as f:
        lines = f.readlines()
        data = np.loadtxt(storm_data_path, skiprows=2)
        return StormData(
            num_casts=int(lines[0]),
            landfall=datetime.fromisoformat(lines[1][:-1]),
            track=data[:, 0:3],
            max_wind_speed=data[:, 3],
            max_wind_radius=data[:, 4],
            central_pressure=data[:, 5],
            radius=data[:, 6],
            velocity=np.empty(shape=(int(lines[0]), 2)))

    return storm


def sign(a, b):
    if a != 0:
        return min(a, a * -1) if b < 0 else max(a, a * -1)







def calc_storm_speed(storm):
    for i in range(storm.num_casts - 1):

        x = storm.track[i, 1:3]
        y = storm.track[i+1, 1:3]

        dt = storm.track[i+1, 0] - storm.track[i, 0]

        # if coordinate_system == 2:
        #     ds = spherical_distance(x[0], 0.5 * (x[1] + y[1]),
        #                             y[0], 0.5 * (x[1] + y[1]))
        #     if y[0] - x[0] != 0:
        #         storm.velocity[i, 0] = sign(ds / dt, y[0] - x[0])
        #     else:
        #         storm.velocity[i,0] = storm.velocity[i-1, 0]
        #
        #     ds = spherical_distance(0.5 * (x[0] + y[0]), x[1],
        #                             0.5 * (x[0] + y[0]), y[1])
        #     if y[1] - x[1] != 0:
        #         storm.velocity[i,1] = sign(ds / dt, y[1] - x[1])
        #     else:
        #         storm.velocity[i,1] = storm.velocity[i-1, 1]
        if coordinate_system == 2:
            ds = spherical_distance(x[0], 0.5 * (x[1] + y[1]),
                                    y[0], 0.5 * (x[1] + y[1]))

            storm.velocity[i,0] = sign(ds / dt, y[0] - x[0])

            ds = spherical_distance(0.5 * (x[0] + y[0]), x[1],
                                    0.5 * (x[0] + y[0]), y[1])
            storm.velocity[i,1] = sign(ds / dt, y[1] - x[1])
        else:
            storm.velocity[i,0] = abs((x[1] - x[0]) / dt)
            storm.velocity[i,1] = abs((y[1] - y[0]) / dt)
        storm.velocity[storm.num_casts - 1, :] = storm.velocity[storm.num_casts - 2, :]

    return storm

