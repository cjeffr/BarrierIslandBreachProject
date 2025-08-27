import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

# Linear interpolation in x direction
def find_near(ds, near_lon, near_lat):
    abslat = np.abs(ds.lat - near_lat)
    abslon = np.abs(ds.lon - near_lon)
    c = np.maximum(abslon, abslat)
    x, y = np.where(c == np.min(c))
    pts = ds.isel(lon=x[0], lat=y[0])
    return pts.lon.values, pts.lat.values


def get_interp_points(ds, lons, lats, llon, llat, ulon, ulat):
    points = [(llon, llat, ds.where((ds.lon == llon) &
                                    (ds.lat == llat), drop=True).values),
              (llon, ulat, ds.where((ds.lon == llon) &
                                    (ds.lat == ulat), drop=True).values),
              (ulon, llat, ds.where((ds.lon == ulon) &
                                    (ds.lat == llat), drop=True).values),
              (ulon, ulat, ds.where((ds.lon == ulon) &
                                    (ds.lat == ulat), drop=True).values)]
    return points



def bilinear_interpolation(ds, x, y, points):
    """
     Interpolate (x,y) from values associated with 4 points.
     the four points are a list of four triplets (x, y, values)
     the four points can be in any order, they should form a rectangle
     """

    points = sorted(points)  # order points by x then by y
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise ValueError('points do not form a rectangle')
    if not x1 <= x <= x2 or not y1 <= y <= y2:
        raise ValueError('(x,y) not within rectangle')

    return (q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)) / ((x2 - x1) * (y2 - y1) + 0.00)

def fill_patch_data(ds_large, ds_patch, dx, dy):
    xlower = ds_patch.lon[0].values + 0.5
    ylower = ds_patch.lat[0].values + 0.5
    xupper = ds_patch.lon[-1].values - 0.5
    yupper = ds_patch.lat[-1].values - 0.5
    mx = int(np.floor((xupper - xlower)/dx))
    my = int(np.floor((yupper - ylower)/dy))
    lons = [xlower + (i + 0.5) * dx for i in range(mx)]
    lats = [ylower + (i + 0.5) * dy for i in range(my)]
    new_speed = np.empty(shape=(len(lons), len(lats)))
    for iidx, i in enumerate(lons):
        for jidx, j in enumerate(lats):
            llon, llat = find_near(ds_large, i-0.25, j-0.25)
            ulon, ulat = find_near(ds_large, i+0.25, j+0.25)
            points = get_interp_points(ds_large, lons, lats, llon, llat, ulon, ulat)
            new_speed[iidx,jidx] = bilinear_interpolation(ds_large, i, j, points)
    return (lons, lats, new_speed)


# load entire storm data file
ds = xr.open_dataset('1938_04_WNATkm.nc')

# Take a single point in time for now
dst = ds.isel(time=0)

### subset lat and lon then slice the ds by lat/lon values/indices
lat = dst.lat
lon = dst.lon
lat1, lat2 = np.array_split(lat, 2)
lon1, lon2 = np.array_split(lon, 2)

t1 = dst.speed.where((dst.lat==lat1) & (dst.lon==lon1))
t2 = dst.speed.where((dst.lat==lat1) & (dst.lon == lon2))
t3 = dst.speed.where((dst.lat==lat2) & (dst.lon == lon1))
t4 = dst.speed.where((dst.lat == lat2) & (dst.lon==lon2))

fig, axs = plt.subplots(nrows=2, ncols=2)
t3.plot(ax=axs[0,0], vmin=0, vmax=45)
t4.plot(ax=axs[0,1], vmin=0, vmax=45)
t1.plot(ax=axs[1,0], vmin=0, vmax=45)
t2.plot(ax=axs[1,1], vmin=0, vmax=45)

lst = [t1, t2, t3, t4]
patches = []
for l in lst:
    patches.append(fill_patch_data(dst.speed, l, 0.1, 0.1))

fig, axes = plt.subplots(ncols=2, nrows=2)
xxx = []
yyy = []
for i, p in enumerate(patches):
    X, Y = np.meshgrid(p[0], p[1])
    xxx.append(X)
    yyy.append(Y)

im = axes[1, 0].pcolormesh(xxx[0], yyy[0], patches[0][2].T)
axes[0, 0].set_title('[0,0]')
axes[1, 1].pcolormesh(xxx[1], yyy[1], patches[1][2].T)
axes[0, 1].set_title('[0,1]')
axes[0, 0].pcolormesh(xxx[2], yyy[2], patches[2][2].T)
axes[1, 0].set_title('[1,0]')
axes[0, 1].pcolormesh(xxx[3], yyy[3], patches[3][2].T)
axes[1, 1].set_title('[1,1]')
plt.colorbar(im)
plt.show()



