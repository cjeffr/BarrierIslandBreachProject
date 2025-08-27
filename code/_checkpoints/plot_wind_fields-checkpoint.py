import holland_wind_fields as hwf
import numpy as np
from map_making.fortq_tools import read_fortq
import cartopy
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


# storm = hwf.load_storm_data('/Users/catherinej/projects/panda-solution/480.storm')
storm = hwf.load_storm_data('/Users/catherinej/clawpack_src/clawpack-v5.7.1/'
                            'geoclaw/examples/storm-surge/isaac_test/isaac.storm')

storm = hwf.calc_storm_speed(storm)

xlower = -99.0
xupper = -70.
ylower = 8.00
yupper = 32.0
mx = (xupper - xlower) * 4
my = (yupper - ylower) * 4
dx = 0.25
dy = 0.25
aux = np.empty(shape=(2, int(mx), int(my)))
wind_index = 0
pressure_index = 2
pdd = []
for i in range(7):
    pd = read_fortq.read_fortq(i, '/Users/catherinej/clawpack_src/clawpack-v5.7.1/geoclaw/'
                               'examples/storm-surge/isaac_test/_output', 'ascii')
    pdd.append(pd)

auxs = []
for i in range(0, storm.num_casts, 1):
    t = storm.track[i, 0]
    print(t)
#     for pd in pdd:
#         dsd, ts = read_fortq.organize_patches(pd)
#         if ts == t/3600:
#             fig, axs = plt.subplots(1,2, subplot_kw=dict(projection=ccrs.Mercator()))
#             for amr in dsd:
#                 dsl = dsd[amr]
# #                 # v = np.linspace(-.5, 3.0, 15, endpoint=True)
#                 for idx, p in enumerate(dsl):
#                     sp = np.sqrt(p.v**2 + p.u**2)
#                     sp.plot.pcolormesh(ax=axs[1], transform=ccrs.PlateCarree(),vmin=0, vmax=30,add_colorbar=False)
    aux = hwf.calculate_holland_param(mx, my, xlower, ylower, dx, dy, t,
                            wind_index, pressure_index, storm)
    auxs.append(aux)
            # # axs[1].coastlines()
            # x = np.linspace(xlower, xupper, int(mx+2))
            # y = np.linspace(ylower, yupper, int(my+2))
            # X, Y = np.meshgrid(x, y)
            # s = np.sqrt(aux[0]**2 + aux[1]**2).T
            # plt.show()
            # im = axs[0].pcolor(X, Y, s, transform=ccrs.PlateCarree())
#             fig.colorbar(im)
#             plt.title(f'{t/3600} hours from landfall')
#             axs[0].coastlines()
#
#     else:
#         aux = hwf.calculate_holland_param(mx, my, xlower, ylower, dx, dy, t,
#                                       wind_index, pressure_index, storm)
#         x = np.linspace(xlower, xupper, int(mx + 2))
#         y = np.linspace(ylower, yupper, int(my + 2))
#         X, Y = np.meshgrid(x, y)
#         # print(X.shape, Y.shape)
#         fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.Mercator()))
#         s = np.sqrt(aux[0] ** 2 + aux[1] ** 2).T
#         im = ax.pcolor(X, Y, s, transform=ccrs.PlateCarree())
#         fig.colorbar(im)
#         plt.title(f'{t / 3600} hours from landfall')
#         ax.coastlines()
#     plt.show()
#
#
