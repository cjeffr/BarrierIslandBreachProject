
from __future__ import absolute_import
from __future__ import print_function

import os

import numpy
import matplotlib.pyplot as plt
import datetime

import clawpack.visclaw.colormaps as colormap
import clawpack.visclaw.gaugetools as gaugetools
import clawpack.clawutil.data as clawutil
import clawpack.amrclaw.data as amrclaw
import clawpack.geoclaw.data as geodata
from clawpack.visclaw import particle_tools
from clawpack.visclaw import legend_tools
from clawpack.visclaw import geoplot


import clawpack.geoclaw.surge.plot as surgeplot

try:
    from setplotfg import setplotfg
except:
    setplotfg = None


def setplot(plotdata=None):
    """"""

    if plotdata is None:
        from clawpack.visclaw.data import ClawPlotData
        plotdata = ClawPlotData()

    # clear any old figures,axes,items data
    plotdata.clearfigures()
    plotdata.format = 'ascii'

    # Load data from output
    clawdata = clawutil.ClawInputData(2)
    clawdata.read(os.path.join(plotdata.outdir, 'claw.data'))
    physics = geodata.GeoClawData()
    physics.read(os.path.join(plotdata.outdir, 'geoclaw.data'))
    surge_data = geodata.SurgeData()
    surge_data.read(os.path.join(plotdata.outdir, 'surge.data'))
    friction_data = geodata.FrictionData()
    friction_data.read(os.path.join(plotdata.outdir, 'friction.data'))

    # Load storm track
    track = surgeplot.track_data(os.path.join(plotdata.outdir, 'fort.track'))

    # Set afteraxes function
    def surge_afteraxes(cd):
        surgeplot.surge_afteraxes(cd, track, plot_direction=False,
                                             kwargs={"markersize": 4})

    # Color limits
    surface_limits = [-3.0, 3.0]
    speed_limits = [0.0, 2.0]
    wind_limits = [0, 64]
    pressure_limits = [935,1013]
    friction_bounds = [0.01, 0.04]

    def friction_after_axes(cd):
        plt.title(r"Manning's $n$ Coefficient")

    # ==========================================================================
    #   Plot specifications
    # ==========================================================================
    regions = {"Full Domain": {"xlimits": (clawdata.lower[0], clawdata.upper[0]),
                               "ylimits": (clawdata.lower[1], clawdata.upper[1]),
                               "figsize": (6.4, 4.8)},
               "Moriches Bay": {"xlimits": (-72.885652, -72.634247),
                               "ylimits": (40.718299, 40.828344),
                               "figsize": (6.4, 4.8)},
               "Entire Island System": {"xlimits": (-73.75,-72.25),
                               "ylimits": (40.50,41.0),
                               "figsize": (6.4, 4.8)}}


    for (name, region_dict) in regions.items():

        # Surface Figure
        plotfigure = plotdata.new_plotfigure(name="Surface - %s" % name)
        plotfigure.kwargs = {"figsize": region_dict['figsize']}
        plotaxes = plotfigure.new_plotaxes()
        plotaxes.title = "Surface"
        plotaxes.xlimits = region_dict["xlimits"]
        plotaxes.ylimits = region_dict["ylimits"]
        plotaxes.afteraxes = surge_afteraxes

        surgeplot.add_surface_elevation(plotaxes, bounds=surface_limits)
        surgeplot.add_land(plotaxes)
        plotaxes.plotitem_dict['surface'].amr_patchedges_show = [0] * 10
        plotaxes.plotitem_dict['land'].amr_patchedges_show = [0] * 10

        # Speed Figure
        plotfigure = plotdata.new_plotfigure(name="Currents - %s" % name)
        plotfigure.kwargs = {"figsize": region_dict['figsize']}
        plotaxes = plotfigure.new_plotaxes()
        plotaxes.title = "Currents"
        plotaxes.xlimits = region_dict["xlimits"]
        plotaxes.ylimits = region_dict["ylimits"]
        plotaxes.afteraxes = surge_afteraxes

        surgeplot.add_speed(plotaxes, bounds=speed_limits)
        surgeplot.add_land(plotaxes)
        plotaxes.plotitem_dict['speed'].amr_patchedges_show = [0] * 10
        plotaxes.plotitem_dict['land'].amr_patchedges_show = [0] * 10
    #
    # Friction field
    #
    plotfigure = plotdata.new_plotfigure(name='Friction')
    plotfigure.show = friction_data.variable_friction and True

    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = regions['Full Domain']['xlimits']
    plotaxes.ylimits = regions['Full Domain']['ylimits']
    # plotaxes.title = "Manning's N Coefficient"
    plotaxes.afteraxes = friction_after_axes
    plotaxes.scaled = True

    surgeplot.add_friction(plotaxes, bounds=friction_bounds, shrink=0.9)
    plotaxes.plotitem_dict['friction'].amr_patchedges_show = [0] * 10
    plotaxes.plotitem_dict['friction'].colorbar_label = "$n$"

    #
    #  Hurricane Forcing fields
    #
    # Pressure field
    plotfigure = plotdata.new_plotfigure(name='Pressure')
    plotfigure.show = surge_data.pressure_forcing and True

    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = regions['Full Domain']['xlimits']
    plotaxes.ylimits = regions['Full Domain']['ylimits']
    plotaxes.title = "Pressure Field"
    plotaxes.afteraxes = surge_afteraxes
    plotaxes.scaled = True
    surgeplot.add_pressure(plotaxes, bounds=pressure_limits)
    surgeplot.add_land(plotaxes)

    # Wind field
    plotfigure = plotdata.new_plotfigure(name='Wind Speed')
    plotfigure.show = surge_data.wind_forcing and True

    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = regions['Full Domain']['xlimits']
    plotaxes.ylimits = regions['Full Domain']['ylimits']
    plotaxes.title = "Wind Field"
    plotaxes.afteraxes = surge_afteraxes
    plotaxes.scaled = True
    surgeplot.add_wind(plotaxes, bounds=wind_limits)
    surgeplot.add_land(plotaxes)

    # ========================================================================
    #  Figures for gauges
    # ========================================================================
    plotfigure = plotdata.new_plotfigure(name='Gauge Surfaces', figno=300,
                                         type='each_gauge')
    plotfigure.show = True
    plotfigure.clf_each_gauge = True

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = [0, 2.5]
    # plotaxes.xlabel = "Days from landfall"
    # plotaxes.ylabel = "Surface (m)"
    plotaxes.ylimits = [-1, 5]
    plotaxes.title = 'Surface'

    def gauge_afteraxes(cd):

        axes = plt.gca()
        surgeplot.plot_landfall_gauge(cd.gaugesoln, axes)

        # Fix up plot - in particular fix time labels
        axes.set_title('Station %s' % cd.gaugeno)
        axes.set_xlabel('Days relative to landfall')
        axes.set_ylabel('Surface (m)')
        axes.set_xlim([0, 2.5])
        axes.set_ylim([-1, 5])
        axes.set_xticks([0, 1, 1.5, 2.0])
        axes.set_xticklabels([r"$-2$", r"$-1$", r"$0$", r"$1$"])
        axes.grid(True)
    plotaxes.afteraxes = gauge_afteraxes

    # Plot surface as blue curve:
    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    # plotitem.plot_var = 3
    # plotitem.plotstyle = 'b-'

    #
    #  Gauge Location Plot
    #
    def gauge_location_afteraxes(cd):
        plt.subplots_adjust(left=0.12, bottom=0.06, right=0.97, top=0.97)
        surge_afteraxes(cd)
        gaugetools.plot_gauge_locations(cd.plotdata, gaugenos='all',
                                        format_string='ko', add_labels=True)

    plotfigure = plotdata.new_plotfigure(name="Gauge Locations")
    plotfigure.show = True

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.title = 'Gauge Locations'
    plotaxes.scaled = True
    plotaxes.xlimits = regions['Moriches Bay']['xlimits']#[-74.03, -73.95]
    plotaxes.ylimits = regions['Moriches Bay']['ylimits']#[40.48, 40.44]
    plotaxes.afteraxes = gauge_location_afteraxes
    surgeplot.add_surface_elevation(plotaxes, bounds=surface_limits)
    surgeplot.add_land(plotaxes)
    plotaxes.plotitem_dict['surface'].amr_patchedges_show = [0] * 10
    plotaxes.plotitem_dict['land'].amr_patchedges_show = [0] * 10

    

# ========================================================================
    #  Set up for moving particles
    # ========================================================================
    print('Reading all gauges...')
    gauge_solutions = particle_tools.read_gauges(gaugenos='all', 
                                                 outdir=plotdata.outdir)

    gaugenos_lagrangian = [k for k in gauge_solutions.keys() \
                if gauge_solutions[k].gtype=='lagrangian']
    gaugenos_stationary = [k for k in gauge_solutions.keys() \
                if gauge_solutions[k].gtype=='stationary']

    #print('+++ gaugenos_lagrangian: ',gaugenos_lagrangian)
    
    def add_particles(current_data):
        t = current_data.t

        # plot recent path:
        t_path_length = 0.5   # length of path trailing particle
        kwargs_plot_path = {'linewidth':1, 'color':'k'}
        particle_tools.plot_paths(gauge_solutions, 
                                  t1=t-t_path_length, t2=t, 
                                  gaugenos=gaugenos_lagrangian, 
                                  kwargs_plot=kwargs_plot_path)

        # plot current location:
        kwargs_plot_point = {'marker':'o','markersize':3,'color':'k'}
        particle_tools.plot_particles(gauge_solutions, t, 
                                      gaugenos=gaugenos_lagrangian, 
                                      kwargs_plot=kwargs_plot_point)  

        # plot any stationary gauges:
        gaugetools.plot_gauge_locations(current_data.plotdata, \
             gaugenos=gaugenos_stationary, format_string='kx', add_labels=False)
        kwargs={'loc':'upper left'}
        legend_tools.add_legend(['Lagrangian particle','Stationary gauge'],
                linestyles=['',''], markers=['o','x'],
                loc='lower right', framealpha=0.5, fontsize=10)


    def speed(current_data):
        from pylab import sqrt, where, zeros
        from numpy.ma import masked_where, allequal
        q = current_data.q
        h = q[0,:,:]
        hs = sqrt(q[1,:,:]**2 + q[2,:,:]**2)
        where_hpos = (h > 1e-3)
        s = zeros(h.shape)
        s[where_hpos] = hs[where_hpos]/h[where_hpos]
        s = masked_where(h<1e-3, s) # if you want 0's masked out
        #s = s * 1.94384  # convert to knots
        return s

    speed_cmap = colormap.make_colormap({0:[0,1,1], 0.5:[1,1,0], 1:[1,0,0]})

    # -----------------------------------------
    # Figure for pcolor plot
    # -----------------------------------------
    plotfigure = plotdata.new_plotfigure(name='pcolor', figno=0)
    plotfigure.kwargs = {'figsize': (6.4, 4.8)}

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes('pcolor')
    plotaxes.title = 'Speed'
    plotaxes.scaled = False
    plotaxes.xlimits = regions['Moriches Bay']['xlimits']
    plotaxes.ylimits = regions['Moriches Bay']['ylimits']
    plotaxes.afteraxes = add_particles

    # Water
    plotitem = plotaxes.new_plotitem(plot_type='2d_pcolor')
    plotitem.plot_var = speed
    plotitem.pcolor_cmap = speed_cmap

    plotitem.pcolor_cmin = 0.
    plotitem.pcolor_cmax = 2.
    plotitem.add_colorbar = True
    plotitem.colorbar_label = 'm/s'
    plotitem.amr_celledges_show = [0, 0, 0]
    plotitem.amr_patchedges_show = [0]
    plotitem.amr_patchedges_color = ['m', 'g', 'w']

    # Land
    plotitem = plotaxes.new_plotitem(plot_type='2d_pcolor')
    # plotitem.show = False
    plotitem.plot_var = geoplot.land
    plotitem.pcolor_cmap = geoplot.land_colors
    plotitem.pcolor_cmin = 0.0
    plotitem.pcolor_cmax = 100.0
    plotitem.add_colorbar = False
    plotitem.amr_celledges_show = [0, 0, 0]

    # Add contour lines of topography:
    plotitem = plotaxes.new_plotitem(plot_type='2d_contour')
    plotitem.show = False
    plotitem.plot_var = geoplot.topo
    from numpy import arange, linspace
    plotitem.contour_levels = arange(-75, 75, 10)
    # plotitem.contour_nlevels = 10
    plotitem.amr_contour_colors = ['g']  # color on each level
    plotitem.kwargs = {'linestyles': 'solid'}
    plotitem.amr_contour_show = [1, 1, 1]  # show contours only on finest level
    plotitem.celledges_show = 0
    plotitem.patchedges_show = 0


    # -----------------------------------------
    # Figure for grids alone
    # -----------------------------------------
    plotfigure = plotdata.new_plotfigure(name='grids', figno=2)
    plotfigure.show = False

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = [0, 1]
    plotaxes.ylimits = [0, 1]
    plotaxes.title = 'grids'
    plotaxes.scaled = True

    # Set up for item on these axes:
    plotitem = plotaxes.new_plotitem(plot_type='2d_patch')
    plotitem.amr_patch_bgcolor = ['#ffeeee', '#eeeeff', '#eeffee']
    plotitem.amr_celledges_show = [1, 1, 0]
    plotitem.amr_patchedges_show = [1]

    # -----------------------------------------
    # Figures for fgmax plots
    # -----------------------------------------
    # Note: You need to move fgmax png files into _plots/fgmax_plots after
    # creating them, e.g., by running the process_fgmax notebook or script.
    # The lines below just create links to these figures from _PlotIndex.html

    otherfigure = plotdata.new_otherfigure(name='max depth on fgmax grid 1',
                                           fname='fgmax_plots/fgmax0001_h_onshore.png')

    otherfigure = plotdata.new_otherfigure(name='max speed on fgmax grid 1',
                                           fname='fgmax_plots/fgmax0001_speed.png')

    otherfigure = plotdata.new_otherfigure(name='max elevation on fgmax grid 2',
                                           fname='fgmax_plots/fgmax0002_surface.png')

    otherfigure = plotdata.new_otherfigure(name='max depth on fgmax grid 3',
                                           fname='fgmax_plots/fgmax0003_h_onshore.png')

    otherfigure = plotdata.new_otherfigure(name='max speed on fgmax grid 3',
                                           fname='fgmax_plots/fgmax0003_speed.png')

    # add additional lines for any other figures you want added to the index.

    # -----------------------------------------
    # Plots of timing (CPU and wall time):

    def make_timing_plots(plotdata):
        import os
        from clawpack.visclaw import plot_timing_stats
        try:
            timing_plotdir = plotdata.plotdir + '/timing_figures'
            os.system('mkdir -p %s' % timing_plotdir)
            units = {'comptime': 'seconds', 'simtime': 'hours', 'cell': 'millions'}
            plot_timing_stats.make_plots(outdir=plotdata.outdir, make_pngs=True,
                                         plotdir=timing_plotdir, units=units)
            os.system('cp %s/timing.* %s' % (plotdata.outdir, timing_plotdir))
        except:
            print('*** Error making timing plots')

    # create a link to this webpage from _PlotIndex.html:
    otherfigure = plotdata.new_otherfigure(name='timing',
                                           fname='timing_figures/timing.html')
    otherfigure.makefig = make_timing_plots

    # -----------------------------------------
    # Parameters used only when creating html and/or latex hardcopy
    # e.g., via pyclaw.plotters.frametools.printframes:

    plotdata.printfigs = True                # print figures
    plotdata.print_format = 'png'            # file format
    plotdata.print_framenos = 'all'          # list of frames to print
    plotdata.print_gaugenos = 'all'   # list of gauges to print
    plotdata.print_fignos = 'all'            # list of figures to print
    plotdata.html = True                     # create html files of plots?
    plotdata.html_homelink = '../README.html'  # pointer for top of index
    plotdata.latex = True                    # create latex file of plots?
    plotdata.latex_figsperline = 2           # layout of plots
    plotdata.latex_framesperline = 1         # layout of plots
    plotdata.latex_makepdf = False           # also run pdflatex?
    plotdata.parallel = True                 # parallel plotting

    return plotdata
