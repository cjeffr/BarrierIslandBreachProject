#!/usr/bin/env python

import os
import datetime
import numpy
import matplotlib.pyplot as plt

import clawpack.geoclaw.surge.storm

path = os.path.join(os.getcwd(), "1938_hurdat2.txt")
storm = clawpack.geoclaw.surge.storm.Storm(path=path, file_format="HURDAT")

for (i, t) in enumerate(storm.t):
    if storm.central_pressure[i] == -99900:
        # From Kossin, J. P. WAF 2015
        a = -0.0025
        b = -0.36
        c = 1021.36
        storm.central_pressure[i] = (  a * storm.max_wind_speed[i]**2
                                     + b * storm.max_wind_speed[i]
                                     + c) * 100.0

fig = plt.figure()
axes = fig.add_subplot(1, 1, 1)
axes.plot(storm.t, storm.max_wind_speed)
axes.set_xlabel("Date")
axes.set_ylabel("m/s")

fig = plt.figure()
axes = fig.add_subplot(1, 1, 1)
axes.plot(storm.t, storm.central_pressure / 100.0)
axes.set_xlabel("Date")
axes.set_ylabel("hPa")
plt.show()