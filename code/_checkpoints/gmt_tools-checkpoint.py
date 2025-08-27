import subprocess
import os

class GmtTools:
    gmt = '/usr/bin/gmt/bin/gmt'
    gs = '-G/home/cat/.pyenv/shims/gs'
    append = '-O'

    def __init__(self, region, projection, infile, outfile, bathy, boundaries):
        self.region = region
        self.projection = projection
        self.infile = infile
        self.outfile = outfile
        self.bathy = bathy
        self.boundaries = boundaries

    def bmap(self, gmt=gmt):
        subprocess.run([gmt, 'psbasemap', self.region, self.projection,
                        self.boundaries[0],self.boundaries[1],
                        self.boundaries[2], '-K', '>', self.outfile])

    def image(self, gmt=gmt):
        subprocess.run([gmt, 'grdimage', self.bathy, self.projection, self.region, '-Cgeo', '-O', '>>', self.outfile])

    def convert_to_png(self, gmt=gmt, gs=gs):
        P = '-P'
        T = '-TG'
        A = '-A0/2/0/0'  # .2c/3c'
        output = '-F{}'.format('test')


        subprocess.run([gmt, 'psconvert',   T,A, output ])

    def water_img(self, water, gmt=gmt):
        subprocess.run([gmt, 'grdimage', water,
                        self.projection, self.region, '-Cgeo',
                        '-Q'])

    def create_cmap(self, color, gmt=gmt):
        A = '-A'
        C=f'-C'


