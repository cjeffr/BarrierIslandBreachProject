__author__ = 'Pushkar Kumar Jain'

import pandas as pd
import numpy as np
import os

class ReadAmr(object):
    """ The object created via the fortq file has the information of domain as attributes

    Attributes:
        all_data (pandas dataframe): Fortq file read into pandas dataframe. Raw format
        Grid_level: Numpy array of grid numbers (patchwise) in fortq file
        AMR_level: Numpy array of amr levels (patchwise) in fortq file
        mx: Numpy array of number of cells in x direction (patchwise) in fortq files
        my: Numpy array of number of cells in y direction (patchwise) in fortq files
        xlow: Numpy array of southwest corners of x coordiates (patchwise) in fortq files
        ylow: Numpy array of southwest corners of y coordiates (patchwise) in fortq files
        dx: Numpy array of spacing x direction (patchwise) in fortq files
        dy: Numpy array of spacing in y direction (patchwise) in fortq files
        pandas_dataframe: Pandas dataframe that incorporates all the metadata to give a clean dataframe of fortq file

    """

    def __init__(self, path, frame, sort_in_frame=True):
        """ Constructs the object

        Args:
            filename (str): forq file name along with the path
            sort_in_frame (bool): If the final dataframe should have rows sorted with respect to amr_level, y_coord_ x_coord
        """
        q_file = os.path.join(path, f'fort.q{frame:04}')
        a_file = os.path.join(path, f'fort.a{frame:04}')
        q_data = pd.read_table(q_file, header=None, names = ["height","xvel","yvel","eta"], index_col=False, sep=r"\s+")
        a_data = pd.read_table(a_file, header=None, names=['bathy', 'capacity', 'length_ratio', 'friction', 'wind_x', 'wind_y', 'pressure'], index_col=False, sep=r'\s+')
        self.Grid_level = self.__capture_metadata(q_data, "grid_number")
        self.AMR_level = self.__capture_metadata(q_data, "AMR_level")
        self.mx = self.__capture_metadata(q_data, "mx")
        self.my = self.__capture_metadata(q_data, "my")
        self.x_low = self.__capture_metadata(q_data, "xlow")
        self.y_low = self.__capture_metadata(q_data, "ylow")
        self.dx = self.__capture_metadata(q_data, "dx")
        self.dy = self.__capture_metadata(q_data, "dy")
        self.pandas_dataframe = pd.merge(self.amrdataframe(q_data, sort_in_frame), self.amrdataframe(a_data, sort_in_frame), 
                                         on=['xcoord', 'ycoord', 'amrlevel', 'dx', 'dy'])
        


    def __capture_metadata(self, all_data, data_string):
        """ Scans through metadata and extracts AMR_level, mx , my of all the patches """
        out = all_data[all_data.xvel==data_string].height.values
        if(data_string in ["grid_number", "AMR_level", "mx", "my"]):
            out = out.astype('int32')
        return out

    def amrdataframe(self, all_data, sort_in_frame=True):
        """ Reads a pandas dataframe along with the metadata in fortq file. 

        Args:
            all_data (pandas dataframe): fortq data read as pandas dataframe along with the metadata
            sort_in_frame (bool): Sort pandas columns with respect to amr level, ycoord, xcoord (default True)

        Returns
        pandas dataframe:
            Returns a new pandas data frame with columns amrlevel, xcoord, ycoord, height, xvel, yvel, eta.
            amrlevel        -  1, 2, 3 etc. 
            xcoord, ycoord  -  The 2D coordinates in the domain
            height          -  Total height
            xvel, yvel      -  Momentum in x and y directions respectively
            eta             -  Water surface elevation
            
        """
        # Read all levels grid
        data = all_data.dropna()
        data = data.reset_index(drop=True)

        # Iterate over captured metadata and construct pandas column patch-wise
        for num,levelnum in enumerate(self.AMR_level):
            if num == 0:
                firstpoint = 0
            else:
                firstpoint = firstpoint + self.mx[num-1]*self.my[num-1]
            secondpoint = firstpoint + self.mx[num]*self.my[num]-1
            data.loc[firstpoint:secondpoint, 'amrlevel'] = levelnum 
            data.loc[firstpoint:secondpoint, 'dx'] = self.dx[num]
            data.loc[firstpoint:secondpoint,'dy'] = self.dy[num]
            data.loc[firstpoint:secondpoint, 'grid_number'] = self.Grid_level[num]
            x_left = self.x_low[num] + self.dx[num]/2.0
            x_right = self.x_low[num] + self.dx[num]*self.mx[num] - self.dx[num]/2.0
            y_down = self.y_low[num] + self.dy[num]/2.0
            y_up = self.y_low[num] + self.dy[num]*self.my[num] - self.dy[num]/2.0
            xrow = np.linspace(x_left , x_right, num = self.mx[num], dtype='float64')
            yrow = np.linspace(y_down , y_up, num = self.my[num], dtype='float64')
            xmesh,ymesh = np.meshgrid(xrow, yrow)
            data.loc[firstpoint:secondpoint,'xcoord']=np.ravel(xmesh)
            data.loc[firstpoint:secondpoint,'ycoord']=np.ravel(ymesh)
        
        #Rearranging xcoord and ycoord as per domain mesh
        if sort_in_frame:
            data.sort_values(by=['amrlevel','ycoord', 'xcoord'], ascending=[True,True,True],inplace=True)
            data = data.reset_index(drop=True)
        return data

     
if __name__=="__main__":
    hello = ReadAmr("../_output/fort.q0001", sort_in_frame=False)
    yoyo = hello.pandas_dataframe
    print (yoyo)
    #print yoyo[(yoyo.xcoord==47.0) & (yoyo.ycoord==5.0)]
    #print yoyo[(yoyo.ycoord > -29.7) & (yoyo.xcoord > -89.7)]
    #print yoyo[np.isclose(yoyo.xcoord, -64.5)]
    #print yoyo[(yoyo.ycoord==5.0)]
    #print yoyo.dtypes
