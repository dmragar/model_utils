import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
#import contextily as ctx
import pyproj
import xarray as xr
import datetime
import datetime as dt
import dask


def H24_to_datetime(date_str):
    """changes odd 24-hr times used by CSAS to datetime object
    """
    if date_str[9:11] != '24':
        return pd.to_datetime(date_str, format='%Y %j %H%M')

    date_str = date_str[0:9] + '00' + date_str[11:]
    return pd.to_datetime(date_str, format='%Y %j %H%M') + \
           dt.timedelta(days=1)


def load_SBB_10year():
    """load 10 year SBB dataset from shared_cryosphere and format datetime
    returns: df 
    """
    df = pd.read_csv('../shared_cryosphere/dragar/SBSP/SBSP_1hr_2010-2020.csv')
    df['DOY'] = df['DOY'].astype(str).apply(lambda x: x.zfill(3))
    df['Hour'] = df['Hour'].astype(str).apply(lambda x: x.zfill(4))
    df['datetime']=df['Year'].astype(str)+" "+df['DOY']+" "+df['Hour'].astype(str)
    df.index=df['datetime'].apply(H24_to_datetime)
    
    return df

def load_SBB_03_09():
    """load 10 year SBB dataset from shared_cryosphere and format datetime
    returns: df 
    """
    df = pd.read_csv('../shared_cryosphere/dragar/SBSP/SBSP_1hr_2003-2009.csv')
    df['DOY'] = df['DOY'].astype(str).apply(lambda x: x.zfill(3))
    df['Hour'] = df['Hour'].astype(str).apply(lambda x: x.zfill(4))
    df['datetime']=df['Year'].astype(str)+" "+df['DOY']+" "+df['Hour'].astype(str)
    df.index=df['datetime'].apply(H24_to_datetime)
    
    return df


def load_SASP_10year():
    """load 10 year SBB dataset from shared_cryosphere and format datetime
    returns: df 
    """
    df = pd.read_csv('../shared_cryosphere/dragar/SASP/SASP_1hr_2010-2020.csv')
    # outrageous date handling 
    df['DOY'] = df['DOY'].astype(str).apply(lambda x: x.zfill(3))
    df['Hour'] = df['Hour'].astype(str).apply(lambda x: x.zfill(4))
    df['datetime']=df['Year'].astype(str)+" "+df['DOY']+" "+df['Hour'].astype(str)
    df.index=df['datetime'].apply(H24_to_datetime)
    
    return df

def load_SASP_wy2021(previous_df):
    """load wy2021 and rename headers to match the previous longer file
    """
    df = pd.read_excel('../shared_cryosphere/dragar/SASP/SASP_w2021.xlsx', 
              sheet_name=0, 
              header=[7]
          )
    # wy2021 file has different names for headers, solved by replacing
    #cols = dict(df.columns.values, previous_df.columns.values)
    
    # upper ano is first in the xlsx file, lower is second. This matches the order in the "processed" data in the dict below.
    res = {df.columns.values[i]: previous_df.columns.values[i] for i in range(len(df.columns.values))}
    df = df.rename(columns=res)
    
    # same outrageous data handling as before
    df['DOY'] = df['DOY'].astype(str).apply(lambda x: x.zfill(3))
    df['Hour'] = df['Hour'].astype(str).apply(lambda x: x.zfill(4))
    df['datetime']=df['Year'].astype(str)+" "+df['DOY']+" "+df['Hour'].astype(str)
    df.index=df['datetime'].apply(H24_to_datetime)
    
    return df
    

def load_SASP_03_09():
    """load 10 year SBB dataset from shared_cryosphere and format datetime
    returns: df 
    """
    df = pd.read_csv('../shared_cryosphere/dragar/SASP/SASP_1hr_2003-2009.csv')
    df['DOY'] = df['DOY'].astype(str).apply(lambda x: x.zfill(3))
    df['Hour'] = df['Hour'].astype(str).apply(lambda x: x.zfill(4))
    df['datetime']=df['Year'].astype(str)+" "+df['DOY']+" "+df['Hour'].astype(str)
    df.index=df['datetime'].apply(H24_to_datetime)
    
    return df


def albedo_vis(df):
    """process SBB 10 year dataset to visible albedo, *including removal of impossible values*
    returns: df with 'albedo_vis' col added
    """
    df['up_vis'] = df['PyUp_Unfilt_W'] - df['PyUp_Filt_W']
    df['down_vis'] = df['PyDwn_Unfilt_W'] - df['PyDwn_Filt_W']
    df['albedo_vis'] = df['down_vis'] / df['up_vis']
    # masking
    #df['albedo_vis'] = df['albedo_vis'].mask(df['albedo_vis'] > 1)
    #df['albedo_vis'] = df['albedo_vis'].mask(df['albedo_vis'] < 0)
    #df['albedo_vis'] = df['albedo_vis'].mask(df['PyUp_Unfilt_W'] < 5)
    
    return df

def albedo_broadband(df):
    """process SBB 10 year dataset to visible albedo, *including removal of impossible values*
    returns: df with 'albedo_broadband' col added
    """
    #df['up_vis'] = df['PyUp_Unfilt_W'] - df['PyUp_Filt_W']
    #df['down_vis'] = df['PyDwn_Unfilt_W'] - df['PyDwn_Filt_W']
    df['albedo_broadband'] = df['PyDwn_Unfilt_W'] / df['PyUp_Unfilt_W']
    #df['albedo_broadband'] = df['albedo_broadband'].mask(df['albedo_broadband'] > 1)
    #df['albedo_broadband'] = df['albedo_broadband'].mask(df['PyUp_Unfilt_W'] < 20)
    
    return df


def albedo_ir(df):
    """process SBB 10 year dataset to visible albedo, *including removal of impossible values*
    returns: df with 'albedo_ir' col added
    """
    #df['albedo_ir'] = df['albedo_broadband'] - df['albedo_vis']
    
    df['albedo_ir'] = df['PyDwn_Filt_W'] / df['PyUp_Filt_W']
    
    return df
    
    