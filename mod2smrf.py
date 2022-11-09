# gdal_env
import xarray as xr
import numpy as np
import rioxarray 
from rioxarray.merge import merge_arrays
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from pathlib import Path
from datetime import datetime

from osgeo import gdal
from rasterio.enums import Resampling
#import gdal
import geopandas as gpd
from pyproj import Proj
from lib import snow_utils
from lib import SBSP_tools
import glob
import pathlib
import re
import click
import os

import plotly.express as px



def get_timestamp(f):
    """
    get date from filename
    will only parse date if contained within underscore
    e.x. _20221031_
    :param f: path string
    :return: tuple of datetime date and string date
    """
    split_str = f.split("_")
    idx = [char.isdigit() for char in split_str]
    str_fmt = split_str[idx.index(True)]
    dt_fmt = pd.to_datetime(str_fmt)
    
    #nums = re.findall('\d+', f)
    return dt_fmt, str_fmt

# deprecated
def lm(A, B, illum, gs_image):
    """
    from Painter 2009
    """
    alb = 1 - (A * illum * (gs_image ** (B * illum)))
    
    return alb

def broad2spectral(alb_bb):
    """
    :returns: tuple: alb_v, alb_ir
    
    from MS:
    NIRalb =0.1608exp(1.7661*BBalb)
    VISalb = (BBalb-0.4505*NIRalb)/0.5495
    
    Uses 700nm cutoff from vis to nir (to match SMRF)
    """
    
    x = 1.7661 * alb_bb
    alb_nir = 0.1608 * np.exp(x)
    
    y = 0.4505 * alb_nir
    alb_vis = (alb_bb - y) / 0.5495
    
    return alb_vis, alb_nir
    
# deprecated
def spectral_albedo(gs_image):
    """
    solve for visible and IR albedo, using Painter 2009
    
    """
    #30 degree illum angle
    #WARN: A & B coeffs covary with illum angle 
    
    illum = 1 - np.cos(30)
    # VIS
    A_vis = 0.0040
    B_vis = 0.4730
    # NIR
    #A_ir = 0.2725
    #B_ir = 0.1791
    
    A_ir = 0.2725
    B_ir = 0.1591
    # broad 30 deg
    A_br = 0.0765
    B_br = 0.2205
    
    
    #subtract gs from drfs
    #gs_image += drfs_image
    
    
    vis = lm(A_vis, 
             B_vis, 
             illum, 
             gs_image)
    
    nir = lm(A_ir, 
             B_ir, 
             illum, 
             gs_image)
    
    return vis, nir




def s2m(smrf_albedo, mod_albedo, wavelength):
    """
    loads existing albedo .nc file from AWSM run, and replaces albedo with satellite obsevations for the same time  period. 
    
    :param mod_albedo: dir of .nc of remote albedo observations
    :param smrf_albedo: location of template SMRF file to use for rio.reproject_match 
    :param wavelength: which band to process, "albedo_vis" or "albedo_ir"
    :return: smrf .nc files are written back to dir in same format with albedo replaced
    """
    valid = {'albedo_vis', 'albedo_ir'}
    if wavelength not in valid:
        raise ValueError("results: status must be one of %r." % valid)
    
    
    smrf_a = smrf_albedo
    #round coords
    #smrf_a['x'] = smrf_a['x'].round().astype(int)
    #smrf_a['y'] = smrf_a['y'].round().astype(int)
    smrf_a.rio.set_crs('EPSG:32613', inplace=True)
    #strip albedo_vis 
    
    # extract datetime from path
    _, ts = get_timestamp(mod_albedo)

    # open modis
    mod_a = xr.open_dataset(mod_albedo)
      
    # reproject to smrf
    mod_a = mod_a.rio.reproject_match(smrf_a, resampling=Resampling.cubic)
    #avoid float precision errors by reassining coords         
    mod_a = mod_a.assign_coords({
        "x": smrf_a.x,
        "y": smrf_a.y,
    })    
        
    #WARN: must do NA handling before using "other" in .where, which will fill all nans
    mod_a = mod_a.where(mod_a['band_data'] != 0, np.nan)
    mod_a = mod_a.where(mod_a['band_data'] != 65535, np.nan)
        
    #scale MODIS to between 0 and 1
    mod_a['band_data'] /= 100
    mod_a['band_data'] /= 100
    
    #WY2020 has MODIS processing errors causing very low albedo values. 
    mod_a = mod_a.where(mod_a['band_data'] > 0.3, np.nan)
    
    #offset to match SASP station data
    #derived from bias correction of 20 year SASP-MODIS comparison
    mod_a['band_data'] -= 0.1294
    
    if wavelength == 'albedo_vis':
        # returns vis alb
        mod_a['band_data'] = broad2spectral(mod_a['band_data'])[0]
        print('visible case')
 
    elif wavelength == 'albedo_ir':
        # returns ir alb
        mod_a['band_data'] = broad2spectral(mod_a['band_data'])[1]
        print('ir case')
    
    # remove unnesecary dim
    mod_a = mod_a.drop_vars('band')
    mod_a = mod_a.squeeze('band')

    mod_a['band_data'] = mod_a['band_data'].rio.write_nodata(np.nan)  
    
    #mod_a = mod_a.rio.interpolate_na(method="linear")
    #mod_a = mod_a.rio.interpolate_na(method="nearest")
    mod_a['band_data'] = mod_a['band_data'].fillna(0.35)
    
    #mod_a['band_data'] = mod_a['band_data'].interp(method="linear")
    # run "nearest" after "cubic" to clean up any edge effects
    #mod_a = mod_a.rio.interpolate_na(method="nearest")
    
    # create time range for each day
    time = pd.date_range(str(ts + ' 00:00'), 
                         str(ts + ' 23:00'), freq='1h')
    #create 24 hr time range to match SMRF
    hours = []

    for i in range(24):
        hours.append(mod_a)

    mod_a = xr.concat(hours, dim='time')
    mod_a['time'] = time
    
    mod_a = mod_a.drop_vars('spatial_ref')   
    #mod_a = mod_a.transpose("time", "y", "x")
    
    # rename to appropriate name
    mod_a = mod_a.rename({
        "band_data": wavelength
        }
    )
    
    # assign SMRF projection
    mod_a['projection'] = smrf_a['projection']
    
    return(mod_a)




def process_modis_smrf(smrf_albedo, mod_albedo, basin_dir, wy, wavelength):
    """
    for applying the s2m function to an awsm ouput dir

    :param mod_albedo: dir of .nc of remote albedo observations
    :param smrf_albedo: location of template SMRF file to use for rio.reproject_match 
    :param wavelength: which band to process, "albedo_vis" or "albedo_ir"
    """
    # sort inplace
    mod_albedo.sort()
    #smrf_albedo.sort()
    
    smrf_albedo = xr.open_dataset(smrf_albedo)
    # HACK to get wy2020. Needs fix
    #mod_albedo = mod_albedo[6940:]
    
    
    for f in mod_albedo:
        # get date from filename
        # will only parse date if contained within underscore
        # e.x. _20221031_
        dt, _ = get_timestamp(f)
        
        print(dt)
        
        # run for single water water year only
        if (dt >= datetime(wy, 4, 1)) & (dt <= datetime(wy, 9, 30)):
    
                print(f'processing MODIS file: {f}')
                #print(f'processing SMRF file: {b}')
                print('-----------------------')
                #run s2m 
                res = s2m(smrf_albedo, f, wavelength)
                
                # hour 15 is chosen for plot
                #res[wavelength][15].plot()
                #plt.show()
                
                _, ts = get_timestamp(f)
                
                outp = os.path.join(basin_dir, str('run' + ts), str(wavelength + '_modis.nc'))
            
                print(outp)
            
                # delete existing. HDF5 issues when reading existing file.
                if os.path.exists(outp):
                      os.remove(outp)
    
                try:
                    res.to_netcdf(path=outp, 
                                  format='NETCDF4', 
                                  mode='w',
                                  engine='netcdf4')
        
                except: 
                    raise PermissionError(f'WARN: file not saved. check if smrf directory exists for day: {ts}')
                

