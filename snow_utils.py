# processing for iSnobal visualization
import ulmo

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
from datetime import timedelta
import os

wsdlurl = 'https://hydroportal.cuahsi.org/Snotel/cuahsi_1_1.asmx?WSDL'
BASE_PATH = "/uufs/chpc.utah.edu/common/home/u1321700/skiles_storage/AD_isnobal/"


def snotel_fetch(sitecode, start_date, end_date, variablecode='SNOTEL:SNWD_D'):
    #print(sitecode, variablecode, start_date, end_date)
    values_df = None
    #try:
        #Request data from the server
    site_values = ulmo.cuahsi.wof.get_values(wsdlurl, sitecode, variablecode, start=start_date, end=end_date)
        #Convert to a Pandas DataFrame   
    values_df = pd.DataFrame.from_dict(site_values['values'])
        #Parse the datetime values to Pandas Timestamp objects
    values_df['datetime'] = pd.to_datetime(values_df['datetime'], utc=True)
        #Set the DataFrame index to the Timestamps
    values_df = values_df.set_index('datetime')
        #Convert values to float and replace -9999 nodata values with NaN
    values_df['value'] = pd.to_numeric(values_df['value']).replace(-9999, np.nan)
        #Remove any records flagged with lower quality
    values_df = values_df[values_df['quality_control_level_code'] == '1']
    #except:
    #    print("Unable to fetch %s" % variablecode)

    return values_df



def SNOTEL_locs_within_domain(wsdurl, domain_polygon, plot_locs):
    """
    get dataframe of snotel locations that fall within model domain geojson
    """
    sites = ulmo.cuahsi.wof.get_sites(wsdlurl)
    sites_df = pd.DataFrame.from_dict(sites, orient='index').dropna()
    sites_df['geometry'] = [Point(float(loc['longitude']), float(loc['latitude'])) for loc in sites_df['location']]
    sites_df = sites_df.drop(columns='location')
    sites_df = sites_df.astype({"elevation_m":float})
    
    sites_gdf_all = gpd.GeoDataFrame(sites_df, crs='EPSG:4326')
    sites_gdf_all = sites_gdf_all.to_crs('EPSG:32613')
    # load domain polygon as geojson
    ad_poly_gdf = gpd.read_file(domain_polygon)
    
    # change crs to UTM 13 to match isnobal output
    ad_poly_gdf = ad_poly_gdf.to_crs('EPSG:32613')
    #ad_poly_gdf.plot()
    
    ad_poly_geom = ad_poly_gdf.iloc[0].geometry
    # identify SNOTEL sites within geometry
    idx = sites_gdf_all.intersects(ad_poly_geom)
    ad_snotel_sites = sites_gdf_all.loc[idx]
    
    # get data for these stations
    sitecode = ad_snotel_sites.index[-1]
    
    if plot_locs == True:
        fig, ax = plt.subplots(figsize=(17, 9))
        ad_poly_gdf.plot(ax=ax)
        ad_snotel_sites.plot(ax=ax, color='orange')
        
        for x, y, label in zip(ad_snotel_sites.geometry.x, ad_snotel_sites.geometry.y, ad_snotel_sites.name):
            ax.annotate(label, xy=(x, y), xytext=(3, 3), textcoords="offset points")

        
        
    return ad_snotel_sites


def SNOTEL_data_within_domain(snotel_sites_df, start_date, end_date, variablecode='SNOTEL:SNWD_D'):
    value_dict = {}
    
    for i, sitecode in enumerate(snotel_sites_df.index):
        print('%i of %i sites: %s' % (i+1, len(snotel_sites_df.index), sitecode))
        out = snotel_fetch(sitecode = sitecode, 
                           variablecode = variablecode, 
                           start_date = start_date, 
                           end_date = end_date)
        if out is not None:
            value_dict[sitecode] = out['value']
            
        #Convert the dictionary to a DataFrame, automatically handles different datetime ranges (nice!)
        multi_df = pd.DataFrame.from_dict(value_dict)
    
    return multi_df



def reduce_ds_to_cluster(snotel_sites_df, wy_snow, variable, cluster_locs):
    """
    """
    
    df_list = []
    print('WARN: must use rounded ds coords!')
    
    for index, row in snotel_sites_df.iterrows():
        print(f"Indexing {index}")
        ds = wy_snow.sel(x=slice(cluster_locs[index]['lon'][0], cluster_locs[index]['lon'][1]), 
                         y=slice(cluster_locs[index]['lat'][0], cluster_locs[index]['lat'][1]),
                         )
        print(ds.dims)
        df = ds[variable].mean(dim=('x','y')).to_dataframe()
        
        df['site'] = index
        df_list.append(df)
    
    return pd.concat(df_list)

def reduce_ds_to_grid(snotel_sites_df, wy_snow, variable, cluster_locs):
    """
    reduce_ds_to_cluster, except it returns the grid cells around the SNOTEL point. 
    Also there is processing at the end to aling datetime and get rid of the multindex that
    results from keeping the lat-long-time indexing.
    """
    
    df_list = []
    print('WARN: must use rounded ds coords!')
    
    for index, row in snotel_sites_df.iterrows():
        print(f"Indexing {index}")
        ds = wy_snow.sel(x=slice(cluster_locs[index]['lon'][0], cluster_locs[index]['lon'][1]), 
                         y=slice(cluster_locs[index]['lat'][0], cluster_locs[index]['lat'][1]),
                         )
        print(ds.dims)
        
        # does not average, returns data for each of 4 cells
        #df = ds[variable].mean(dim=('x','y')).to_dataframe()
        df = ds[variable].to_dataframe()
        
        df['site'] = index
        
        df_list.append(df)
    
    
    df = pd.concat(df_list)
    
    df = df.reset_index() 
    df.index = pd.to_datetime(df['time'])
    # match snotel and align plot visually
    df.index = df.index - timedelta(hours=22)
    
    
    return df



def reduce_ds_to_cell(snotel_sites_df, wy_snow, variable):
    """
    uses dataframe with site names and geometry to subset xr.dataset from isnobal thickness output (snow.nc)
    *requires opening of nc files with xr.openmfdataset before execution*
    returns: dataframe with times series thickness and site categorical var
    """
    
    df_list = []
    
    for index, row in snotel_sites_df.iterrows():
        print(f"Indexing {index}")
        df = wy_snow.sel(x=row.geometry.x, 
                         y=row.geometry.y,
                         method='nearest', 
                         tolerance=500
                         )[variable].to_dataframe()
        df['site'] = index
        df_list.append(df)
    
    return pd.concat(df_list)





def reduce_nc_to_cell(snotel_sites_df, nc_path, variable='thickness'):
    """
    a faster version of reduce_ds_to_cell. It seems that
    the dask lazy loading method is causing big problems 
    when values are extracted. Use this until a faster
    parallel solution is found. 
    :returns: df with single variable value, with snotel loc as categorical
    """
    dfs = []

    for index, row in snotel_sites_df.iterrows():

        var_list = []
        time_list = []

        print(f"Selecting vars at loc: {index}")

        for file in sorted(nc_path):
            ds = xr.open_dataset(file)
            ds = ds.load()
            ds = ds.sel(time=datetime.time(22))
            ds = ds.sel(x=row.geometry.x, 
                        y=row.geometry.y,
                        method='nearest'
                        )
            var_list.append(ds[variable].item())
            time_list.append(ds['time'].item())
        
        #create df from list
        df = pd.DataFrame(var_list, index=time_list)
        df.index.names = ['time']
        df = df.rename(columns={0: variable})
        df['site'] = index


        dfs.append(df)
        
    return pd.concat(dfs)


def combine_extracted_vals(csvpath, snotel_data, variable):
    """
    read csv with output from xarray.openmfdataset selections
    """
    df = pd.read_csv(csvpath)
    df.index = pd.to_datetime(df['time'])
    #df[[variable,'site']].groupby('site').plot(subplots=True)
    
    df.index = df.index.tz_localize('UTC')
    
    snotel_subset = snotel_data.loc[df.index.min():df.index.max()]
    # truncate to day
    df.index = df.index.floor('d')
    
    if variable=='thickness':
    # convert snotel data to meters to match isnobal output units
        snotel_subset_m = snotel_subset * 0.0254
        
    if variable=='specific_mass':
    # convert snotel data to mm to match isnobal outputs units for SWE
        snotel_subset_m = snotel_subset * 25.4
    
    #print(df.info())
    #print(snotel_subset_m.info())
    
    
    df_n = pd.DataFrame()
    
    for col in snotel_subset_m.columns:
        #print(col)
        df_n[f"isnobal {variable} (at {col}"] = df[variable].loc[df['site']==col]
        df_n[col] = snotel_subset_m[col]
        
    
    return df_n




def precise_coords_for_snotel(ad_snotel_sites):
    """
    update df produced by ulmo containing snotel metadata. This pulls from a csv containing 
    precise locations and replaces them for whatever sites are in the original df. 
    changes CRS to UTM 13. 
    
    updated 20220913 to update existing ulmo df.
    """
    dfCO = pd.read_csv('../shared_cryosphere/dragar/snotel_locs/all_CO_SNOTEL_coords.csv', float_precision='high')
    #rename to match CUAHSI formatting
    dfCO['index'] = str('SNOTEL:') + dfCO['site_id'].astype(str) + str('_CO_SNTL')
    dfCO.index = dfCO['index']
    del dfCO['index']
    #create geodataframe 
    geometry = [Point(xy) for xy in zip(dfCO.lon, dfCO.lat)]
    df = dfCO.drop(['lon', 'lat'], axis=1)
    # to utm
    gdf = gpd.GeoDataFrame(df, crs="EPSG:4326", geometry=geometry)
    # index existing df with locs from ulmo
    gdf = gdf[gdf.index.isin(ad_snotel_sites.index)]
    # to UTM zone 
    gdf = gdf.to_crs(epsg=32613)
    
    print("using precise coordinates for SNOTEL locs")
    
    return gdf


def make_smrf_folders(wy, path):
    """
    create SMRF directory structure for given water year. 
    :param: wy: water-year
    :param: path: added to BASE_PATH, e.x. "animas_calibrated/wy2021/crb/"
    """
    start = f"{wy-1}-10-01" 
    end = f"{wy}-9-30"
    date_range = pd.date_range(start, end, freq='d')
    #print(date_range)
    for day in date_range:
        fmt_day = day.strftime("%Y%m%d")
        #print(fmt_day)
        dirpath = os.path.join(BASE_PATH, path, f"run{fmt_day}")
        os.mkdir(dirpath)


