# run with gdal_env

import pandas as pd
import xarray as xr
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib
import datetime as dt
import glob

from scipy.signal import argrelextrema

from lib import mod2smrf
from lib import albedo_dev

from scipy import optimize

import snobedo
from lib.common import *

import netCDF4 as nc



def get_last(num, xt, yv):
    """
    identifies final n decay periods in list of decay_days (xt)
    """
    #start from end
    rev = list(reversed(xt))
    
    # get loc of index of last n occurance of 0 values (indicating storm period restart
    out = [i for i, n in enumerate(rev) if n == 0]
    
    #use all periods if there are less than n periods
    if num >= len(out):
        return xt, yv
    
    # subtract subset list from original list to get index of start point
    else: 
        print('subsetting to n period')        
        idx = len(xt) - (out[num-1])

        return xt[idx:], yv[idx:]



# apply to area in domain

def decay_params(bb, time_dim, spring, parallel = True, plot=True, return_vals=False):
    """
    retrieve albedo fit coefficients for pixels with domain.
    """
    # format df for signal detection
    df = albedo_signal(bb, time_dim, parallel, plot)
    
    
    # returns x (days) and y (albedo values)for decay events, separated by WY and then concatenated to list
    xt, yv, = split_season_wys(df, spring)
        
   
        
    # total number of decay days. low counts mean pixel rarely sees snow in the 20yr dataset
    if len(yv) < 150:
                               
        print("less than 150 decay days identified")
        # return zeros for all vis model params if there are not enough points
        return np.array([0, 0, 0, 0, 0, 0, 0])
    
        
    
    # check if non-zero values are below threshold
    if any(((i < 0.7) & (i != 0)) for i in yv):
        
        
        # convert to broadband vals vis and ir
        spectral = list(map(mod2smrf.broad2spectral, yv))
        vis = [item[0] for item in spectral]
        ir = [item[1] for item in spectral]
        
        
         
        if return_vals == True:
            print('returning vals, NOT regression')
            return xt, vis
        
        
        ## VIS curve fit
        vis_model, cov = optimize.curve_fit(albedo_dev.albedo_vis_pwr,
                                    xdata=xt, 
                                    ydata=vis,
                                    maxfev=1000000,
                                    p0=[4300, 3600, 1.89],
                                    bounds=((0, 0, 1.5), (10000000, 10000000, 4)))        
                                            

        # print the parameters
        #vis_model

        if plot==True:
            print("plotting not yet updated for pwr function!")
            # PLOT VISIBLE
            #x = sasp_processed['storm_hours'][:100000].values
            fig, ax = plt.subplots(figsize=(18,8))

            #x = np.arange(0, 35.8, (1/24))
            x = np.arange(0, 30)
            x_obs = xt

            #y = albedo_regression(x, logistic_model[0], logistic_model[1])
            y = albedo_dev.albedo_vis(x, vis_model[0], vis_model[1], vis_model[2])
            y2 = albedo_dev.albedo_vis(x, 100, 1000, 2)
            y3 = vis


            plt.plot(x, y, '-', label="observed fit")
            plt.plot(x, y2, '-', label="empirical decay")
            plt.plot(x_obs, y3, '.', label="MODIS albedo (VIS)")
            plt.legend(loc="upper right")
            
            plt.show()
            
            plt.rcParams.update({'font.size': 17})

            #plt.savefig(f"../skiles_storage/AD_isnobal/out_plot/albedo_regression/plot_4.png",
            #                    facecolor='white', 
            #                    transparent=False, 
            #                    bbox_inches='tight')



        ## IR curve fit
        ir_model, cov = optimize.curve_fit(albedo_dev.albedo_ir_pwr,
                                    xdata=xt, 
                                    ydata=ir, 
                                    maxfev=1000000,
                                    p0=[100, 1000, 2],
                                    bounds=((0, 0, 1.5), (100000, 100000, 4)))
        # print the parameters
        #ir_model

        if plot==True:
            #PLOT IR
            print("plotting not yet updated for pwr function!")
            
            #x = sasp_processed['storm_hours'][:100000].values
            fig, ax = plt.subplots(figsize=(18,8))

            # x model
            x = np.arange(0, 30)
            # x obs
            x_obs = xt


            #y = albedo_regression(x, logistic_model[0], logistic_model[1])
            y = albedo_dev.albedo_ir(x, ir_model[0], ir_model[1], ir_model[2])
            y2 = albedo_dev.albedo_ir(x, -0.02123, 100, 1000)
            y3 = ir


            plt.plot(x, y, '-', label="observed fit")
            plt.plot(x, y2, '-', label="empirical decay")
            plt.plot(x_obs, y3, '.', label="SASP observed albedo (IR)")
            plt.legend(loc="upper right")


            #plt.savefig(f"../skiles_storage/AD_isnobal/out_plot/albedo_regression/plot_5.png",
            #                    facecolor='white', 
            #                    transparent=False, 
            #                    bbox_inches='tight')

        
        # dask processing was causing an issue when dirt was dropped, 
        # so dummy array is added (vis_model[3]) to return same amount of arrays
    
        # temp remove
        #return vis_model, ir_model
        return np.array([vis_model[0], 
                         vis_model[1], 
                         vis_model[2],
                         vis_model[2],
                         ir_model[0],
                         ir_model[1],
                         ir_model[2]
                        ])

    # if broadband does not drop below threshold
    else: 
        
        print('broadband does not drop below threshold')
        
        return np.array([0, 0, 0, 0, 0, 0, 0])
        

    

def albedo_signal(bb, time_dim, parallel, plot, n=2):
    """
    preprocess to identify local minima and maxima
    
    :param bb: 1D xarray of broadband albedo with dim of time 
    :param n: number of points to be check before and after (days in this case)
    """
    # extract 1D array, subset to time dim, to a df
    #df = bb['band_data'].to_dataframe()
    
    if parallel == True:
        # for ufunc version
        #print(time_dim)
        df = pd.DataFrame(bb)
        df.index = time_dim
        df = df.rename(columns={0: "band_data"})
        df.index.rename('time', inplace=True)
        #print(df)
        #df = bb.to_dataframe()

        # set 0 to nan before doing signal id, so that last decay events are not lost
        df.replace(0, np.nan, inplace=True)
        # works when use offset that makes 0 values negatives
        df = df.where(df > 0, np.nan)
        
    
    elif parallel == False: 
        # extract 1D array, subset to time dim, to a df
        df = bb['band_data'].to_dataframe()
        # format dataframe for processing
        df = df.drop(columns=['x', 'y', 'spatial_ref'])
        # not needed
        #df.reset_index(level=-1, drop=True, inplace=True)
        df.index = pd.to_datetime(df.index)
        
        # set 0 to nan before doing signal id, so that last decay events are not lost
        df.replace(0, np.nan, inplace=True)
        
    
        # works when use offset that makes 0 values negatives
        df = df.where(df > 0, np.nan)
        

    # Find local peaks
    df['min'] = df.iloc[argrelextrema(df.band_data.values, np.less_equal,
                        order=n)[0]]['band_data']
    df['max'] = df.iloc[argrelextrema(df.band_data.values, np.greater_equal,
                        order=n)[0]]['band_data']

    # Plot results
    if plot == True:
        plt.scatter(df.index, df['min'], c='r')
        plt.scatter(df.index, df['max'], c='g')
        plt.plot(df.index, df['band_data'])

        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(18.5, 10)

        plt.show()
        #plt.close()
        
        # do yearly plots
        for y in range(2001, 2021):

            start = pd.to_datetime(f"{y-1}-10-01")
            end = pd.to_datetime(f"{y}-09-30")
            
            dfwy = df[(df.index > start) & (df.index < end)]
            
            plt.scatter(dfwy.index, dfwy['min'], c='r')
            plt.scatter(dfwy.index, dfwy['max'], c='g')
            plt.plot(dfwy.index, dfwy['band_data'])

            fig = matplotlib.pyplot.gcf()
            fig.set_size_inches(18.5, 10)

            plt.show()
            plt.close()
            

    # subset to remove zero values
    #df['min'] = df['min'].where(df['min'] != 0, np.nan)
    #df['max'] = df['max'].where(df['max'] != 0, np.nan)
    
    return df



def decay_series(df):
    """
    process time series df with extrema (min and max) indentified from scipy.signal 
    to indivudal decay events, for one pixel
    """
    
    
    x_time = [] #pd.Series(dtype='float')
    y_val = [] #pd.Series(dtype='float')
   
    
    
    #subset to rows with a min OR max value, excluding all rows where both min and max are nan.
    idx_df = df[(df['min'].notna()) | (df['max'].notna())]
    #print(idx_df)
    
    # min and max should alteratively be nan to ensure we are looking at the decay, not reset.
    # subset to remove consecutive max values identified by scipy.signal
    # this won't effect decay because we use (-1) which compares to previous row, 
    # thereby removing a day during the reset, which is ignored.
    idx_df = idx_df[idx_df['max'].diff(-1).isna()]
    # same for min col
    idx_df = idx_df[idx_df['min'].diff(-1).isna()]
    # make sure all nans are np nans
    idx_df = idx_df.replace('nan', np.nan)
    
    
    #skip if there are no valid observations
    if idx_df.empty:
        print(f"no valid data")
        return x_time, y_val
        
    
    else:
        # make sure that it does not start with minimum, before albedo reset
        if pd.isna(idx_df['min'][0]) == False:
                idx_df = idx_df.iloc[1:]
    
    
        for n in np.arange(len(idx_df)-1, step=2):
 
   
            decay = df[idx_df.index[n]:idx_df.index[n+1]].copy()
            decay['count'] = range(len(decay))

            #print(decay)
            #print(decay.index)
            #print(decay[['band_data','count']].values)
            
            #print(decay['band_data'].iloc[0] - decay['band_data'].iloc[-1])
            
            # only add decay period if it occurs long enough to drop significantly            
            #if decay['band_data'].iloc[0] - decay['band_data'].iloc[-1] > 0.2:
            
            # remove duplicate min values
            #if decay['min'].count() > 1:
            #    idx = decay['min'].first_valid_index()
            #    decay = decay.loc[:idx]
            
            
            #must have significant decay 
            #if np.abs(decay['band_data'].iloc[0] - decay['band_data'].iloc[-1]) > 0.05:
                
            # reset must be a new snow event, so albedo must start relatively high
            # v1 results used 0.30
            if decay['band_data'].iloc[0] > 0.60:
                    
                    # avoid summer snow events
                    #if len(decay['band_data']) < 40:
                        
                        
                        #if decay['band_data'].median() > 0.28:
                
                             # check for na, 0 values
                            if decay['band_data'].isnull().values.any():
                                print(f"na found")
                                
                                # linear interpolate option
                                #decay['band_data'] = decay['band_data'].interpolate(method='linear')
                                
                                # drop row if band_data col is nan
                                decay = decay.dropna(subset=['band_data'])
                
                            print(decay)
                            #print(f" difference: {decay['band_data'].iloc[0] - decay['band_data'].iloc[-1]}")
            
            
                            #add to list
                            x_time.extend(decay['count'].tolist())
                            y_val.extend(decay['band_data'].tolist())
            
            #    #add to list
            #plt.show()
        
    return x_time, y_val


def split_season_wys(df, spring, last_events=False, n=3):
    
    xtl = []
    yvl = []
    
    for y in range(2001, 2021):
        if spring == False:
            start = pd.to_datetime(f"{y-1}-12-01")
            end = pd.to_datetime(f"{y}-02-28")
            #print(f"start: {start}, end: {end}")
            
        elif spring == True:
            start = pd.to_datetime(f"{y}-04-01")
            end = pd.to_datetime(f"{y}-07-28")
            #print(f"start: {start}, end: {end}")
            
        elif spring == None:
            print("must set spring or winter decay period!")
            
        dfwy = df[(df.index > start) & (df.index < end)]

        xt, yv = decay_series(dfwy)
        #plt.scatter(xt, yv, marker = '.')
        #plt.show()
        
        # for each WY, only return last n *valid* decay events
        if last_events == True:
            print('indexing last periods')
            xt, yv = get_last(num=n, xt=xt, yv=yv)
        
        
        xtl.extend(xt)
        yvl.extend(yv)
    
    return xtl, yvl
