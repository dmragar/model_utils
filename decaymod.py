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



def albedo_signal(df, n=4, plot=False):
    """
    preprocess to identify local minima and maxima
    
    :param df: result of xarray to_dateframe() 
    :param n: number of points to be check before and after (days in this case)
    """
    # format dataframe for processing
    df = df.drop(columns=['x', 'y', 'spatial_ref'])
    df.reset_index(level=-1, drop=True, inplace=True)
    df.index = pd.to_datetime(df.index)

    # Find local peaks
    df['min'] = df.iloc[argrelextrema(
        df.band_data.values, 
        np.less_equal,
        order=n)[0]]['band_data']
    df['max'] = df.iloc[argrelextrema(
        df.band_data.values, 
        np.greater_equal,
        order=n)[0]]['band_data']

    # Plot results
    if plot == True:
        plt.scatter(df.index, df['min'], c='r')
        plt.scatter(df.index, df['max'], c='g')
        plt.plot(df.index, df['band_data'])

        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(18.5, 10)
        plt.show()
    
    # subset to remove zero values
    df['min'] = df['min'].where(df['min'] != 0, np.nan)
    df['max'] = df['max'].where(df['max'] != 0, np.nan)
    
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
    idx_df = idx_df.replace('nan', np.nan)
    
    #skip if there are no valid observations
    if idx_df.empty:
        print(f"no valid data for {df.info()}")
    
    else:
        # make sure that it does not start with minimum, before albedo reset
        if pd.isna(idx_df['min'][0]) == False:
                idx_df = idx_df.iloc[1:]
    
        # iterate max min pairs and add each decay event to list
        for n in np.arange(len(idx_df)-1, step=2):
            #print(n)
            #print(n+1)
            #df[idx_df.index[n]:idx_df.index[n+1]].plot()
            decay = df[idx_df.index[n]:idx_df.index[n+1]].copy()
            decay['count'] = range(len(decay))

            #print(decay)
            #print(decay.index)
            #print(decay[['band_data','count']].values)

            x_time.extend(decay['count'].tolist())
            y_val.extend(decay['band_data'].tolist())
            plt.show()
        
    return x_time, y_val

