#---------------------- proj_funcs.py ----------------------
# This file contains the functions needed to project historical
# data using delta changes from climate models.

import xarray as xr
import numpy as np


def proj_climate(ds_ref,ds_base,ds_fut,var='tas'):
    ''' Project ds_ref by the difference in monthly averages between ds_fut and ds_base
    
    
    Parameters
    --------------------------
    ds_ref : xr.Dataset
        The dataset to be projected. Must be compatible
        with ds_base, ds_fut (see below). 
        
    ds_base, ds_fut : xr.Dataset
        The datasets used for projecting; the difference 
        ds_fut-ds_base in their monthly averages will be 
        used. As a result, ds_base and ds_fut must have
        compatible dimensions, variables, etc. (and with
        ds_ref), and a time dimension that allows for 
        monthly averages to be calculated. Not sure if
        existing errors can ensure that, so you just gotta
        be careful.
    
    
    Returns
    --------------------------
    ds_proj : xr.Dataset
        ds_ref, with the average change in monthly 
    
    '''
    
    # Get change in monthly averages
    d_ds = ds_fut.groupby('time.month').mean()-ds_base.groupby('time.month').mean()
    
    # Function for actual projection
    def delta_proj(x,origin=d_ds):
        if len(d_ds[var].dims) == 4:
            return x + origin[var][x.time.dt.month-1,:,:,:]
        elif len(d_ds[var].dims) == 3:
            return x + origin[var][x.time.dt.month-1,:,:]
    
    # Add to ds_ref
    ds_proj = ds_ref.groupby('time.month').apply(delta_proj)
    
    # Return
    return ds_proj
    