#---------------------- dist_funcs.py ----------------------
# This file contains the functions needed to create the 
# distributions of temperature or impact exposure in main text
# Figure 1, panels a.-d. and h.

import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import geopandas as gpd
import xagg as xa
import xhistogram.xarray as xhist
from tqdm.notebook import tqdm as tqdm
import pandas as pd
import geopandas as gpd
import copy
from sklearn.neighbors import KernelDensity
import os
import glob
import re


def calc_t_kdes(dss,mod,master_params,data_counties,
                seas=[60,243], # code also calculates growing season kde alone (just for tmean though)
                X_T = np.arange(250,320,1), # this is the X for the kde call; this seems to work well (in K)
                return_output = False,
                fn_suffix = '',
               ):
    w_out = dict()
    for k in dss:
        output_fn = master_params['impact_data_dir']+master_params['obs_mod']+'/aux_kdes_tas_'+mod+'_'+k+fn_suffix+'.nc'

        if master_params['overwrite'] or (not os.path.exists(output_fn)):
            w_out[k] = copy.deepcopy(dss[k].tas).to_dataset(name='tas')

            # Create variable 
            w_out[k]['kde'] = (w_out[k].isel(time=0).tas*np.nan).expand_dims(dim={'kdex':(X_T.squeeze())}).copy()
            w_out[k]['kde_grwng'] = (w_out[k].isel(time=0).tas*np.nan).expand_dims(dim={'kdex':(X_T.squeeze())}).copy()

            # Stack for iteration
            w_out[k] = w_out[k].stack(allv=[v for v in w_out[k].dims if v not in ['time','kdex']])
            
            # Calculate kdes
            for loc_idx in tqdm(np.arange(0,w_out[k].dims['allv'])):
                # Some models (GFDL-CM3-LE, grr) had some nans thrown in during 
                # pre-processing apparently; it doesn't affec the processing 
                # (they're at the margins and the bounding box is large enough)
                # but it does kill the kde code. This just leaves the kde as nan
                # if all are nan at a given loc/run combination.
                if not np.all(np.isnan(w_out[k]['tas'].isel(allv=loc_idx))):
                    try:
                        kde = KernelDensity(kernel="gaussian", bandwidth=0.1).fit(w_out[k].tas.isel(allv=loc_idx).values.reshape(-1,1))
                    except ValueError:
                        breakpoint()
                    w_out[k]['kde'][:,loc_idx] = kde.score_samples(X_T.reshape(-1,1))

                    kde = (KernelDensity(kernel="gaussian", bandwidth=0.1).
                           fit(w_out[k].tas.isel(time=(w_out[k].time.dt.dayofyear>=seas[0]-1)*
                                                     (w_out[k].time.dt.dayofyear<=seas[1]-1),allv=loc_idx).values.reshape(-1,1)))
                    w_out[k]['kde_grwng'][:,loc_idx] = kde.score_samples(X_T.reshape(-1,1))

            # Unstack
            w_out[k] = w_out[k].unstack()

            # Aggregate outcome
            wm = xa.pixel_overlaps(w_out[k],data_counties)
            agg = xa.aggregate(w_out[k][['kde','kde_grwng']],wm)
            w_out[k] = agg.to_dataset().drop('time',errors='ignore')

            w_out[k].attrs['SOURCE'] = 'calc_t_kdes()'

            if os.path.exists(output_fn):
                print(output_fn+' removed to allow overwrite.')
                os.remove(output_fn)
            w_out[k].to_netcdf(output_fn)
            print(output_fn+' saved!')
        else:
            if return_output:
                w_out[k] = xr.open_dataset(output_fn)
                print(output_fn+' already exists; loaded!')
            else:
                print(output_fn+' already exists; skipped!')
    if return_output:
        return w_out
    
    
def calc_binned_kdes(dss,mod,master_params,
                     var_params,data_counties,return_output = False):
    w_out = dict()
    for k in dss:
        output_fn = master_params['impact_data_dir']+master_params['obs_mod']+'/aux_kdes_'+mod+'_'+var_params['name']+'_'+k+'.nc'

        if master_params['overwrite'] or (not os.path.exists(output_fn)):
            w_out[k] = copy.deepcopy(dss[k].tas)

            tmp = w_out[k].values
            for bin_idx in np.arange(0,len(var_params['dmgf'])):
                tmp[(tmp>=var_params['bins'][bin_idx]) & 
                    (tmp<var_params['bins'][bin_idx+1])] = var_params['dmgf'][bin_idx]

            w_out[k][:,:] = tmp
            w_out[k] = w_out[k].to_dataset(name='outcome')

            # Create variable 
            w_out[k]['kde'] = (w_out[k].isel(time=0).outcome*np.nan).expand_dims(dim={'kdex':(var_params['X'].squeeze())}).copy()

            # Stack for iteration
            w_out[k] = w_out[k].stack(allv=[v for v in w_out[k].dims if v not in ['time','kdex']])

            # Calculate kdes
            for loc_idx in tqdm(np.arange(0,w_out[k].dims['allv'])):
                # Some models (GFDL-CM3-LE, grr) had some nans thrown in during 
                # pre-processing apparently; it doesn't affec the processing 
                # (they're at the margins and the bounding box is large enough)
                # but it does kill the kde code. This just leaves the kde as nan
                # if all are nan at a given loc/run combination. 
                if not np.all(np.isnan(w_out[k]['outcome'].isel(allv=loc_idx))):
                    kde = KernelDensity(kernel="gaussian", bandwidth=var_params['bandwidth']).fit(w_out[k].outcome.isel(allv=loc_idx).values.reshape(-1,1))
                    w_out[k]['kde'][:,loc_idx] = kde.score_samples(var_params['X'].reshape(-1,1))

            # Unstack
            w_out[k] = w_out[k].unstack()

            # Aggregate outcome
            wm = xa.pixel_overlaps(w_out[k],data_counties)
            agg = xa.aggregate(w_out[k].drop('outcome'),wm)
            w_out[k] = agg.to_dataset().drop('time',errors='ignore')

            w_out[k].attrs['SOURCE'] = 'calc_binned_kdes()'
            w_out[k].attrs['DESCRIPTION'] = 'kdes of exposure to intensity of outcome'
            w_out[k].attrs['kde_bandwidth'] = var_params['bandwidth']
            w_out[k].attrs['model'] = mod
            w_out[k].attrs['variable'] = var_params['name']

            if os.path.exists(output_fn):
                print(output_fn+' removed to allow overwrite.')
                os.remove(output_fn)
            w_out[k].to_netcdf(output_fn)
            print(output_fn+' saved!')
        else:
            if return_output:
                w_out[k] = xr.open_dataset(output_fn)
                print(output_fn+' already exists; loaded!')
            else:
                print(output_fn+' already exists; skipped!')
    if return_output:
        return w_out
    
    
def calc_piecewise_hists(dss,mod,master_params,
                        var_params,data_counties,
                        seas_range = [60,254],
                        return_output = False):
    w_out = dict()
    for k in dss:
        output_fn = master_params['impact_data_dir']+master_params['obs_mod']+'/aux_hists_'+mod+'_'+var_params['name']+'_'+k+'.nc'
        
        if master_params['overwrite'] or (not os.path.exists(output_fn)):

            w_out[k] = copy.deepcopy(dss[k].tas.isel(time=((dss[k].time.dt.dayofyear>=seas_range[0]) &
                                                                     (dss[k].time.dt.dayofyear<=seas_range[1]))))

            tmp = w_out[k].values
            for bin_idx in np.arange(0,len(var_params['dmgf'])):
                tmp[(tmp>=var_params['dmgf'][bin_idx]['bounds'][0]) & 
                    (tmp<var_params['dmgf'][bin_idx]['bounds'][1])] = (tmp[(tmp>=var_params['dmgf'][bin_idx]['bounds'][0]) & 
                                                                (tmp<var_params['dmgf'][bin_idx]['bounds'][1])] *
                                                              var_params['dmgf'][bin_idx]['m'] + 
                                                              var_params['dmgf'][bin_idx]['b'])

            w_out[k][:,:] = tmp

            w_out[k] = w_out[k].to_dataset(name='outcome')

            # calculate histogram bins (kde doesn't do great here since there are 
            # several saturation / point values in the outcomes...)
            w_out[k]['hist'] = xhist.histogram(w_out[k].outcome,bins=[var_params['X']],dim=['time'])
            
            # Unstack
            #w_out[k] = w_out[k].unstack()

            # Aggregate outcome
            wm = xa.pixel_overlaps(w_out[k],data_counties)
            agg = xa.aggregate(w_out[k].drop('outcome'),wm)
            w_out[k] = agg.to_dataset().drop('time',errors='ignore')
            
            w_out[k].attrs['SOURCE'] = 'calc_piecewise_hists()'
            w_out[k].attrs['DESCRIPTION'] = 'histogram of exposure to intensity of outcome'
            w_out[k].attrs['seas_range'] = str(seas_range)
            w_out[k].attrs['model'] = mod
            w_out[k].attrs['variable'] = var_params['name']

            
            if os.path.exists(output_fn):
                print(output_fn+' removed to allow overwrite.')
                os.remove(output_fn)
            w_out[k].to_netcdf(output_fn)
            print(output_fn+' saved!')
        else:
            if return_output:
                w_out[k] = xr.open_dataset(output_fn)
                print(output_fn+' already exists; loaded!')
            else:
                print(output_fn+' already exists; skipped!')
    if return_output:
        return w_out

