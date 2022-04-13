#---------------------- proc_funcs.py ----------------------
# This file contains the functions needed for the core 
# processing of the project

import xarray as xr
import xhistogram.xarray as xhist
import xagg as xa
import xesmf as xe
import numpy as np
import pandas as pd
import geopandas as gpd
from matplotlib import pyplot as plt
import matplotlib as mpl
import os
import re
import glob
import warnings
from tqdm.notebook import tqdm as tqdm
from aux_funcs import get_params


def bin_data(master_params,subset_params,bins,bin_name,save_output=True,return_output=False):
    ''' Calculate bin-days across all models
    
    Parameters
    --------------------------
    master_params : dict
    
    
    subset_params : dict
    
    bins : np.array
    
    bin_name : str
        used as an additional suffix for filenames, to differentiate
        between different bin schemes 
    
    save_output : bool, default True
    
    return_output : bool, default False
    
    Returns
    --------------------------
    hists_all : dict()
        if return_output is True, then a dict, with model keys, for each
        dataset with all the calculated bin-days, concatenated across all 
        timeframes set by subset_params for each model
    
    '''
    
    #-----------------------------------
    # Get list of models to process
    #-----------------------------------
    # Get list of all directory names in master_params['mod_data_dir']
    # that have files that satisfy master_params['mod_search_str']
    fns_base = [fn for fn in glob.glob(master_params['mod_data_dir']+'*/*.nc') if re.search(master_params['mod_search_str'],re.split('\/',fn)[-1])]
    models = [re.split('\/',fn)[-2] for fn in fns_base]
    models = np.unique(models)
    
    if return_output:
        hists_all = dict()
    
    #-----------------------------------
    # Process by model 
    #-----------------------------------
    for mod in models:
        #-------------------------------
        # Get output filename
        #-------------------------------
        # Get output filename 
        fns_tmp = [fn for fn in fns_base if re.search(mod+'\_',fn)]
        if len(fns_tmp)>1:
            warnings.warn('More than one file found for model '+mod+' and search string: '+master_params['mod_search_str']+'; the first one, '+fns_tmp[0]+', will be loaded.')
        fn = re.split('\/',fns_tmp[0])[-1]

        fn_out = (master_params['impact_data_dir']+master_params['obs_mod']+'/' + 
                  re.sub('\.nc',bin_name+'.nc',
                         re.sub('[0-9]{8}\-[0-9]{8}','-'.join(subset_params.keys()),
                            re.sub(re.split('\_',fn)[0],re.split('\_',fn)[0]+'bin',
                               re.sub(re.split('\_',fn)[4],mod+'proj',
                                   re.sub(mod,master_params['obs_mod'],
                                      re.sub(re.split('\_',fn)[1],'day',fn)))))))
        
        #-------------------------------
        # Process
        #-------------------------------
        if master_params['overwrite'] or (not os.path.exists(fn_out)):

            #-------------------------------
            # Load component files
            #-------------------------------
            fns = dict()

            # Get filename for historical file (this is "rg_fn" in wrapper_proj())
            obs_fn = [fn for fn in [k for k in os.walk(master_params['obs_data_dir']+master_params['obs_mod'])][0][2] 
                             if re.search(master_params['obs_search_str'],fn)]
            if len(obs_fn)>1:
                warnings.warn('More than one file found for observation data ('+master_params['obs_mod']+')...') 
            obs_fn = obs_fn[0]
            fns['hist'] = (master_params['proc_data_dir']+master_params['obs_mod']+'/'+re.sub(re.split('\_',obs_fn)[4],mod+'grid',obs_fn))


            # Get filenames for future files 
            for timeframe in [k for k in subset_params.keys() if k != 'hist']:
                fns[timeframe] = (master_params['proc_data_dir']+master_params['obs_mod']+'/'+
                                   re.sub(re.split('\_',obs_fn)[3],master_params['exp_name'],
                                   re.sub('[0-9]{8}\-[0-9]{8}','-'.join(subset_params[timeframe]),
                                   re.sub(re.split('\_',obs_fn)[4],mod+'proj',obs_fn))))


            #-------------------------------
            # Calculate histograms (bin-days)
            #-------------------------------
            hists = dict()
            for timeframe in subset_params:
                # Load file
                try:
                    ds_tmp = xr.open_dataset(fns[timeframe])
                except FileNotFoundError:
                    print(fns[timeframe]+' does not exist! Skipping.')
                    continue
                # Calculate histogram
                hists[timeframe] = xhist.histogram(ds_tmp[[v for v in ds_tmp if 'nds' not in v][0]],dim=['time'],bins=bins)

            del ds_tmp

            # Merge into one ds
            hists = xr.merge([v.to_dataset(name=k) for k,v in hists.items()])

            # Rename the *_bin dimension to "bin" 
            hists = hists.rename({[k for k in hists.dims if re.search('\_bin',k)][0]:'bin'})

            # Change all 0s (where every bin is 0) back to nan
            # (this is necessary if the geographical bboxes don't 
            # quite overlap between the timeframes)
            (off_map,_)=xr.broadcast(hists.sum('bin'),hists)
            if 'run' in off_map.dims:
                off_map['hist'] = off_map.hist.isel(run=0)
            hists = hists.where(off_map>0) 


            #-------------------------------
            # Export
            #-------------------------------
            # Add bin bounds to ds
            hists['bin_bnds'] = xr.DataArray(np.vstack([bins[0:-1],bins[1:]]),
                                             dims=['bnds','bin'],
                                             coords=[np.arange(0,2),hists.bin])

            # Export
            hists.attrs['SOURCE'] = 'bin_data()'
            hists.attrs['DESCRIPTION'] = 'number of days in each bin at each location, concatenated from the following timeframes: '+', '.join(subset_params.keys())

            if save_output:
                hists.to_netcdf(fn_out)
                print(fn_out+' saved!')

            if return_output:
                hists_all[mod] = hists
        else:
            print(fn_out+' already exists! Skipped.')
            
            if return_output:
                hists_all[mod] = xr.open_dataset(fn_out)
    #-------------------------------
    # Export
    #-------------------------------
    if return_output:
        return hists_all
        

def bin_dmgf(master_params,subset_params,hists_all,dmgf,data_counties,var_name = 'tpop',scale=1/100000):
    ''' Calculate impact using a damage function relating a change in a variable to number of days in bins 
    
    Change in a variable due to a day in each bin, vs. a day in the neutral 
    bin (arbitrarily set, but usually set to be at the inflection point at 50-60 F).
    Similar setup to bin_log_dmgf(), but with the [#Get change per bin] and [#Get 
    total change sections different]; these two functions should be combined probably. 
    
    Assumptions: the dmgf is based on the additional impact of a single day in a bin 
    on a variable's *per year* rate. So first, the impact of each bin is calculated:
    
        dI =  Sum_b[(1/n_years)*(X_b_f - X_b_i)*D_b]
    
    where dI is the impact, on each pixel, of the relative number of days vs. optimum
    on a variable Y, calculated by the change in number of days (X_b_*) in each bin (b) 
    per year, multiplied by the damage function at that bin (D_b). 
    
    This is the rate change in the variable, and is saved as "dimp_[timeframe]"
    in the dataset for each timeframe. 
    
    The rate change is then aggregated to the counties in data_counties - so
    each county has an average change in the rate variable. 
    
    The total change is then calculated by: 
    
        dI_tot = Sum_c[(dI_c*Y_c*a]
        
    for the sum counties c (and the now aggregated dI_c), 
    and scale variable "a" (by default 1/100,000; see parameters below)
         
    This variable is saved as "sum_dimp_[timeframe]" for each timeframe. 
    
    Parameters
    -------------------
    master_params : dict
    
    subset_params : dict
    
    hists_all : dict of xr.Datasets
        the output to bin_data()
        
    dmgf : xr.DataArray with dimension "bin"
        the damage function. Must have the same number of 
        bins as the datasets in hists_all(). If the bins 
        aren't identical in their coordinates, the bins
        of the dmgf are set to the bins in hists_all().
        
    data_counties : gpd.GeoDataFrame
        the rate change, calculated using the dmgf 
        above, is aggregated over the counties in data_counties,
        and the total impact is then calculated using the 
        variable [var_name], which must be a column 
        in data_counties
        
    var_name : str; default 'tot_pop'
        the variable name in data_counties with which to 
        calculate the actual change in levels from the 
        rate change
        
    scale : int; default 1/100,000
        the change in levels, as calculated above, is multiplied
        by this scale before being summed over counties. The
        default scale assumes the input default variable 
        ('tot_pop' above) is a (mortality) rate / 100,000 
        
    
    Returns
    -------------------
    impact_ds : xr.Dataset
        a dataset with dimensions "county_idx" (and "run")
        with all the original variables in [hists_all] + 
        the calculated "dimp_[timeframe]" and "sum_dimp_[timeframe]",
        aggregated to be county averages
    
    '''

    
    #-------------------------------
    # Get change in impact variable
    #-------------------------------
    impact_ds = dict()
    for mod in tqdm(hists_all):
        print('\nprocessing model '+mod+'!')
        # Make sure the bin coordinates are unaligned, otherwise
        # the dmgf 
        if not hists_all[mod].bin.equals(dmgf.bin):
            dmgf['bin'] = hists_all[mod].bin

        prob_timeframes = [] # to filter out timeframes that weren't calculated
        for timeframe in [k for k in subset_params.keys() if k != 'hist']:
            try: 
                # Get change per bin
                hists_all[mod]['d'+timeframe] = (hists_all[mod][timeframe]-hists_all[mod].hist)/len(pd.date_range(*subset_params['hist'],freq='Y'))
                # Get change in mortality
                hists_all[mod]['dimp_'+timeframe] = hists_all[mod]['d'+timeframe].dot(dmgf)
            except KeyError: 
                prob_timeframes = prob_timeframes + [timeframe]
                print('Issue with model '+mod+', timeframe '+timeframe)
            
        #-------------------------------
        # Aggregate impact variable to polygons
        #-------------------------------
        if os.path.exists(master_params['aux_data_dir']+'wm_'+mod+'/'):
            weightmap = xa.read_wm(master_params['aux_data_dir']+'wm_'+mod)
        else:
            weightmap = xa.pixel_overlaps(hists_all[mod],data_counties,subset_bbox=False)
        impact_agg = xa.aggregate(hists_all[mod][['dimp_'+k for k in subset_params if k not in prob_timeframes + ['hist']]],weightmap)
        impact_ds[mod] = impact_agg.to_dataset(loc_dim='county_idx')

        #-------------------------------
        # Get total change
        #-------------------------------
        # Sum total change in impact 
        for timeframe in [k for k in subset_params.keys() if k not in prob_timeframes + ['hist']]:
            impact_ds[mod]['sum_dimp_'+timeframe] = ((impact_ds[mod]['dimp_'+timeframe]*scale)*impact_ds[mod][var_name]).sum('county_idx')
            
    #-------------------------------
    # Return
    #-------------------------------
    return impact_ds



def bin_log_dmgf(master_params,subset_params,hists_all,dmgf,data_counties,var_name = 'gdp_pp',op='mean',scale=1):
    ''' Calculate impact using a damage function relating a change in a logged variable to number of days in bins 
    
    Change in a logged variable due to a day in each bin, vs. a day in the neutral 
    bin (arbitrarily set, but usually set to be at the inflection point at 50-60).
    Similar setup to bin_dmgf() above, but with the [#Get change per bin] and [#Get 
    total change] sections different; these two functions should be combined probably. 
    
    Assumptions: the dmgf is based on the additional impact of a single day in a bin 
    on a logged variable's *per year* rate. So first, the impact of each bin is calculated:
    
                e^[sum(X_b_f*D_b)/n_years]
        dI =  ------------------------------
                e^[sum(X_b_i*D_b)/n_years]
    
    where dI is the impact, on each pixel, of the days vs. optimum in the future / 
    days vs. optimum in the reference climate (i.e., the impact of the temperature 
    changes between the future and reference climates based on the damage function D, 
    which relates number of days X in bin b to some variable Y).
    
    This is the fractional change in the variable, and is saved as "dimp_[timeframe]"
    in the dataset for each timeframe. 
    
    The fractional change is then aggregated to the counties in data_counties - so
    each county has an average fractional change in the variable. 
    
    The actual change is then calculated by 
    (if op='sum', giving the sum total effect over all counties): 
    
        dI_tot = Sum_c[(dI_c - 1)*Y_c*a]
        
    or (if op='mean', giving the effect on an average county):
    
        dI_mean = Sum_c[(dI_c - 1)*Y_c*a] / n_counties
        
    for the sum or mean over counties c (and the now aggregated dI_c), 
    and scale variable "a" (by default 1; see parameters below)
    (dI_c - 1 is just [I_f-I_i]/I_i)
        
    Regardless of if the sum or mean is calculated, this variable is saved as 
    "sum_dimp_[timeframe]" for each timeframe. 
    
    To get the average fractional change, therefore: 
    impact_ds['dimp_[timeframe]'].mean('county_idx')
    
    
    Parameters
    -------------------
    master_params : dict
    
    subset_params : dict
    
    hists_all : dict of xr.Datasets
        the output to bin_data()
        
    dmgf : xr.DataArray with dimension "bin"
        the damage function. Must have the same number of 
        bins as the datasets in hists_all(). If the bins 
        aren't identical in their coordinates, the bins
        of the dmgf are set to the bins in hists_all().
        
    data_counties : gpd.GeoDataFrame
        the fractional change, calculated using the dmgf 
        above, is aggregated over the counties in data_counties,
        and the total impact is then calculated using the 
        variable [var_name], which must be a column 
        in data_counties
        
    var_name : str; default 'gdp_pp'
        the variable name in data_counties with which to 
        calculate the actual change in levels from the 
        fractional change
        
    op : str; default 'mean'
        if 'mean', then the mean change in the variable
        [var_name] over all counties is returned as 
        "sum_dimp_[timeframe]"; otherwise the total 
        (summed) change over all counties is returned.
        
    scale : int; default 1
        the change, as calculated above, is multiplied
        by this scale before being summed / averaged. 
        
    
    Returns
    -------------------
    impact_ds : xr.Dataset
        a dataset with dimensions "county_idx" (and "run")
        with all the original variables in [hists_all] + 
        the calculated "dimp_[timeframe]" and "sum_dimp_[timeframe]",
        aggregated to be county averages
    
    '''
    #-------------------------------
    # Get change in impact variable
    #-------------------------------
    impact_ds = dict()
    for mod in tqdm(hists_all):
        print('\nprocessing model '+mod+'!')
        
        # Make sure the bin coordinates are unaligned, otherwise
        # the dmgf doesn't match up and the result is all 0s
        if not hists_all[mod].bin.equals(dmgf.bin):
            dmgf['bin'] = hists_all[mod].bin

        prob_timeframes = []
        for timeframe in [k for k in subset_params.keys() if k != 'hist']:
            try:
                # Get change per bin
                hists_all[mod]['dimp_'+timeframe] = (np.exp(hists_all[mod][timeframe].dot(dmgf)/len(pd.date_range(*subset_params['hist'],freq='Y')))/
                                                     np.exp(hists_all[mod]['hist'].dot(dmgf)/len(pd.date_range(*subset_params['hist'],freq='Y'))))
            except KeyError: 
                prob_timeframes = prob_timeframes + [timeframe]
                print('Issue with model '+mod+', timeframe '+timeframe)
            
        #-------------------------------
        # Aggregate impact variable to polygons
        #-------------------------------
        if os.path.exists(master_params['aux_data_dir']+'wm_'+mod+'/'):
            weightmap = xa.read_wm(master_params['aux_data_dir']+'wm_'+mod)
        else:
            weightmap = xa.pixel_overlaps(hists_all[mod],data_counties,subset_bbox=False)
        impact_agg = xa.aggregate(hists_all[mod][['dimp_'+k for k in subset_params if k not in prob_timeframes+['hist']]],weightmap)
        impact_ds[mod] = impact_agg.to_dataset(loc_dim='county_idx')

        #-------------------------------
        # Get total change
        #-------------------------------
        # Sum total change in impact 
        if op == 'sum':
            for timeframe in [k for k in subset_params.keys() if k not in prob_timeframes+['hist']]:
                impact_ds[mod]['sum_dimp_'+timeframe] = ((impact_ds[mod]['dimp_'+timeframe]-1)*impact_ds[mod][var_name]*scale).sum('county_idx')
        elif op == 'mean':
            for timeframe in [k for k in subset_params.keys() if k not in prob_timeframes+['hist']]:
                impact_ds[mod]['sum_dimp_'+timeframe] = ((impact_ds[mod]['dimp_'+timeframe]-1)*impact_ds[mod][var_name]*scale).mean('county_idx')
    #-------------------------------
    # Return
    #-------------------------------
    return impact_ds


def d_yield(master_params,subset_params,
            plin_func = [{'bounds':[-np.inf,10+273.15],'m':0,'b':0},
                         {'bounds':[10+273.15,29+273.15],'m':(0.0057-0)/(29-10),'b':0-(10+273.15)*((0.0057-0)/(29-10))},
                         {'bounds':[29+273.15,39+273.15],'m':(-0.0625-0.0057)/(39-29),'b':-0.0625-(39+273.15)*((-0.0625-0.0057)/(39-29))},
                         {'bounds':[39+273.15,np.inf],'m':0,'b':-0.0625}],
            seas_range=[60,243], #Growing season (by day of year index)
            var_name='dyield',
            n_sub = 15, # How many points to simulate the sinusoid curve with
            process_by_lat_band=True, # Whether to process by latitude (True) or load all into memory at once (False)
            save_output = True,return_output=False):
    '''
    
    
    '''

    #-----------------------------------
    # Get list of models to process
    #-----------------------------------

    # Get list of all directory names in master_params['mod_data_dir']
    # that have files that satisfy master_params['mod_search_str']
    fns_base = [fn for fn in glob.glob(master_params['mod_data_dir']+'*/*.nc') if re.search(master_params['mod_search_str'],re.split('\/',fn)[-1])]
    models = [re.split('\/',fn)[-2] for fn in fns_base]
    models = np.unique(models)
    
    if return_output:
        yield_sums_all = dict()

    #-----------------------------------
    # Process by model 
    #-----------------------------------
    for mod in tqdm(models):
        #-------------------------------
        # Get output filename
        #-------------------------------
        # Get output filename 
        fns_tmp = [fn for fn in fns_base if re.search(mod+'\_',fn)]
        if len(fns_tmp)>1:
            warnings.warn('More than one file found for model '+mod+' and search string: '+master_params['mod_search_str']+'; the first one, '+fns_tmp[0]+', will be loaded.')
        fn = re.split('\/',fns_tmp[0])[-1]
        
        fn_out = (master_params['impact_data_dir']+master_params['obs_mod']+'/'+
                  var_name+'_'+mod+'_'+master_params['exp_name']+'_'+'-'.join(subset_params.keys())+'_plinsum_'+re.split('\.',re.split('\_',fn)[6])[0]+'.nc')

        #-------------------------------
        # Process
        #-------------------------------
        if master_params['overwrite'] or (not os.path.exists(fn_out)):
            #-------------------------------
            # Get filenames for files to load
            #-------------------------------
            fns = {'tasmax':dict(),
                   'tasmin':dict()}

            for v in fns:

                # Get filename for historical file (this is "rg_fn" in wrapper_proj())
                obs_fn = [fn for fn in [k for k in os.walk(master_params['obs_data_dir']+master_params['obs_mod'])][0][2] 
                                 if re.search(re.sub('tas',v,master_params['obs_search_str']),fn)]
                if len(obs_fn)>1:
                    warnings.warn('More than one file found for observation data ('+master_params['obs_mod']+')...') 
                obs_fn = obs_fn[0]
                fns[v]['hist'] = (master_params['proc_data_dir']+master_params['obs_mod']+'/'+re.sub(re.split('\_',obs_fn)[4],mod+'grid',obs_fn))

                # Get filenames for future files 
                for timeframe in [k for k in subset_params.keys() if k != 'hist']:
                    fns[v][timeframe] = (master_params['proc_data_dir']+master_params['obs_mod']+'/'+
                                       re.sub(re.split('\_',obs_fn)[3],master_params['exp_name'],
                                       re.sub('[0-9]{8}\-[0-9]{8}','-'.join(subset_params[timeframe]),
                                       re.sub(re.split('\_',obs_fn)[4],mod+'proj',obs_fn))))
                    
            #-----------------------------------
            # Calculate yield changes
            #-----------------------------------
            yield_sums = dict()
            for timeframe in subset_params:
                # Load data 
                ds_seas = xr.merge([xr.open_dataset(fns['tasmax'][timeframe]),
                          xr.open_dataset(fns['tasmin'][timeframe])])
    
                # Subset to season
                ds_seas = ds_seas.isel(time=((ds_seas.time.dt.dayofyear>=seas_range[0])&
                                             (ds_seas.time.dt.dayofyear<=seas_range[1])))

                ds_seas['yield_sums'] = ([k for k in ds_seas.tasmax.dims if k != 'time'],np.zeros_like(ds_seas.tasmax.isel(time=0).values)*np.nan)
                

                # Process by lat band to avoid memory issues (since generating large sin)
                if process_by_lat_band:
                    for lat_idx in np.arange(0,ds_seas.dims['lat']):
                    #for lat_idx in np.arange(0,2):
                        ds_seas_tmp = ds_seas.isel(lat=lat_idx).drop('yield_sums')
                        ds_seas_tmp = (0.5*(ds_seas_tmp.tasmax-ds_seas_tmp.tasmin)*np.sin(xr.DataArray(np.arange(0,n_sub)/n_sub,dims=['sub'])*2*np.pi) + 
                                    ds_seas_tmp.tasmin + 0.5*(ds_seas_tmp.tasmax-ds_seas_tmp.tasmin))

                        # Calculate sum over piecewise linear
                        yield_sums[timeframe] = [None]*len(plin_func)

                        for piece_idx in np.arange(0,len(plin_func)):
                            ds_tmp = ds_seas_tmp.where((ds_seas_tmp>plin_func[piece_idx]['bounds'][0]) & (ds_seas_tmp<=plin_func[piece_idx]['bounds'][1]))
                            yield_sums[timeframe][piece_idx] = (plin_func[piece_idx]['m']*ds_tmp+plin_func[piece_idx]['b']).sum(('time','sub'))

                        # Get the sum of all of these components of the piecewise linear, scale by 
                        # total length of days (since this is a by-day-at-a-certain-temperature effect),
                        # get exponential of it
                        #ds_seas['yield_sums'][{'lat':lat_idx}] = (np.exp(xr.concat(yield_sums[timeframe],dim='subset').sum('subset')/
                        #                                        ((ds_seas.dims['time']*n_sub)/len(pd.date_range(*subset_params['hist'],freq='Y')))))
                        try: 
                            ds_seas['yield_sums'][{'lat':lat_idx}] = (np.exp(xr.concat(yield_sums[timeframe],dim='subset').sum('subset')/
                                                                    (ds_seas.dims['time']*n_sub/(seas_range[1]-seas_range[0]+1))))
                        except:
                            breakpoint()
                    yield_sums[timeframe] = ds_seas['yield_sums']
                else:
                    ds_seas_tmp = (0.5*(ds_seas.tasmax-
                                        ds_seas.tasmin)*np.sin(xr.DataArray(np.arange(0,n_sub)/n_sub,dims=['sub'])*2*np.pi) + 
                                        ds_seas.tasmin + 0.5*(ds_seas.tasmax-ds_seas.tasmin))
                    ds_seas_tmp = ds_seas_tmp.load()
                    
                    # Calculate sum over piecewise linear
                    yield_sums[timeframe] = [None]*len(plin_func)

                    for piece_idx in tqdm(np.arange(0,len(plin_func))):
                        if (plin_func[piece_idx]['m'] == 0) and (plin_func[piece_idx]['b'] == 0 ):
                            # If the plin_func is just 0 over the whole domain of this piece, 
                            # just return a 0 array (saves time)
                            yield_sums[timeframe][piece_idx] = ds_seas_tmp.isel(sub=0,time=0).drop(['time','month'])*0
                        else:
                            ds_tmp = ds_seas_tmp.where((ds_seas_tmp>plin_func[piece_idx]['bounds'][0]) & 
                                                       (ds_seas_tmp<=plin_func[piece_idx]['bounds'][1]))
                            yield_sums[timeframe][piece_idx] = (plin_func[piece_idx]['m']*ds_tmp+
                                                                plin_func[piece_idx]['b']).sum(('time','sub'))
                    # Get the sum of all of these components of the piecewise linear, scale by 
                    # total length of days (since this is a by-day-at-a-certain-temperature effect),
                    # get exponential of it
                    try: 
                        yield_sums[timeframe] = (np.exp(xr.concat(yield_sums[timeframe],dim='subset').sum('subset')/
                                                        (ds_seas.dims['time']*n_sub/(seas_range[1]-seas_range[0]+1))))
                    except:
                        breakpoint()

                    
            del ds_tmp,ds_seas

            # Merge into single dataset
            yield_sums = xr.merge([v.to_dataset(name=k) for k,v in yield_sums.items()]).drop('height',errors='ignore')
            
            #-------------------------------
            # Export
            #-------------------------------
            # Export
            yield_sums.attrs['SOURCE'] = 'd_yield()'
            yield_sums.attrs['DESCRIPTION'] = 'exp of sum of piecewise linear effect each location, concatenated from the following timeframes: '+', '.join(subset_params.keys())
            yield_sums.attrs['growing_season'] = seas_range
            
            if save_output:
                if os.path.exists(fn_out):
                    os.remove(fn_out)
                    print(fn_out+' removed to allow overwrite!')
                yield_sums.to_netcdf(fn_out)
                print(fn_out+' saved!')

            if return_output:
                yield_sums_all[mod] = yield_sums
        else:
            yield_sums_all[mod] = xr.open_dataset(fn_out)
            print(fn_out+' already exists, skipped!')
            
    #-------------------------------
    # Export
    #-------------------------------
    if return_output:
        return yield_sums_all


def yield_dmg(master_params,subset_params,data_counties,yield_sums_all,scalevar_name='corn_prod'):
    
    impact_ds = dict()
    for mod in yield_sums_all:
        print('\nprocessing model '+mod+'!')
        #-------------------------------
        # Aggregate impact variable to polygons
        #-------------------------------
        if os.path.exists(master_params['aux_data_dir']+'wm_'+mod):
            weightmap = xa.read_wm(master_params['aux_data_dir']+'wm_'+mod)
        else:
            weightmap = xa.pixel_overlaps(yield_sums_all[mod],data_counties,subset_bbox=False)
        impact_agg = xa.aggregate(yield_sums_all[mod],weightmap)
        impact_ds[mod] = impact_agg.to_dataset(loc_dim='county_idx')

        for timeframe in subset_params:
            impact_ds[mod]['dimp_'+timeframe] = impact_ds[mod][timeframe]/impact_ds[mod]['hist']
            impact_ds[mod]['dimp_'+timeframe+'_tot'] = (impact_ds[mod]['dimp_'+timeframe]-1)*impact_ds[mod][scalevar_name]
            impact_ds[mod]['dimp_'+timeframe+'_avg'] = (impact_ds[mod]['dimp_'+timeframe]*impact_ds[mod][scalevar_name]).sum('county_idx')/np.sum(impact_ds[mod][scalevar_name])
            
    return impact_ds
            
    
def var_partitioning(master_params,params_all,return_output=True,
                     subset_to_common_runs=False,drop_models=None,
                     exp_vars = ['FIPS','NAME','tpop','gdp2015','corn_prod','gdp_pp'],
                     save_output=False,fn='../data/climate_proc/var_partitioning_all'):
    uncs_dss = dict()
    uncs_byc_dss = dict()
    mods_used = dict()
    for params in params_all:
        var_list = params['var_list'] # Variables for the agged calculation
        cvars = params['cvars'] # Variables for the county-level calculation
        
        drop_var_list = [k for k in var_list if k not in cvars]
        drop_cvars = [k for k in cvars if k not in var_list]
        #-----------------------------------------------------------------
        # Setup
        #-----------------------------------------------------------------

        # Get list of all files that correspond to the desired variable
        fns_base = [fn for fn in glob.glob(master_params['impact_data_dir']+
                                           master_params['obs_mod']+'/'+params['var']+'*.nc')]

        uncs = dict()
        uncs_byc = dict()

        #-----------------------------------------------------------------
        # Scenario uncertainty
        #-----------------------------------------------------------------

        # First, get filenames for models that aren't the large ensembles 
        # (all the others are the CMIP5 models)
        fns_su = [f for f in fns_base if re.search('^.*'+params['var']+'((?!LE).)*hist-begc.*\.nc$',f)]
        # Then, split up files by experiment
        fns_su_byexp = {exp:[f for f in fns_su if re.search(exp,f)] for exp in np.unique([re.split('\_',re.split('\/',f)[-1])[3] for f in fns_su])}

        # Load files by experiment
        dss = {exp:{re.split('\_',re.split('\/',fn)[-1])[2]:xr.open_dataset(fn) for fn in fns_su_byexp[exp]} for exp in fns_su_byexp}
        
        # Concatenate across models into one ds per experiment
        ds = {exp:xr.concat([v for k,v in dss[exp].items()],dim='model') for exp in dss}
        for exp in ds:
            ds[exp]['model'] = [k for k in dss[exp]]
        # Concatenate across experiments into one ds per experiment
        dse = xr.concat([v for k,v in ds.items()],dim='exp')
        dse['exp'] = [k for k in ds]
        # Lose the baggage
        dse = dse.drop([k for k in dse.keys() if k not in [params['wvar'],*var_list,*cvars]])

        # Fix up the wvar if needed (duplicates values across the model 
        # dimension otherwise). Assuming wvar is a county_idx only var. 
        if params['op_over_counties'] == 'wmean':
            # The -1 is a hack that makes this work; if the plot returns all 
            # nans in the scenario, this line is at fault. This hack is 
            # necessary because the weight variables get transferred oddly
            # between the concatenations above. They get copied through the
            # other dimensions, but only if the models have data for that 
            # experiment; otherwise it's all nans. The -1 model happens
            # to not be all nans. 
            dse[params['wvar']] = dse[params['wvar']].isel({k:-1 for k in dse[params['wvar']].dims if k not in ['county_idx']})

        # Only use models that have data for all experiments
        #breakpoint()
        dse = dse.isel(model=(~np.isnan(dse.mean(('county_idx'))[var_list[0]]).any('exp')))
        mods_used[params['var']] = dse.model

        # Get variance
        #breakpoint()
        if params['op_over_counties'] == 'sum':
            uncs['scenario'] = dse.drop([*drop_cvars]).sum('county_idx').mean('model').var('exp')
        elif params['op_over_counties'] == 'mean':
            uncs['scenario'] = dse.drop([*drop_cvars]).mean('county_idx').mean('model').var('exp')
        elif params['op_over_counties'] == 'wmean':
            # Dot product calculation instead
            #breakpoint()
            uncs['scenario'] = (dse.drop([*drop_cvars])[[var for var in dse if var not in [params['wvar']]]]*dse[params['wvar']]).sum('county_idx')/dse[params['wvar']].sum('county_idx')
            uncs['scenario'] = uncs['scenario'].where(uncs['scenario']!=0).mean('model').var('exp')
        # Get variance by county (no agg calculation across counties needed)
        uncs_byc['scenario'] = dse.drop(drop_var_list).mean('model').var('exp')

        #-----------------------------------------------------------------
        # Model uncertainty
        #-----------------------------------------------------------------
        # First, get filenames for LEs
        fns_mu = [f for f in fns_base if re.search('\-LE.*hist-begc',f)]

        # Load files 
        dss = {re.split('\_',re.split('\/',fn)[-1])[2]:xr.open_dataset(fn) for fn in fns_mu}
        # Concatenate across models into one ds per experiment
        ds = xr.concat([v for k,v in dss.items()],dim='model')
        ds['model'] = [k for k in dss]
        # Lose the baggage
        dse = ds.drop([k for k in ds.keys() if k not in [params['wvar'],*var_list,*cvars]])
        #breakpoint()
        # Subset to only a common denominator of number of runs
        # if desired
        if subset_to_common_runs: 
            # Choose a county id to use to subset runs (just any one that isn't a nan everywhere)
            county_idx_set = np.isnan(dse.isel(model=0,run=0)).any('county_idx')[var_list[0]].argmin()
            # Figure out minimum number of runs shared between all of them 
            max_run_idx = np.isnan(dse.isel(county_idx=0))[var_list[0]].any('model').argmax('run')
            # Subset dataset to just those X runs
            dse = dse.isel(run=slice(0,max_run_idx.values))

        # Remove model if desired 
        if drop_models != None:
            dse = dse.sel(model=[mod for mod in dse.model.values if mod not in drop_models])
        
        if params['op_over_counties'] == 'wmean':
            dse[params['wvar']] = dse[params['wvar']].isel({k:0 for k in dse[params['wvar']].dims if k not in ['county_idx']})   
        
        # Get variance
        if params['op_over_counties'] == 'sum':
            uncs['model'] = dse.drop([*drop_cvars]).sum('county_idx').mean('run').var('model')
        elif params['op_over_counties'] == 'mean':
            uncs['model'] = dse.drop([*drop_cvars]).mean('county_idx').mean('run').var('model')
        elif params['op_over_counties'] == 'wmean':
            # Dot product calculation instead
            uncs['model'] = (dse.drop([*drop_cvars])[[var for var in dse if var not in [params['wvar']]]]*dse[params['wvar']]).sum('county_idx')/dse[params['wvar']].sum('county_idx')
            uncs['model'] = uncs['model'].where(uncs['model']!=0).mean('run').var('model')
        # Get variance by county (no agg calculation across counties needed)
        uncs_byc['model'] = dse.drop(drop_var_list).mean('run').var('model')

        #-----------------------------------------------------------------
        # Internal uncertainty
        #-----------------------------------------------------------------
        # As above, using dse
        if params['op_over_counties'] == 'sum':
            uncs['internal'] = dse.drop([*drop_cvars]).sum('county_idx').var('run').mean('model')
        elif params['op_over_counties'] == 'mean':
            uncs['internal'] = dse.drop([*drop_cvars]).mean('county_idx').var('run').mean('model')
        elif params['op_over_counties'] == 'wmean':
            # Dot product calculation instead
            uncs['internal'] = (dse.drop([*drop_cvars])[[var for var in dse if var not in [params['wvar']]]]*dse[params['wvar']]).sum('county_idx')/dse[params['wvar']].sum('county_idx')
            uncs['internal'] = uncs['internal'].where(uncs['internal']!=0).var('run').mean('model')
        # Get variance by county (no agg calculation across counties needed)
        uncs_byc['internal'] = dse.drop(drop_var_list).var('run').mean('model')

        #-----------------------------------------------------------------
        # Get relative uncertainty
        #-----------------------------------------------------------------
        # Join uncertainties into one
        uncs_dss[params['var']] = xr.concat([v for k,v in uncs.items()],dim='source')
        uncs_dss[params['var']]['source'] = [k for k in uncs]
        # Make into DataArray that has a dimension for timeframe
        uncs_dss[params['var']] = xr.DataArray(data=np.vstack([uncs_dss[params['var']][v].values for v in var_list]),
                             coords={'time':['begc','midc','endc'],'source':uncs_dss[params['var']].source.values},
                             dims=['time','source'])
        
        # Join by county uncertainties into one
        uncs_byc_dss[params['var']] = xr.concat([v for k,v in uncs_byc.items()],dim='source')
        uncs_byc_dss[params['var']]['source'] = [k for k in uncs_byc]

        # Make into DataArray that has a dimension for timeframe
        #breakpoint()
        uncs_byc_dss[params['var']] = xr.DataArray(data=np.stack([uncs_byc_dss[params['var']][v].values for v in cvars],0),
                             coords={'time':['begc','midc','endc'],'source':uncs_byc_dss[params['var']].source.values,
                                     'county_idx':uncs_byc_dss[params['var']].county_idx.values},
                             dims=['time','source','county_idx'])

    # Concatenate variances of aggregated estimates into one final ds
    uncs_ds = xr.concat([v for k,v in uncs_dss.items()],dim='impact')
    uncs_ds['impact'] = [k for k in uncs_dss]
    uncs_ds = uncs_ds.to_dataset(name='variance')
    uncs_ds['models_used'] = xr.concat([v for k,v in mods_used.items()],dim='impact')
    
    # Concatenate variances of county level estimates into one final ds
    uncs_byc_ds = xr.concat([v for k,v in uncs_byc_dss.items()],dim='impact')
    uncs_byc_ds['impact'] = [k for k in uncs_byc_dss]
    uncs_byc_ds = uncs_byc_ds.to_dataset(name='variance')
    uncs_byc_ds['models_used'] = xr.concat([v for k,v in mods_used.items()],dim='impact')
    # Add back in county-level variables (this is a hack! just taking
    # advantage of the last 'ds' to have been loaded. But it should generally
    # work; since if different models / impact variables have different
    # county-level variable setups, we have a whole nother problem going on...)
    #breakpoint()
    uncs_byc_ds = xr.merge([uncs_byc_ds,ds[exp_vars].isel(model=0)])
    

    if save_output:
        uncs_ds.attrs['SOURCE'] = 'var_partitioning(), possibly via wrapper_var_partitioning()'
        uncs_ds.attrs['DESCRIPTION'] = 'variance by source; scenario calculated from CMIP5, internal and model calculated from LEs '
        if not fn.endswith('.nc'):
            fn = fn+'.nc'
        if os.path.exists(fn):
            print(fn+' removed to allow overwrite.')
            os.remove(fn)
        uncs_ds.to_netcdf(fn)
        print(fn+' saved!')
        
        uncs_byc_ds.attrs['SOURCE'] = 'var_partitioning(), possibly via wrapper_var_partitioning()'
        uncs_byc_ds.attrs['DESCRIPTION'] = 'variance by source and county; scenario calculated from CMIP5, internal and model calculated from LEs '
        if not fn.endswith('.nc'):
            fn = fn+'_bycounty.nc'
        else:
            fn = re.sub('\.nc','_bycounty.nc',fn)
        if os.path.exists(fn):
            print(fn+' removed to allow overwrite.')
            os.remove(fn)
        uncs_byc_ds.to_netcdf(fn)
        print(fn+' saved!')
        

    if return_output:
        return uncs_ds,uncs_byc_ds
    
    

def calc_dT(master_params,subset_params,
            mods = ['CESM1-CAM5-LE','CSIRO-Mk3-6-0-LE','CanESM2-LE','EC-EARTH-LE','GFDL-CM3-LE','GFDL-ESM2M-LE','MPI-ESM-LE'],
            save_output=False,fn=None,
            return_output=False):
    
    if fn is None:
        fn = master_params['impact_data_dir']+master_params['obs_mod']+'/dT_avg_allLEs_bycounty_begc-midc-endc.nc'

    # Make sure file ends with .nc
    if not fn.endswith('.nc'):
        fn = fn+'.nc'
        
    if (not os.path.exists(fn)) or master_params['overwrite']:
    
        #----------------------------------------------
        # Setup
        #----------------------------------------------
        # Load temperature data
        dss = dict()
        for exp in subset_params:
            dss[exp] = dict()
            # Some filename housekeeping
            if 'hist' in exp:
                exp_name = 'historical'
                mod_suffix = 'grid'
            else:
                exp_name = master_params['exp_name']
                mod_suffix = 'proj'
            # Load
            for mod in mods: 
                dss[exp][mod] = xr.open_dataset(master_params['proc_data_dir']+master_params['obs_mod']+
                                                '/tas_day_'+master_params['obs_mod']+'_'+
                                                exp_name+'_'+
                                                mod+mod_suffix+'_'+
                                                subset_params[exp][0]+'-'+subset_params[exp][1]+
                                                '_CUSA.nc')

        # Load county data
        dir_list = get_params()
        data_counties = gpd.read_file(dir_list['geo']+'UScounties_proc.shp')

        #----------------------------------------------
        # Calculate changes in temperature by month
        #----------------------------------------------
        d_dss = dict()
        for exp in [exp for exp in dss if exp not in ['hist']]:
            print('processing exp '+exp)
            d_dss[exp] = dict()
            for mod in tqdm(dss[exp]):
                d_dss[exp][mod] = (dss[exp][mod].groupby('time.month').mean() - 
                                 dss['hist'][mod].groupby('time.month').mean())

        #----------------------------------------------
        # Aggregate temperature changes to counties
        #----------------------------------------------
        d_counties = dict()
        for exp in [exp for exp in dss if exp not in ['hist']]:
            d_counties[exp] = dict()
        for mod in d_dss[exp]:
            wm = xa.pixel_overlaps(d_dss['begc'][mod],data_counties)
            for exp in [exp for exp in dss if exp not in ['hist']]:
                agg = xa.aggregate(d_dss[exp][mod],wm)
                d_counties[exp][mod] = agg.to_dataset()



        #----------------------------------------------
        # Save and output
        #----------------------------------------------
        # Concatenate into single dataset
        d_counties_out = xr.concat([xr.concat([v for k,v in d_counties[t].items()],dim='model') for t in d_counties],
                                  dim='timeframe')

        # Remove unnecesseary dimensions in non-tas variables
        for v in [v for v in d_counties_out if v !='tas']:
            d_counties_out[v] = d_counties_out[v].isel(model=0,timeframe=0)

        # Name the new dimension coordinates 
        d_counties_out['model'] = [k for k in d_counties[[k for k in d_counties][0]]]
        d_counties_out['timeframe'] = [k for k in d_counties]

        # Describe file for output
        d_counties_out.attrs['SOURCE'] = 'calc_dT()'
        d_counties_out.attrs['DESCRIPTION'] = ('Average change in monthly temperature by model, timeframe (vs hist), county. dT is calculated for each month as the difference between the average of that month in the future ('+
                                                ', '.join([k for k in subset_params if k != 'hist'])+') and the historical, at the pixel level and then aggregated.')
        for t in subset_params:
            d_counties_out.attrs['timeframe_'+t] = '-'.join(subset_params[t][0:2])


        if save_output:
            # Save if necessary
            if not os.path.exists(fn):
                d_counties_out.to_netcdf(fn)
                print(fn+' saved!')
            else:
                if master_params['overwrite']:
                    os.remove(fn)
                    print(fn+' removed to allow overwrite!')
                    d_counties_out.to_netcdf(fn)
                    print(fn+' saved!')
                else:
                    print(fn+ 'aready exists, skipped!')
                    
    else:
        # or, load if the file exists and it's not to be overwritten
        d_counties_out = xr.open_dataset(fn)
        
    
    if return_output:
        return d_counties_out
    
    
def calc_dtas(master_params,subset_params,data_counties,
             fn_prefix='dtasmean'):
    
    ''' Calculate changes in temperatures
    
    
    '''
    

    #-----------------------------------
    # Get list of models to process
    #-----------------------------------
    # Get list of all directory names in master_params['mod_data_dir']
    # that have files that satisfy master_params['mod_search_str']
    fns_base = [fn for fn in glob.glob(master_params['mod_data_dir']+'*/*.nc') if re.search(master_params['mod_search_str'],re.split('\/',fn)[-1])]
    models = [re.split('\/',fn)[-2] for fn in fns_base]
    models = np.unique(models)

    #-------------------------------
    # Get output filenames
    #-------------------------------
    # (and separate models that don't need to be processed
    # because of overwrite rules)
    output_fns = dict()
    toload_fns = dict()
    for mod in models:
        output_fns[mod] = (master_params['impact_data_dir']+master_params['obs_mod']+'/'+
                            fn_prefix+'_tot_'+mod+'_'+master_params['exp_name']+'_'+'-'.join(subset_params.keys())+'_bycounty_CUSA.nc')

        if os.path.exists(output_fns[mod]) and not master_params['overwrite']:
            print(output_fns[mod]+' already exists!')
            toload_fns[mod] = output_fns.pop(mod)

    #-------------------------------
    # Process
    #-------------------------------
    dmeans=dict()
    for mod in tqdm(output_fns):
        if master_params['overwrite'] or (not os.path.exists(output_fns[mod])):
            #-------------------------------
            # Load component files
            #-------------------------------
            fns = dict()

            # Get filename for historical file (this is "rg_fn" in wrapper_proj())
            obs_fn = [fn for fn in [k for k in os.walk(master_params['obs_data_dir']+master_params['obs_mod'])][0][2] 
                             if re.search(master_params['obs_search_str'],fn)]
            if len(obs_fn)>1:
                warnings.warn('More than one file found for observation data ('+master_params['obs_mod']+')...') 
            obs_fn = obs_fn[0]
            fns['hist'] = (master_params['proc_data_dir']+master_params['obs_mod']+'/'+re.sub(re.split('\_',obs_fn)[4],mod+'grid',obs_fn))

            # Get filenames for future files 
            for timeframe in [k for k in subset_params.keys() if k != 'hist']:
                fns[timeframe] = (master_params['proc_data_dir']+master_params['obs_mod']+'/'+
                                   re.sub(re.split('\_',obs_fn)[3],master_params['exp_name'],
                                   re.sub('[0-9]{8}\-[0-9]{8}','-'.join(subset_params[timeframe]),
                                   re.sub(re.split('\_',obs_fn)[4],mod+'proj',obs_fn))))

            #-------------------------------
            # Calculate means
            #-------------------------------
            means = dict()
            for timeframe in subset_params:
                # Load file
                try:
                    ds_tmp = xr.open_dataset(fns[timeframe])
                except FileNotFoundError:
                    print(fns[timeframe]+' does not exist! Skipping.')
                    continue
                # Calculate histogram
                means[timeframe] = ds_tmp.mean('time')
                del ds_tmp

            # Merge into one ds
            means = xr.merge([v.rename({'tas':'tas_'+k}) for k,v in means.items()])

            #-------------------------------
            # Get change in tas
            #-------------------------------
            prob_timeframes = []
            for timeframe in [k for k in subset_params.keys() if k != 'hist']:
                try:
                    # Get change per bin
                    means['dtas_'+timeframe] = means['tas_'+timeframe] - means['tas_hist']
                except KeyError: 
                    prob_timeframes = prob_timeframes + [timeframe]
                    #breakpoint()
                    print('Issue with model '+mod+', timeframe '+timeframe)

            # Just extract the dtas variables
            try:
                dmeans[mod] = means[['dtas_'+timeframe for timeframe in [k for k in subset_params.keys() if k != 'hist']]]
            except KeyError:
                print('Issue with model '+mod+', no file saved.')
                continue

            #-------------------------------
            # Aggregate dtas to polygons
            #-------------------------------
            if os.path.exists(master_params['aux_data_dir']+'wm_'+mod+'/'):
                weightmap=xa.read_wm(master_params['aux_data_dir']+'wm_'+mod)
            else:
                weightmap = xa.pixel_overlaps(dmeans[mod],data_counties,subset_bbox=True)
            impact_agg = xa.aggregate(dmeans[mod][['dtas_'+k for k in subset_params if k not in prob_timeframes+['hist']]],weightmap)
            dmeans[mod] = impact_agg.to_dataset(loc_dim='county_idx')

            #-------------------------------
            # Output
            #-------------------------------
            dmeans[mod].attrs['SOURCE'] = 'calc_dtas()'
            dmeans[mod].attrs['DESCRIPTION'] = 'change in mean temperature calculated by county, concatenated from the following timeframes: '+', '.join(subset_params.keys())

            if os.path.exists(output_fns[mod]):
                os.remove(output_fns[mod])
                print(output_fns[mod]+' removed to allow overwrite!')
            dmeans[mod].to_netcdf(output_fns[mod])
            print(output_fns[mod]+' saved!')

    for mod in toload_fns:
        dmeans[mod] = xr.open_dataset(toload_fns[mod])

    return dmeans


def calc_dtas_bymonth(master_params,subset_params,data_counties):
    #-----------------------------------
    # Get list of models to process
    #-----------------------------------
    # Get list of all directory names in master_params['mod_data_dir']
    # that have files that satisfy master_params['mod_search_str']
    fns_base = [fn for fn in glob.glob(master_params['mod_data_dir']+'*/*.nc') if 
                re.search(master_params['mod_search_str'],re.split('\/',fn)[-1])]
    models = [re.split('\/',fn)[-2] for fn in fns_base]
    models = np.unique(models)

    #-----------------------------------
    # Process by model
    #-----------------------------------
    for mod_idx in tqdm(np.arange(0,len(models))):
        mod = models[mod_idx]
        fn_in = fns_base[mod_idx]
        fn = (master_params['proc_data_dir']+master_params['obs_mod']+
              '/dtasmean_bymon_'+mod+'_'+master_params['exp_name']+'_hist-begc-midc-endc_bycounty_CUSA.nc')
        
        if master_params['overwrite'] or not os.path.exists(fn):
            ## Load and clean up base temeprature file
            ds = xr.open_dataset(fn_in)
            ds = ds.drop([v for v in ds if v not in ['tas','lat_bnds','lon_bnds']],errors='ignore')
            ds = xa.fix_ds(ds)
            # one of the files weirdly had a bad 'time' dimension in 
            # pre-existing lon bnds, which breaks pixel overlaps.
            if 'lon_bnds' in ds:
                if 'time' in ds['lon_bnds'].dims:
                    ds = ds.drop(['lon_bnds','lat_bnds'])

            # Create weightmap
            try:
                wm = xa.pixel_overlaps(ds,data_counties)
            except:
                print('issue with model '+mod)
                continue

            # Calculate change in monthly means 
            dc = dict()
            for k in [k for k in subset_params if k not in ['hist']]:
                try:
                    ds_tmp = (ds.sel(time=slice(*(subset_params[k]))).groupby('time.month').mean() - 
                              ds.sel(time=slice(*(subset_params['hist']))).groupby('time.month').mean())
                except ValueError:
                    # For 360-day calendars, switch out the end date
                    subset_params_360 = {k:[re.sub('31','30',t) for t in v] for k,v in subset_params.items()}
                    ds_tmp = (ds.sel(time=slice(*(subset_params_360[k]))).groupby('time.month').mean() - 
                              ds.sel(time=slice(*(subset_params_360['hist']))).groupby('time.month').mean())
                    
                dc[k] = xa.aggregate(ds_tmp.drop(['average_DT'],errors='ignore'),wm)
                dc[k] = dc[k].to_dataset()
                dc[k] = dc[k].rename({'tas':'dtas_'+k})

                dc[k]['dtasavg_'+k] = dc[k]['dtas_'+k].mean('month')

            dcs = {k:v for k,v in dc.items()}
            for k in [k for k in dcs][1:]:
                dcs[k] = dcs[k].drop([var for var in dcs[k] if 'tas' not in var])
            dcs = xr.merge([v for k,v in dcs.items()])

            dcs = dcs.rename({'poly_idx':'county_idx'})

            dcs.attrs['SOURCE'] = 'calc_dtas_bymonth()'
            dcs.attrs['DESCRIPTION'] = 'changes in T, aggregated to county level'
            dcs.attrs['model'] = mod

            if os.path.exists(fn):
                os.remove(fn)
                print(fn+' removed to allow overwrite.')
            dcs.to_netcdf(fn)
            print(fn+' saved!')
        else:
            print(fn+' exists, skipped!')

    