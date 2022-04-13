#---------------------- wrapper.py ----------------------
# This file contains wrapper functions for many of the 
# processing, projecting, and plotting functions. These
# wrapper functions generally load data and output files/
# figures, while the actual processing occurs in subfunctions
# in proc_funcs.py, proj_funcs.py, plot_funcs.py, and 
# dist_funcs.py. 


import xarray as xr
import xesmf as xe
import xagg as xa
import numpy as np
import geopandas as gpd
import os
import re
import glob
import warnings
from tqdm.notebook import tqdm as tqdm

from proc_funcs import bin_data,bin_dmgf,bin_log_dmgf,d_yield,yield_dmg,var_partitioning,calc_dT
from proj_funcs import proj_climate
from plot_funcs import plot_hist_bauble_stacked_single,figure_dT,figure_panel_dTmeans,figure_dT_bymonth_maps
from dist_funcs import calc_t_kdes,calc_binned_kdes,calc_piecewise_hists


def wrapper_bin_dmgf(master_params,subset_params,
                     dmgf = xr.DataArray([0.69,0.59,0.64,0.36,0.27,0,0.12,0.23,0.33,0.94],dims=['bin'],coords=[np.arange(1,11)]),
                     bins = np.append(np.append(-np.inf,(np.arange(10,91,10)+459.67)*5/9),np.inf),
                     bin_name='_dng',
                     var_name='tpop', # What to use to get levels of the impact
                     fn_prefix='dmort', # 'dmort' or 'dgdp-pp'
                     description='heat-related mortality',
                     dmgf_type='level', # or log, to use bin_log_dmgf instead of bin_dmgf
                     op = 'mean',scale=1000, # used for bin_log_dmgf
                     data_counties = None):
    ''' This also loads the hists_all file from bin_data
    
    (and loads all the county data)
    '''
    
    print('\n---------------------------')
    print('Calculating bin-days')
    print('---------------------------\n')
    
    
    # Get bin-days
    hists_all = bin_data(master_params,subset_params,bins,bin_name,save_output=True,return_output=True)
    
    
    print('\n---------------------------')
    print('Calculating impact')
    print('---------------------------\n')
    
    # Subset hists to stuff where the full damage projection 
    # per county has already been calculated and saved
    output_fns = dict()
    toload_fns = dict()
    for mod in hists_all:
        output_fns[mod] = (master_params['impact_data_dir']+master_params['obs_mod']+'/'+
                           fn_prefix+'_tot_'+mod+'_'+master_params['exp_name']+'_'+'-'.join(subset_params.keys())+'_bycounty_CUSA.nc')
        if os.path.exists(output_fns[mod]) and not master_params['overwrite']:
            print(output_fns[mod]+' already exists!')
            toload_fns[mod] = output_fns.pop(mod)
    for mod in toload_fns:
        del hists_all[mod]

    # Calculate impact by county
    if dmgf_type == 'log':
        impact_ds = bin_log_dmgf(master_params,subset_params,hists_all,dmgf,data_counties,var_name = var_name,
                                 op=op,scale=scale)
    elif dmgf_type == 'level':
        impact_ds = bin_dmgf(master_params,subset_params,hists_all,dmgf,data_counties,var_name = var_name)

    # Output
    for mod in impact_ds: 
        impact_ds[mod].attrs['SOURCE'] = 'wrapper_bin_dmgf()'
        impact_ds[mod].attrs['DESCRIPTION'] = 'full damages calculated by county for '+description+', concatenated from the following timeframes: '+', '.join(subset_params.keys())

        output_fn = (master_params['impact_data_dir']+master_params['obs_mod']+'/'+fn_prefix+'_tot_'+mod+'-'.join(subset_params.keys())+'_bycounty_CUSA.nc')
        impact_ds[mod].to_netcdf(output_fns[mod])
        print(output_fns[mod]+' saved!')
        
    for mod in toload_fns:
        impact_ds[mod] = xr.open_dataset(toload_fns[mod])
        
    return impact_ds


def wrapper_proj(master_params,subset_params,save_output=True,return_output=False,exp_name = 'rcp85'):
    ''' Wrapper for delta-method projecting function, which loads, projects, and saves
    
    Parameters
    --------------------------
    master_params : dict
    
    
    subset_params : dict
    
    
    save_output : bool, default True
    
    return_output : bool, default False
    
    Returns
    --------------------------
    out_data : dict()
        if return_output is True, then a dictionary containing the 
        projected datasets is returned
    
    '''
    
    #-----------------------------------
    # Load observation data
    #-----------------------------------
    # Get observation file name
    obs_fn = [fn for fn in [k for k in os.walk(master_params['obs_data_dir']+master_params['obs_mod'])][0][2] 
                 if re.search(master_params['obs_search_str'],fn)]
    if len(obs_fn)>1:
        warnings.warn('More than one file found for observation data ('+master_params['obs_mod']+'); the first one, '+obs_fn[0]+', will be loaded.')
    obs_fn = obs_fn[0]    
    
    # Load file
    ds_ref = xr.open_dataset(master_params['obs_data_dir']+master_params['obs_mod']+'/'+obs_fn)
    
    #-----------------------------------
    # Get list of models to process
    #-----------------------------------
    # Get list of all directory names in master_params['mod_data_dir']
    # that have files that satisfy master_params['mod_search_str']
    fns = [fn for fn in glob.glob(master_params['mod_data_dir']+'*/*.nc') if re.search(master_params['mod_search_str'],re.split('\/',fn)[-1])]
    models = [re.split('\/',fn)[-2] for fn in fns]
    models = np.unique(models)
    
    #-----------------------------------
    # Process by model 
    #-----------------------------------
    if return_output:
        out_data = {mod:dict() for mod in models}
    for mod in tqdm(models): 
        # Load model data 
        ds_le = dict()
        fns_tmp = [fn for fn in fns if re.search(mod+'\_',fn)]
        if len(fns_tmp)>1:
            warnings.warn('More than one file found for model '+mod+' and search string: '+master_params['mod_search_str']+'; the first one, '+fns_tmp[0]+', will be loaded.')
        fns_tmp = fns_tmp[0]
        for timeframe in subset_params.keys():
            try:
                ds_le[timeframe] = (xr.open_dataset(fns_tmp).sel(time=slice(*subset_params[timeframe])))
            except ValueError:
                tf360 = subset_params[timeframe]
                tf360 = [re.sub('31','30',t) for t in tf360]
                ds_le[timeframe] = (xr.open_dataset(fns_tmp).sel(time=slice(*tf360)))
                
            ds_le[timeframe] = xa.fix_ds(ds_le[timeframe])
            if 'record' in ds_le[timeframe].dims:
                ds_le[timeframe] = ds_le[timeframe].rename({'record':'run'})
        
        
        # Get filename of regridded obs file, which is the input observed
        # filename, but with the 'run' element (element #5) replaced with
        # "[mod]proj"
        rg_fn = (master_params['proc_data_dir']+master_params['obs_mod']+'/'+re.sub(re.split('\_',obs_fn)[4],mod+'grid',obs_fn))
        
        # Either create this regridded file if it doesn't yet exist, 
        # or load it if it does
        if master_params['overwrite'] or (not os.path.exists(rg_fn)):
            # Regrid ERA to model grid
            regridder = xe.Regridder(ds_ref,ds_le[[k for k in subset_params][0]],'bilinear')
            # Regrid
            ds_ref_rg = regridder(ds_ref)

            # Replace 0s with nans (THIS SHOULD NOT BE NECESSARY)
            for var in ds_ref_rg:
                print('replaced 0s with NaNs after regridding for variable '+var+
                      ' since xesmf generates 0 out of the original bbox. '+
                      'If this variable shows data that often is exactly "0" (e.g. precip), then this may be problematic.')
                ds_ref_rg[var] = ds_ref_rg[var].where(ds_ref_rg[var].mean('time')!=0)
                
            # Subset to bbox (hopefully this excludes only _bnds / _bounds variables)
            if len([v for v in ds_ref_rg if 'nds' not in v])>1:
                print('more than one core variable found in reference file... subsetting to nans may delete some data, since only based on the nans in variable "'+[v for v in ds_ref_rg if 'nds' not in v][0]+'".')
            ds_ref_rg = ds_ref_rg.isel(lat=~np.isnan(ds_ref_rg.isel(time=0)[[v for v in ds_ref_rg if 'nds' not in v][0]]).all('lon').values)
            ds_ref_rg = ds_ref_rg.isel(lon=~np.isnan(ds_ref_rg.isel(time=0)[[v for v in ds_ref_rg if 'nds' not in v][0]]).all('lat').values)

            # Save regrid historical file
            ds_ref_rg.attrs['SOURCE'] = 'wrapper_proj()'
            ds_ref_rg.attrs['DESCRIPTION'] = 'Regridded '+master_params['obs_mod']+' to the '+mod+' grid.'
            ds_ref_rg.to_netcdf(rg_fn)
            print(rg_fn+' saved!')
        else:
            print(rg_fn+' already exists!')
            ds_ref_rg = xr.open_dataset(rg_fn)
    
        for timeframe in [k for k in subset_params.keys() if k != 'hist']:
            fn_out = (master_params['proc_data_dir']+master_params['obs_mod']+'/'+
                       re.sub(re.split('\_',obs_fn)[3],exp_name,
                       re.sub('[0-9]{8}\-[0-9]{8}','-'.join(subset_params[timeframe]),
                       re.sub(re.split('\_',obs_fn)[4],mod+'proj',obs_fn))))

            if master_params['overwrite'] or (not os.path.exists(fn_out)):
                try:
                    ds_proj = proj_climate(ds_ref_rg,ds_le['hist'],ds_le[timeframe],var=master_params['varname'])
                except ValueError:
                    breakpoint()
                    print('issue with model '+mod+', timeframe '+timeframe+'. skipping for now.')
                    continue
                        
                ds_proj.attrs['SOURCE'] = 'wrapper_proj()'

                ds_proj.attrs['DESCRIPTION'] = master_params['obs_mod']+' regridded to and delta-projected by '+mod+'.'
                
                # Subset to bbox (hopefully this excludes only _bnds / _bounds variables)
                if len([v for v in ds_proj if 'nds' not in v])>1:
                    print('more than one core variable found in projected file... subsetting to nans may delete some data, since only based on the nans in variable "'+[v for v in ds_proj if 'nds' not in v][0]+'".')
                #ds_proj = ds_proj.isel(lat=~np.isnan(ds_proj.isel(time=0,run=0)[[v for v in ds_proj if 'nds' not in v][0]]).all('lon').values)
                #ds_proj = ds_proj.isel(lon=~np.isnan(ds_proj.isel(time=0,run=0)[[v for v in ds_proj if 'nds' not in v][0]]).all('lat').values)
                ds_proj = ds_proj.isel(lat=~np.isnan(ds_proj.isel({k:0 for k,v in ds_proj.dims.items() if k not in ['lat','lon']})[[v for v in ds_proj if 'nds' not in v][0]]).all('lon').values)
                ds_proj = ds_proj.isel(lon=~np.isnan(ds_proj.isel({k:0 for k,v in ds_proj.dims.items() if k not in ['lat','lon']})[[v for v in ds_proj if 'nds' not in v][0]]).all('lat').values)
                
                if save_output:
                    if not os.path.exists(master_params['proc_data_dir']+master_params['obs_mod']+'/'):
                        os.mkdir(master_params['proc_data_dir']+master_params['obs_mod']+'/')
                    ds_proj.to_netcdf(fn_out)
                    print(fn_out+' saved!')
            else:
                print(fn_out+' already exists; skipped.')

                if return_output:
                    ds_proj = xr.open_dataset(fn_out)

            if return_output:
                out_data[mod][timeframe] = ds_proj
            
    if return_output:
        return out_data
    
    
def wrapper_dyield(master_params,subset_params,
                   data_counties,
                   plin_func = [{'bounds':[-np.inf,10+273.15],'m':0,'b':0},
                                 {'bounds':[10+273.15,29+273.15],'m':(0.0057-0)/(29-10),'b':0-(10+273.15)*((0.0057-0)/(29-10))},
                                 {'bounds':[29+273.15,39+273.15],'m':(-0.0625-0.0057)/(39-29),'b':-0.0625-(39+273.15)*((-0.0625-0.0057)/(39-29))},
                                 {'bounds':[39+273.15,np.inf],'m':0,'b':-0.0625}],
                   seas_range=[60,243],
                   var_name='dyield', #file output name
                   n_sub=15, # number of points to calculate sinusoid with
                   scalevar_name='corn_prod', # impact variable
                   process_by_lat_band=True): 
    
    print('\n---------------------------')
    print('Calculating damage function output')
    print('---------------------------\n')
    
    yield_sums_all = d_yield(master_params,subset_params,
                            plin_func,
                            seas_range,var_name='dyield',
                            n_sub = n_sub, # How many points to simulate the sinusoid curve with
                            save_output = True,return_output=True)
    
    print('\n---------------------------')
    print('Calculating yield change')
    print('---------------------------\n')
    impact_yield = yield_dmg(master_params,subset_params,data_counties,yield_sums_all,scalevar_name=scalevar_name)
    
    
    for mod in tqdm(impact_yield): 
        impact_yield[mod].attrs['SOURCE'] = 'wrapper_dyield()'
        impact_yield[mod].attrs['DESCRIPTION'] = 'full damages calculated by county for corn yields, concatenated from the following timeframes: '+', '.join(subset_params.keys())
        impact_yield[mod].attrs['growing_season'] = seas_range


        output_fn = (master_params['impact_data_dir']+master_params['obs_mod']+
                     '/dyield_tot_'+mod+'_'+master_params['exp_name']+'_'+'-'.join(subset_params.keys())+'_bycounty_CUSA.nc')
        if os.path.exists(output_fn):
            os.remove(output_fn)
            print(output_fn+' removed to allow overwrite!')
        impact_yield[mod].to_netcdf(output_fn)
        print(output_fn+' saved!')
    
    
    
def wrapper_var_partitioning(master_params,varlist,
                             save_output=False,fn='../data/climate_proc/var_partitioning_all',
                             return_output=False,other_kws = {}):
    
    warnings.filterwarnings('ignore')
    params_all = [{'var':'dmort','wvar':'tpop',
                    'var_list':['sum_dimp_begc','sum_dimp_midc','sum_dimp_endc'],'op_over_counties':'sum',
                    'cvars':['dimp_begc','dimp_midc','dimp_endc']},
                  {'var':'dgdp-pp','wvar':'tpop',
                    'var_list':['sum_dimp_begc','sum_dimp_midc','sum_dimp_endc'],'op_over_counties':'sum',
                    'cvars':['dimp_begc','dimp_midc','dimp_endc']},
                  {'var':'dyield_tot','wvar':'corn_yield',
                    'var_list':['dimp_begc_avg','dimp_midc_avg','dimp_endc_avg'],'op_over_counties':'sum',
                    'cvars':['dimp_begc_tot','dimp_midc_tot','dimp_endc_tot']},
                  {'var':'dtasmean','wvar':'tpop',
                    'var_list':['dtas_begc','dtas_midc','dtas_endc'],'op_over_counties':'wmean',
                    'cvars':['dtas_begc','dtas_midc','dtas_endc']}]
    
    params_all = [p for p in params_all if p['var'] in varlist]
    
    if return_output:
        uncs_ds,uncs_byc_ds = var_partitioning(master_params,params_all,save_output=save_output,fn=fn,return_output=return_output,**other_kws)
        return uncs_ds,uncs_byc_ds
    else:
        var_partitioning(master_params,params_all,save_output=save_output,fn=fn,return_output=return_output,**other_kws)
    

def wrapper_exposure_distributions(master_params,subset_params,data_counties):
    
    Xs = {'dng':np.arange(0,1,1/50),
      'dnh':np.arange(-0.0009,0.0007,(0.0016)/50),
      'snr':np.arange(-0.08,0.006,0.04/200)}

    dmgfs = {'dng':xr.DataArray([0.69,0.59,0.64,0.36,0.27,0,0.12,0.23,0.33,0.94],dims=['bin'],coords=[np.arange(1,11)]),
             'dnh':xr.DataArray([0.000234,0.000126,-0.000144,-0.000269,-0.000322,-0.000195,-0.000119,-0.000074,-0.000003,-0.000036,
                             0,-0.000111,-0.000311,-0.000294,-0.000585,-0.000646,-0.000757],
                               dims=['bin'],coords=[np.arange(1,18)]),
             'snr':[{'bounds':[-np.inf,10+273.15],'m':0,'b':0},
                    {'bounds':[10+273.15,29+273.15],'m':(0.0057-0)/(29-10),'b':0-(10+273.15)*((0.0057-0)/(29-10))},
                    {'bounds':[29+273.15,39+273.15],'m':(-0.0625-0.0057)/(39-29),'b':-0.0625-(39+273.15)*((-0.0625-0.0057)/(39-29))},
                    {'bounds':[39+273.15,np.inf],'m':0,'b':-0.0625}]}

    bins = {'dng':np.append(np.append(-np.inf,(np.arange(10,91,10)+459.67)*5/9),np.inf),
            'dnh':np.append(np.append(-np.inf,np.arange(-15,31,3)+273.15),np.inf),
            'snr':None}
    
    # Set up 
    var_params_all = [{'name':'mort',
                      'dmgf':dmgfs['dng'],
                      'bins':bins['dng'],
                      'X':Xs['dng'],
                      'bandwidth':0.1},
                      {'name':'gdp-pp',
                      'dmgf':dmgfs['dnh'],
                      'bins':bins['dnh'],
                      'X':Xs['dnh'],
                      'bandwidth':0.0001},
                      {'name':'yield',
                       'dmgf':dmgfs['snr'],
                       'bins':bins['snr'],
                       'X':Xs['snr']}]
    
    # Get list of all directory names in master_params['mod_data_dir']
    # that have files that satisfy master_params['mod_search_str']
    fns_base = [fn for fn in glob.glob(master_params['mod_data_dir']+'*/*.nc') if 
                re.search(master_params['mod_search_str'],re.split('\/',fn)[-1])]
    models = [re.split('\/',fn)[-2] for fn in fns_base]
    models = np.unique(models)
    
    for mod_idx in tqdm(np.arange(0,len(models))): 
        mod = models[mod_idx]
        print('Processing model '+mod+'\n----------------------\n')

        # Load temperature data
        dss = dict()
        hist_fn = [fn for fn in glob.glob(master_params['obs_data_dir']+'*/*.nc') if 
            re.search(master_params['obs_search_str'],re.split('\/',fn)[-1])][0]
        dss['hist'] = xr.open_dataset(hist_fn)
        fut_fns = [fn for fn in glob.glob(master_params['proc_data_dir']+master_params['obs_mod']+
                                          '/'+master_params['varname']+'_*'+master_params['exp_name']+'*.nc') if 
                    re.search(mod+'proj',fn)]
        for t in [t for t in subset_params if t != 'hist']:
            dss[t] = xr.open_dataset([fn for fn in fut_fns if '-'.join(subset_params[t]) in fn][0])

        # Temperature exposure kde
        calc_t_kdes(dss,mod,master_params,data_counties)

        # Now process outcome variable distributions
        for var_params in var_params_all:
            print('\nProcessing '+var_params['name']+', model '+mod)

            # Outcome variable exposure kdes
            if var_params['bins'] is not None:
                # Mortality or GDP/capita (through the binned function)
                calc_binned_kdes(dss,mod,master_params,
                                 var_params,data_counties)
            else:
                # Yield (through the piecewise linear function)
                calc_piecewise_hists(dss,mod,master_params,
                                     var_params,data_counties)
    
    
    
def wrapper_figure_s4(master_params,subset_params,save_fig=True,fn=None):
    # Get dT data
    d_counties = calc_dT(master_params,subset_params,return_output=True,save_output=True)
    
    # Plot figure
    figure_dT(d_counties,save_fig=save_fig,output_fn=fn)
    
def wrapper_figure_s5(master_params,subset_params,
                      vmin=0,vmax=16,nlevels=17,
                      tframes = ['hist','endc'],
                      save_fig=True,output_fn = '../figures/tas_seas_LEs_rcp85-hist_seasavgs_avgacrossruns_2070-2099_1980-2009'):
    #----------------------------------------------------
    # Load changes in temperature by season
    #----------------------------------------------------
    dTmean_fn = (master_params['aux_data_dir']+'tas_seas_LEs_'+master_params['exp_name']+'-hist_seasavgs_avgacrossruns_'+
                 '_'.join(['-'.join([t[0:4] for t in subset_params[tf]]) for tf in tframes[::-1]])+'.nc')
    if os.path.exists(dTmean_fn):
        dms = xr.open_dataset(dTmean_fn)
    else:
        print('generating dT data for figure...')
        # Load raw climate data
        print('loading raw data')
        dss = dict()
        # Get list of all directory names in master_params['mod_data_dir']
        # that have files that satisfy master_params['mod_search_str']
        fns_base = [fn for fn in [fn for fn in glob.glob(master_params['mod_data_dir']+'*/*.nc') 
                    if re.search(master_params['mod_search_str'],re.split('\/',fn)[-1])] if '-LE' in fn]
        models = [re.split('\/',fn)[-2] for fn in fns_base]
        models = np.unique(models)
        for mod in models:
            fn = [fn for fn in glob.glob(master_params['mod_data_dir']+mod+'/*.nc') 
                  if re.search(master_params['mod_search_str'],fn)][0]
            dss[mod] = xr.open_dataset(fn)
            if 'record' in dss[mod].dims:
                dss[mod] = dss[mod].rename({'record':'run'})
            dss[mod] = dss[mod].load() # Load it for quicker work below
                
        # Aggregate to seasonal means
        dmean_by_seas = dict()
        print('aggregating to seasonal means')
        for mod in tqdm(models):
            try:
                dmean_by_seas[mod] = (dss[mod].sel(time=slice(*subset_params[tframes[1]])).tas.groupby('time.season').mean() - 
                                      dss[mod].sel(time=slice(*subset_params[tframes[0]])).tas.groupby('time.season').mean())
            except ValueError: # To deal with 360-day calendars
                subset_params_tmp = {k:[re.sub('31','30',t) for t in v] for k,v in subset_params.items()}
                dmean_by_seas[mod] = (dss[mod].sel(time=slice(*subset_params_tmp[tframes[1]])).tas.groupby('time.season').mean() - 
                                      dss[mod].sel(time=slice(*subset_params_tmp[tframes[0]])).tas.groupby('time.season').mean())
        
        dm_rgrd = dict()
        for mod in models:
            regridder = xe.Regridder(dmean_by_seas[mod], dmean_by_seas[models[0]], 'bilinear')
            dm_rgrd[mod] = regridder(dmean_by_seas[mod])


        dms = xr.concat([v.drop('height',errors='ignore').mean('run') for k,v in dm_rgrd.items()],dim='model')
        dms['model'] = models
        
        dms = dms.to_dataset(name='tas')

        dms.attrs['SOURCE'] = 'wrapper_figure_s5()'
        dms.attrs['DESCRIPTION'] = 'seasonal average change in temperature, mean across runs, regridded to '+models[0]

        dms.to_netcdf(dTmean_fn)
        print(dTmean_fn+' saved!')
            
    #----------------------------------------------------
    # Create figure
    #----------------------------------------------------
    figure_panel_dTmeans(dms,vmin=vmin,vmax=vmax,nlevels=nlevels,
                         save_fig=save_fig,output_fn=output_fn)
    
def wrapper_dT_bymonth_maps(master_params,mod,data_counties,impact_var='dmort_tot',
                            titles=['Ensemble member with lowest mid-century mortality',
                                    'Ensemble member with highest mid-century mortality'],
                            save_fig=False,output_fn='../figures/figure_s7_'):
    
    #-------------------------------------------
    # Load data
    #-------------------------------------------
    # Load change in temperature
    ds_dT = xr.open_dataset(master_params['proc_data_dir']+master_params['obs_mod']+
                            '/dtasmean_bymon_'+mod+'_'+master_params['exp_name']+'_hist-begc-midc-endc_bycounty_CUSA.nc')
    
    # Load mortality data
    dimp = xr.open_dataset(master_params['impact_data_dir']+master_params['obs_mod']+
                        '/'+impact_var+'_'+mod+'_'+master_params['exp_name']+'_hist-begc-midc-endc_bycounty_CUSA.nc')
    
    #-------------------------------------------
    # Get max/min impact runs
    #-------------------------------------------
    agg_var = 'sum_dimp_midc'

    # Get idx of least mortality run and most mortality run
    idxs = [dimp[agg_var].argmin().values,dimp[agg_var].argmax().values]
    
    #-------------------------------------------
    # Create figure
    #-------------------------------------------
    for idx_idx in np.arange(0,len(idxs)):
        figure_dT_bymonth_maps(ds_dT,data_counties,idxs[idx_idx],
                               title=titles[idx_idx],save_fig=save_fig,
                               output_fn=output_fn+str(idx_idx))
        
    
