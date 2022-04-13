#---------------------- aux_funcs.py ----------------------
# This file contains auxiliary functions needed for 
# directory management, etc.



import pandas as pd
import geopandas as gpd
import numpy as np
import xarray as xr
import xagg as xa
import glob
import re
import os

def get_params():
    ''' Get parameters 
    
    Outputs general paths, stored in the dir_list.csv file. 
    
    Parameters:
    ----------------------
    (none)
    
    
    Returns:
    ----------------------
    dir_list : dict()
        a dictionary of directory names for file system 
        managing purposes: 
            - 'obs':   where raw historical files are stored, in 
                       in subdirectories by model/product name
            - 'mod':   where raw model files are stored, in 
                       subdirectories by model/product name
            - 'proc':  where processed/projected climate data
                       files are stored, in subdirectories by the
                       observational data product name
            - 'impact': where projected impact variables are stored
            - 'aux':   where some auxiliary files (mainly related
                       to the geographical aggregation process)
            - 'geo':   where shapefiles are stored
            - 'pop':   where population, GDP, crop yield data are 
                       stored
            - 'figs':  where figures are stored

   '''


    # Dir_list
    dir_list = pd.read_csv('dir_list.csv')
    dir_list = {d:dir_list.set_index('dir_name').loc[d,'dir_path'] for d in dir_list['dir_name']}

    # Return
    return dir_list


    
def generate_weightmaps(master_params,search_str='/tas_day_*rcp85*2010*CUSA.nc'):
    ''' Pre-generate weightmaps for spatial aggregation calculation
    
    Uses xagg.weightmap.to_file() to pre-generate the weightmaps, 
    saving processing time later
    
    Parameters
    ------------------
    - master_params : dict()
        the master_params dict from the master_run.ipynb
    
    '''
    # Get locations of files
    dir_list = get_params()
    
    # Load counties file
    data_counties = gpd.read_file(dir_list['geo']+'UScounties_proc.shp')

    # Get list of all directory names in master_params['mod_data_dir']
    # that have files that satisfy master_params['mod_search_str']
    fns = [fn for fn in glob.glob(master_params['proc_data_dir']+master_params['obs_mod']+
                                 search_str)]
    models = [re.split('pro',re.split('\_',re.split('\/',fn)[-1])[4])[0] for fn in fns]
    models = np.unique(models)

    # Generate by model
    for mod_idx in np.arange(0,len(models)):
        mod = models[mod_idx]
        fn = fns[mod_idx]

        output_fn = dir_list['aux']+'wm_'+mod
        if (not os.path.exists(output_fn)) or (master_params['overwrite']):

            # Load climate data, for lat/lon grid
            ds = xr.open_dataset(fn)

            # Calculate weightmap
            wm = xa.pixel_overlaps(ds,data_counties,subset_bbox=False)

            # Export
            wm.to_file(output_fn,overwrite=master_params['overwrite'])
            print(output_fn+'/* saved.')

        else:
            print(output_fn+'/* files exist, skipped.')
            
            
def check_dirs(master_params):
    dir_list = get_params()
    
    # Check to see if raw data directory exists
    fns = [fn for fn in glob.glob(master_params['mod_data_dir']+'*/*.nc') if 
           re.search(master_params['mod_search_str'],re.split('\/',fn)[-1])]
    if len(fns)>0:
        print('+ model directory: '+master_params['mod_data_dir']+' exists; contains data from '+str(len(fns))+' models and data products in total!')
    else:
        raise FileNotFoundError('- model directory '+master_params['mod_data_dir']+
                    ' either not found, or contains no data in subdirectories that match '+master_params['mod_search_str']+'.')
    
    # Check to see if obs data directory exists
    fns = [fn for fn in glob.glob(master_params['obs_data_dir']+master_params['obs_mod']+'/*.nc') if 
           re.search(master_params['obs_search_str'],re.split('\/',fn)[-1])]
    if len(fns)>0:
        print('+ observation directory: '+master_params['obs_data_dir']+master_params['obs_mod']+
              ' exists, contains at least one file that matches '+master_params['obs_search_str']+'.')
    else:
        raise FileNotFoundError('- observational data diretory '+master_params['obs_data_dir']+master_params['obs_mod']+
                    ' either not found, or contains no data in subdirectories that match '+master_params['obs_search_str']+'.')
    
    # Check to see if shapefile exists
    if os.path.exists(dir_list['geo']+'UScounties_proc.shp'):
        print('+ geo directory: county-level data found at '+dir_list['geo']+'UScounties_proc.shp')
    else:
        raise FileNotFoundError('- no county-level shapefile ('+dir_list['geo']+'UScounties_proc.shp'+') found. Was `create_data_counties.ipynb` run?')
    
    # Directories for output
    for d in ['aux','figs']:
        if not os.path.exists(dir_list[d]):
            os.mkdir(dir_list[d])
            print('+ '+d+' directory: '+dir_list[d]+' created!')
        else:
            print('+ '+d+' directory: '+dir_list[d]+' exists!')
            
    # Directories for output contingent on base observational data
    for d in ['proc','impact']:
        if not os.path.exists(dir_list[d]+master_params['obs_mod']+'/'):
            os.mkdir(dir_list[d]+master_params['obs_mod']+'/')
            print('+ '+d+' directory: '+dir_list[d]+master_params['obs_mod']+'/ created!')
        else:
            print('+ '+d+' directory: '+dir_list[d]+master_params['obs_mod']+'/ exists!')
    
    
    