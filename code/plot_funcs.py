#---------------------- plot_funcs.py ----------------------
# This file contains the functions needed to create figures
# for the project. Not all functions are used by the code
# that creates the main text or supplementary figures. 

from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import cartopy
from cartopy import crs as ccrs
from cartopy import feature as cfeature
import numpy as np
import pandas as pd
import datetime as dt
import cmocean
import glob
import re
import os
import string
import xarray as xr
import warnings
import copy


# from https://www.oreilly.com/library/view/python-cookbook/0596001673/ch03s24.html
# (auxiliary, to allow for better subplot lettering for plots with many panels)
def int_to_roman(input):
    """ Convert an integer to a Roman numeral. """

    if not isinstance(input, type(1)):
        raise TypeError("expected integer, got %s" % type(input))
    if not 0 < input < 4000:
        raise ValueError("Argument must be between 1 and 3999")
    ints = (1000, 900,  500, 400, 100,  90, 50,  40, 10,  9,   5,  4,   1)
    nums = ('M',  'CM', 'D', 'CD','C', 'XC','L','XL','X','IX','V','IV','I')
    result = []
    for i in range(len(ints)):
        count = int(input / ints[i])
        result.append(nums[i] * count)
        input -= ints[i] * count
    return ''.join(result)


def plot_hist_bauble(dmorts,master_params,subset_params,plot_vars=['dmort_midc','dmort_endc'],
                     var_title='heat-related mortality',reg_name = 'CONUS',xlims=None,
                     binwidth=1000,save_fig=False,fn = '../figures/dmort_allLEs_baubleplot'):
    # was binwidth+1 instead of 2*binwidth
    plot_all_bins = np.arange((np.min([[dmorts[k][plot_vars[v]].min().values for k in dmorts] for v in np.arange(0,len(plot_vars))]) // binwidth)*binwidth,
                          (np.max([[dmorts[k][plot_vars[v]].max().values for k in dmorts] for v in np.arange(0,len(plot_vars))]) // binwidth)*binwidth+2*binwidth,binwidth)
    plot_bincounts = np.zeros((len(dmorts),len(plot_vars),len(plot_all_bins)-1))

    for mod_idx in np.arange(0,len(dmorts)):
        mod = [k for k in dmorts.keys()][mod_idx]
        for plot_idx in np.arange(0,len(plot_vars)):
            plot_bincounts[mod_idx,plot_idx,:] = np.histogram(dmorts[mod][plot_vars[plot_idx]],bins=plot_all_bins)[0]

    fig = plt.figure(figsize=(15,15))

    bauble_cols = [[k/255 for k in [153,213,148]],[k/255 for k in [252,141,89]]]

    ax = plt.subplot()
    for mod_idx in np.arange(0,len(dmorts)):
        mod = [k for k in dmorts.keys()][mod_idx]

        # This is the max of the previous mod + 1
        if mod_idx == 0:
            mod_offset = 0
        else:
            mod_offset = mod_offset + np.max(plot_bincounts[mod_idx-1,:,:])+2

        # Stick in vertical line at the mod offset
        ax.axhline(mod_offset,color='k')
    
        hdls = [None]*len(plot_vars)
        for plot_idx in np.arange(0,len(plot_vars)):
            for bin_idx in np.arange(0,len(plot_all_bins)-1):
                for marker_idx in np.arange(0,plot_bincounts[mod_idx,plot_idx,bin_idx]):
                    if plot_idx > 0:
                        scnd_offset = np.sum(plot_bincounts[mod_idx,0:plot_idx,bin_idx])
                    else:
                        scnd_offset = 0

                    hdls[plot_idx] = ax.add_patch(mpl.patches.Ellipse((plot_all_bins[bin_idx]+binwidth/2,mod_offset+marker_idx+0.5+scnd_offset),
                                                    width=0.8*binwidth,height=0.8,facecolor=bauble_cols[plot_idx]))

        # Add model text
        ax.text(plot_all_bins[-1],mod_offset + np.max(plot_bincounts[mod_idx,:,:])+1,mod,
                horizontalalignment='right',verticalalignment='top',fontsize=15)
    
    # Create legend
    ax.legend(hdls,['-'.join([t[0:4] for t in subset_params[k]]) for k in subset_params if k != 'hist'],
              loc='upper right',bbox_to_anchor=(1,
                                                0.95),
              fontsize = 'x-large')
                                                
    
    # Set axis limits
    #breakpoint()
    if xlims is None:
        ax.set_xlim((plot_all_bins[0]-binwidth,plot_all_bins[-1]+binwidth))
    else:
        ax.set_xlim(xlims)
    ax.set_ylim((0,mod_offset+np.max(plot_bincounts[-1,:,:])+2))
    # Set aspect so the run markers look roughly circular
    ax.set_aspect(binwidth)

    # Remove y axis ticks
    ax.set_yticks([])

    # Horizontal line at 0 
    ax.axvline(0,color='r')

    # Plot title
    ax.set_title(('Change in '+var_title+' vs. historical (1980-2009) period in '+reg_name+',\neach ball represents a projection of '+
                 master_params['obs_mod']+' by one run of each model'),
                 fontsize=20)
    ax.set_xlabel('Change in '+var_title+' in '+reg_name,fontsize=15)
    
    # Grid and ticks on both sides, for ease of reading
    plt.grid(True)
    ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True,labelsize=15)
    
    if save_fig:
        fig.patch.set_facecolor('white')
        plt.tight_layout()
        plt.savefig(fn+'.png')
        print(fn+'.png saved!')
        plt.savefig(fn+'.svg')
        print(fn+'.svg saved!')
        
def plot_hist_bauble_stacked_single(dmorts,master_params,subset_params,plot_vars=['dmort_midc','dmort_endc'],
                             plot_titles=['2040-2069','2070-2099'],
                     var_title='mortality (deaths/year)',reg_name = 'CONUS',xlims=None,
                     binwidth=1000,save_fig=False,fn = '../figures/dmort_allLEs-collapsed_baubleplot'):
    # was binwidth+1 instead of 2*binwidth
    plot_all_bins = np.arange((np.min([[dmorts[k][plot_vars[v]].min().values for k in dmorts] for v in np.arange(0,len(plot_vars))]) // binwidth)*binwidth,
                          (np.max([[dmorts[k][plot_vars[v]].max().values for k in dmorts] for v in np.arange(0,len(plot_vars))]) // binwidth)*binwidth+2*binwidth,binwidth)
    plot_bincounts = np.zeros((len(dmorts),len(plot_vars),len(plot_all_bins)-1))

    for mod_idx in np.arange(0,len(dmorts)):
        mod = [k for k in dmorts][mod_idx]
        for plot_idx in np.arange(0,len(plot_vars)):
            plot_bincounts[mod_idx,plot_idx,:] = np.histogram(dmorts[mod][plot_vars[plot_idx]],bins=plot_all_bins)[0]

    #fig = plt.figure(figsize=(12,7))
    fig = plt.figure(figsize=(10,5))

    # One for each LE 
    bauble_cols = [[228/255,26/255,28/255],
                    [55/255,126/255,184/255],
                    [77/255,175/255,74/255],
                    [152/255,78/255,163/255],
                    [255/255,127/255,0/255],
                    [255/255,255/255,51/255],
                    [166/255,86/255,40/255]]

    for plot_idx in np.arange(0,len(plot_vars)):
        ax = plt.subplot(1,len(plot_vars),plot_idx+1)
        
        hdls = [None]*len(dmorts)
        for bin_idx in np.arange(0,len(plot_all_bins)-1):
            for mod_idx in np.arange(0,len(dmorts)):
                mod = [k for k in dmorts.keys()][mod_idx]
                for marker_idx in np.arange(0,plot_bincounts[mod_idx,plot_idx,bin_idx]):
                    # This ensures that the baubles stack on top of those from the previous model 
                    if mod_idx > 0:
                        bauble_offset = np.sum(plot_bincounts[0:mod_idx,plot_idx,bin_idx])
                    else:
                        bauble_offset = 0

                    hdls[mod_idx] = ax.add_patch(mpl.patches.Ellipse((plot_all_bins[bin_idx]+binwidth/2,bauble_offset+marker_idx+0.5),
                                                    width=0.8*binwidth,height=0.8,facecolor=bauble_cols[mod_idx]))

        # Set axis limits
        if xlims is None:
            ax.set_xlim((plot_all_bins[0]-binwidth,plot_all_bins[-1]+binwidth))
        else:
            ax.set_xlim(xlims)
        ax.set_ylim((0,np.max(np.sum(plot_bincounts,0))+2))
        # Set aspect so the run markers look roughly circular
        ax.set_aspect(binwidth)

        # Remove y axis ticks
        ax.set_yticks([])

        # Horizontal line at 0 
        ax.axvline(0,color='r')

        # Plot title
        ax.set_title((plot_titles[plot_idx]),
                     fontsize=12)
        ax.set_xlabel('change in '+var_title,fontsize=10)

        # Grid and ticks on both sides, for ease of reading
        plt.grid(True)
        ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False,labelsize=10)
        
        # Add subplot lettering
        ax.annotate(string.ascii_lowercase[plot_idx]+'.',
                                     (0.01,0.98),
                                     verticalalignment='top',
                                     fontweight='bold',
                                     fontsize=16,
                                     xycoords='axes fraction')
        
    # Add legend
    fig.legend(hdls,[re.sub('\-LE','',k) for k in dmorts],#borderaxespad=0.5, 
               bbox_to_anchor=(0.5,0.01), title='large ensemble', ncol=4,
               loc="lower center")
    
    # Make plots scoot together a little
    plt.subplots_adjust(wspace = 0.05)

    if save_fig:
        fig.patch.set_facecolor('white')
        #plt.tight_layout()
        plt.savefig(fn+'.png',dpi=450)
        print(fn+'.png saved!')
        plt.savefig(fn+'.svg')
        print(fn+'.svg saved!')
        plt.savefig(fn+'.pdf')
        print(fn+'.pdf saved!')
        
        
def plot_hist_bauble_stacked(dmorts,master_params,subset_params,plot_vars=['dmort_midc','dmort_endc'],
                             plot_titles=['2040-2069','2070-2099'],
                             var_title='heat-related mortality',reg_name = 'CONUS',
                             xlims=None,ylims=None,
                             binwidth=1000,save_fig=False,fn = '../figures/dmort_allLEs-collapsed_baubleplot',
                             fig=None,axs=None,add_legend=True):
    # was binwidth+1 instead of 2*binwidth
    plot_all_bins = np.arange((np.min([[dmorts[k][plot_vars[v]].min().values for k in dmorts] for v in np.arange(0,len(plot_vars))]) // binwidth)*binwidth,
                          (np.max([[dmorts[k][plot_vars[v]].max().values for k in dmorts] for v in np.arange(0,len(plot_vars))]) // binwidth)*binwidth+2*binwidth,binwidth)
    plot_bincounts = np.zeros((len(dmorts),len(plot_vars),len(plot_all_bins)-1))

    for mod_idx in np.arange(0,len(dmorts)):
        mod = [k for k in dmorts][mod_idx]
        for plot_idx in np.arange(0,len(plot_vars)):
            plot_bincounts[mod_idx,plot_idx,:] = np.histogram(dmorts[mod][plot_vars[plot_idx]],bins=plot_all_bins)[0]

    # Make 
    if fig is None:
        fig = plt.figure(figsize=(12,7))
    if axs is None:
        axs = plt.subplots(1,len(plot_vars))

    # One for each LE 
    bauble_cols = [[228/255,26/255,28/255],
                    [55/255,126/255,184/255],
                    [77/255,175/255,74/255],
                    [152/255,78/255,163/255],
                    [255/255,127/255,0/255],
                    [255/255,255/255,51/255],
                    [166/255,86/255,40/255]]

    for plot_idx in np.arange(0,len(plot_vars)):
        #ax = plt.subplot(1,len(plot_vars),plot_idx+1)
        
        hdls = [None]*len(dmorts)
        for bin_idx in np.arange(0,len(plot_all_bins)-1):
            for mod_idx in np.arange(0,len(dmorts)):
                mod = [k for k in dmorts.keys()][mod_idx]
                for marker_idx in np.arange(0,plot_bincounts[mod_idx,plot_idx,bin_idx]):
                    # This ensures that the baubles stack on top of those from the previous model 
                    if mod_idx > 0:
                        bauble_offset = np.sum(plot_bincounts[0:mod_idx,plot_idx,bin_idx])
                    else:
                        bauble_offset = 0

                    hdls[mod_idx] = axs[plot_idx].add_patch(mpl.patches.Ellipse((plot_all_bins[bin_idx]+binwidth/2,bauble_offset+marker_idx+0.5),
                                                    width=0.8*binwidth,height=0.8,facecolor=bauble_cols[mod_idx]))

        # Set axis limits
        if xlims is None:
            axs[plot_idx].set_xlim((plot_all_bins[0]-binwidth,plot_all_bins[-1]+binwidth))
        else:
            axs[plot_idx].set_xlim(xlims)
        if ylims is None:
            axs[plot_idx].set_ylim((0,np.max(np.sum(plot_bincounts,0))+2))
        else:
            axs[plot_idx].set_ylim(ylims)
        # Set aspect so the run markers look roughly circular
        axs[plot_idx].set_aspect(binwidth)

        # Remove y axis ticks
        axs[plot_idx].set_yticks([])

        # Horizontal line at 0 
        axs[plot_idx].axvline(0,color='r')

        # Plot title
        axs[plot_idx].set_title((plot_titles[plot_idx]),
                     fontsize=12)
        axs[plot_idx].set_xlabel('change in '+var_title,fontsize=10)

        # Grid and ticks on both sides, for ease of reading
        axs[plot_idx].grid(True)
        axs[plot_idx].tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False,labelsize=10)
    
    if add_legend:
        plt.legend(hdls,[k for k in dmorts],
               borderaxespad=0.5, 
               bbox_to_anchor=(1,0.5), title='model', 
               loc="center left")

    if save_fig:
        fig.patch.set_facecolor('white')
        plt.tight_layout()
        plt.savefig(fn+'.png')
        print(fn+'.png saved!')
        plt.savefig(fn+'.svg')
        print(fn+'.svg saved!')
        
    return hdls

def figure_part_uncert(master_params,var,title,
                       fn_load = '../data/climate_proc/var_partitioning_all.nc',
                       sources = ['model','scenario','internal'],
                       colors = ['tab:blue','tab:green','tab:orange'],
                       times = ['2010-2039','2040-2069','2070-2099'],
                       save_fig=False,output_fn='../figures/unc-part_figure',
                       add_legend=True,fig=None,ax=None,
                       op_over_counties = 'sum',wvar = None,
                       bold_title=False,
                       normalize=True):
    
    #-----------------------------------------------------------------
    # Setup
    #-----------------------------------------------------------------

    # Load relevant file
    uncs_ds = xr.open_dataset(fn_load)
    
    # Subset to desired variable
    uncs_ds = uncs_ds.sel(impact=var).variance
    
    if normalize:
        uncs_ds = uncs_ds/uncs_ds.sum('source')
    
    #-----------------------------------------------------------------
    # Figure
    #-----------------------------------------------------------------
    
    # Make 
    if fig is None:
        fig = plt.figure(facecolor='white')
    
    if ax is None:
        ax = plt.subplot()

    # Sort into right order for plotting / colors
    uncs_ds = uncs_ds.sel(source=sources)
    bars = [None]*len(sources)
    for src_idx in np.arange(0,len(sources)):
        bars[src_idx] = ax.bar(times,uncs_ds.isel(source=src_idx).values,
                                width=0.5,bottom = [uncs_ds.isel(source=slice(0,src_idx)).sum('source').values if src_idx>0 else 0][0],
                               label=uncs_ds.source.isel(source=src_idx).values,
                               color=colors[src_idx])
    
    
    if normalize: 
        ax.set_ylabel('fractional contribution to total uncertainty')
    else:
        ax.set_ylabel('contribution to total uncertainty')
    
    if bold_title:
        ax.set_title(title,fontweight='bold')
    else:
        ax.set_title(title)
    
    if add_legend:
        plt.legend(borderaxespad=0.5, 
                   bbox_to_anchor=(1,0.7), title='source', 
                   loc="upper left")
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(output_fn+'.png')
        print(output_fn+'.png saved!')
        plt.savefig(output_fn+'.svg')
        print(output_fn+'.svg saved!')
    
    return ax,bars
        
        
def figure_part_uncert_old(master_params,var,title,
                       sources = ['model','scenario','internal'],
                      colors = ['tab:blue','tab:green','tab:orange'],
                       times = ['2010-2039','2040-2069','2070-2099'],
                       var_list = ['sum_dimp_begc','sum_dimp_midc','sum_dimp_endc'],
                       save_fig=False,output_fn='../figures/unc-part_figure',
                       add_legend=True,fig=None,ax=None,
                       op_over_counties = 'sum',wvar = None,
                       normalize=True,
                       bold_title=False):
    
    #-----------------------------------------------------------------
    # Setup
    #-----------------------------------------------------------------

    # Get list of all files that correspond to the desired variable
    fns_base = [fn for fn in glob.glob(master_params['proc_data_dir']+'ERA-INTERIM/'+var+'*.nc')]

    uncs = dict()

    #-----------------------------------------------------------------
    # Scenario uncertainty
    #-----------------------------------------------------------------

    # First, get filenames for models that aren't the large ensembles 
    # (all the others are the CMIP5 models)
    fns_su = [f for f in fns_base if re.search('^.*'+var+'((?!LE).)*hist-begc.*\.nc$',f)]
    # Then, split up files by experiment
    fns_su_byexp = {exp:[f for f in fns_su if re.search(exp,f)] for exp in np.unique([re.split('\_',re.split('\/',f)[-1])[3] for f in fns_su])}

    # Load files by experiment
    dss = {exp:{re.split('\_',re.split('\/',fn)[-1])[2]:xr.open_dataset(fn) for fn in fns_su_byexp[exp]} for exp in fns_su_byexp}
    # Get rid of rcp85 CanESM2 because it's causing an issue
    print('getting rid of rcp85 CanESM2,GFDL-CM3,GFDL-ESM2M,CSIRO-Mk3-6-0')
    try:
        del dss['rcp85']['CanESM2']
    except:
        print()
    try:
        del dss['rcp85']['CSIRO-Mk3-6-0']
    except:
        print()
    del dss['rcp85']['GFDL-CM3'],dss['rcp85']['GFDL-ESM2M'] # for some reason, has the runs of GFDL-CM3-LE??

    # Concatenate across models into one ds per experiment
    ds = {exp:xr.concat([v for k,v in dss[exp].items()],dim='model') for exp in dss}
    for exp in ds:
        ds[exp]['model'] = [k for k in dss[exp]]
    # Concatenate across experiments into one ds per experiment
    dse = xr.concat([v for k,v in ds.items()],dim='exp')
    dse['exp'] = [k for k in ds]
    # Lose the baggage
    dse = dse.drop([k for k in dse.keys() if k not in [wvar,*var_list]])
    
    # Fix up the wvar if needed (duplicates values across the model 
    # dimension otherwise). Assuming wvar is a county_idx only var. 
    if op_over_counties == 'wmean':
        # The -1 is a hack that makes this work; if the plot returns all 
        # nans in the scenario, this line is at fault. This hack is n
        # necessary because the weight variables get transferred oddly
        # between the concatenations above. They get copied through the
        # other dimensions, but only if the models have data for that 
        # experiment; otherwise it's all nans. The -1 model happens
        # to not be all nans. 
        dse[wvar] = dse[wvar].isel({k:-1 for k in dse[wvar].dims if k not in ['county_idx']})

    # Only use models that have data for all experiments
    #breakpoint()
    dse = dse.isel(model=(~np.isnan(dse.mean(('county_idx'))[var_list[0]]).any('exp')))
    
    # Get variance
    if op_over_counties == 'sum':
        uncs['scenario'] = dse.sum('county_idx').mean('model').var('exp')
    elif op_over_counties == 'mean':
        uncs['scenario'] = dse.mean('county_idx').mean('model').var('exp')
    elif op_over_counties == 'wmean':
        # Dot product calculation instead
        #breakpoint()
        uncs['scenario'] = (dse[[var for var in dse if var not in [wvar]]]*dse[wvar]).sum('county_idx')/dse[wvar].sum('county_idx')
        uncs['scenario'] = uncs['scenario'].where(uncs['scenario']!=0).mean('model').var('exp')

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
    dse = ds.drop([k for k in ds.keys() if k not in [wvar,*var_list]])
    
    if op_over_counties == 'wmean':
        dse[wvar] = dse[wvar].isel({k:0 for k in dse[wvar].dims if k not in ['county_idx']})

    # Get variance
    if op_over_counties == 'sum':
        uncs['model'] = dse.sum('county_idx').mean('run').var('model')
    elif op_over_counties == 'mean':
        uncs['model'] = dse.mean('county_idx').mean('run').var('model')
    elif op_over_counties == 'wmean':
        # Dot product calculation instead
        uncs['model'] = (dse[[var for var in dse if var not in [wvar]]]*dse[wvar]).sum('county_idx')/dse[wvar].sum('county_idx')
        uncs['model'] = uncs['model'].where(uncs['model']!=0).mean('run').var('model')

    #-----------------------------------------------------------------
    # Internal uncertainty
    #-----------------------------------------------------------------
    # As above, using dse
    if op_over_counties == 'sum':
        uncs['internal'] = dse.sum('county_idx').var('run').mean('model')
    elif op_over_counties == 'mean':
        uncs['internal'] = dse.mean('county_idx').var('run').mean('model')
    elif op_over_counties == 'wmean':
        # Dot product calculation instead
        uncs['internal'] = (dse[[var for var in dse if var not in [wvar]]]*dse[wvar]).sum('county_idx')/dse[wvar].sum('county_idx')
        uncs['internal'] = uncs['internal'].where(uncs['internal']!=0).var('run').mean('model')
        
    #-----------------------------------------------------------------
    # Get relative uncertainty
    #-----------------------------------------------------------------
    # Join uncertainties into one
    uncs_ds = xr.concat([v for k,v in uncs.items()],dim='source')
    uncs_ds['source'] = [k for k in uncs]
    # Make into DataArray that has a dimension for timeframe
    uncs_ds = xr.DataArray(data=np.vstack([uncs_ds[var_list[0]].values,uncs_ds[var_list[1]].values,uncs_ds[var_list[2]].values]),
                         coords={'time':['begc','midc','endc'],'source':uncs_ds.source.values},
                         dims=['time','source'])
    # Get in percent
    if normalize:
        uncs_ds = uncs_ds/uncs_ds.sum('source')
    
    #-----------------------------------------------------------------
    # Figure
    #-----------------------------------------------------------------
    
    # Make 
    if fig is None:
        fig = plt.figure(facecolor='white')
    
    if ax is None:
        ax = plt.subplot()

    # Sort into right order for plotting / colors
    uncs_ds = uncs_ds.sel(source=sources)
    bars = [None]*len(sources)
    for src_idx in np.arange(0,len(sources)):
        bars[src_idx] = ax.bar(times,uncs_ds.isel(source=src_idx).values,
                                width=0.5,bottom = [uncs_ds.isel(source=slice(0,src_idx)).sum('source').values if src_idx>0 else 0][0],
                               label=uncs_ds.source.isel(source=src_idx).values,
                               color=colors[src_idx])
    
    if normalize: 
        ax.set_ylabel('fractional contribution to total uncertainty')
    else:
        ax.set_ylabel('contribution to total uncertainty')
    
    if bold_title:
        ax.set_title(title,fontweight='bold')
    else:
        ax.set_title(title)
    
    if add_legend:
        plt.legend(borderaxespad=0.5, 
                   bbox_to_anchor=(1,0.7), title='source', 
                   loc="upper left")

    plt.tight_layout()
    
    if save_fig:
        plt.savefig(output_fn+'.png')
        print(output_fn+'.png saved!')
        plt.savefig(output_fn+'.svg')
        print(output_fn+'.svg saved!')
    
    return ax,bars


def figure_dT(d_counties,envelope='minmax',main='mean',
                wvars = ['tpop','corn_prod'],
                exps = ['begc','midc','endc'],
               seas_range = [60,243],
                exp_colors = {'hist':'tab:blue',
                              'begc':'tab:green',
                              'midc':'tab:orange',
                              'endc':'tab:red'},
                exp_cicolors = {'hist':'lightblue',
                                'begc':'lightgreen',
                                'midc':'bisque',
                                'endc':'lightpink'},
                exp_names = {'begc':'2010-2039',
                             'midc':'2040-2069',
                             'endc':'2070-2099'},
                w_names = {'tpop':'population',
                           'corn_prod':'corn yield'},
              save_fig=False,output_fn=None):
    ''' Figure showing changes in monthly temperature in LEs    
    
    Parameters
    ---------------------
    d_counties : xr.Dataset
        the output to calc_dT()
        
    envelope : str (def: 'minmax')
        'minmax': envelope shows minimum and maximum run
        '1std': envelope shows +/- 1 standard deviation
        '2std': envelope shows +/- 2 standard deviations
        
    main : str (def: 'mean')
        'mean': main line shows mean between runs
        'median': main line shows median between runs
        
    wvars : list (def: ['tpop','corn_prod'])
        the weighting of each row, but a variable in d_counties
        
    exps : list (def: ['begc',',midc','endc'])
        which timeframes to plot
        
    save_fig : bool (def: False)
        if True, saves figure (as .png, .svg, .pdf)
        
    output_fn : str (def: None, see below)
        if not specified, figure is saved as 
        '../figures/dT_weighted_bymod_'+'-'.join(exps)+'_min-max-shading'
    
    '''


    corn_season = [pd.date_range('2001-01-01','2001-01-01')+dt.timedelta(days=seas_range[0]-1),
                   pd.date_range('2001-01-01','2001-01-01')+dt.timedelta(days=seas_range[1]-1)]

    fig,axs = plt.subplots(len(wvars),d_counties.dims['model'],figsize=(15,5))

    for mod_idx in np.arange(0,d_counties.dims['model']):

        for w_idx in np.arange(0,len(wvars)):
            mod = d_counties.isel(model=mod_idx).model.values

            #Xs = np.arange(1,13)
            Xs = pd.date_range('2001-01-01','2001-12-31',freq='MS')

            for exp in exps:
                plot_data = (d_counties.sel(timeframe=exp,model=mod).
                             tas.dot(d_counties.sel(timeframe=exp,model=mod)[wvars[w_idx]].fillna(0))/
                                     d_counties.sel(timeframe=exp,model=mod)[wvars[w_idx]].fillna(0).sum())

                if 'run' not in plot_data.dims:
                    line_data = plot_data
                else:
                    if main == 'mean':
                        line_data = plot_data.mean('run')
                    elif main == 'median':
                        line_data = plot_data.median('run')
                    else:
                        raise KeyError(main + ' is not a supported operation for the main line. Choosen median or mean.')

                    # Plot envelope
                    if envelope == '1std':
                        env_data = [plot_data.mean('run')-plot_data.std('run'),
                                    plot_data.mean('run')+plot_data.std('run')]
                    elif envelope == '2std':
                        env_data = [plot_data.mean('run')-2*plot_data.std('run'),
                                    plot_data.mean('run')+2*plot_data.std('run')]
                    elif envelope == 'minmax':
                        env_data = [plot_data.min('run'),
                                    plot_data.max('run')]
                    else:
                        raise KeyError(envelope + ' is not a supported operation for the enevelope. Choose 1std, 2std, or minmax.')

                    axs[w_idx,mod_idx].fill_between(Xs,
                                                    env_data[0],
                                                    env_data[1],
                                                    color=[exp_cicolors[exp]],alpha=0.5)

                # Plot main line
                axs[w_idx,mod_idx].plot(Xs,line_data,color=exp_colors[exp],label=exp_names[exp])


            axs[w_idx,mod_idx].set_ylim(-1,10)
            axs[w_idx,mod_idx].axhline(0,color='k',linestyle='--')

            if mod_idx > 0:
                axs[w_idx,mod_idx].tick_params(axis='y',which='both',left=False,labelleft=False)
            if mod_idx == (d_counties.dims['model']-1):
                axs[w_idx,mod_idx].tick_params(axis='y',which='both',right=True,labelright=True)
            if mod_idx == 0:
                axs[w_idx,mod_idx].set_ylabel('weighted by \n'+w_names[wvars[w_idx]]+'\n'+r'$\Delta T\ [K]$',fontsize=12)
            if w_idx < (len(wvars)-1):
                axs[w_idx,mod_idx].tick_params(axis='x',which='both',bottom=False,labelbottom=False)
            else:
                axs[w_idx,mod_idx].get_xaxis().set_major_formatter(mdates.DateFormatter('%b'))
                axs[w_idx,mod_idx].tick_params(axis='x', labelrotation = 45)

            if wvars[w_idx] == 'corn_prod':
                axs[w_idx,mod_idx].axvline(corn_season[0],color='k',linestyle=':')
                axs[w_idx,mod_idx].axvline(corn_season[1],color='k',linestyle=':')

            if w_idx == 0:
                axs[w_idx,mod_idx].set_title(re.sub('\-LE','',str(mod)),fontsize=12)


            axs[w_idx,mod_idx].grid(True)

            # Add subplot lettering
            axs[w_idx,mod_idx].annotate(string.ascii_lowercase[w_idx*d_counties.dims['model']+mod_idx]+'.',
                                         (0.01,0.98),
                                         verticalalignment='top',
                                         fontweight='bold',
                                         fontsize=16,
                                         xycoords='axes fraction')


    # Add arrow explaining the corn production season
    axs[1,0].text(corn_season[0] + (corn_season[1]-corn_season[0])/2,
                    8.25,
                    'growing\nseason',horizontalalignment='center')
    axs[1,0].annotate('',
                    (corn_season[0],7.75),
                    (corn_season[1],7.75),
                    ha="right", va="center",
                    arrowprops={'arrowstyle':'<->'})     


    # Add legend        
    fig.subplots_adjust(right=0.85,hspace=0.1)
    plt.legend(borderaxespad=2,bbox_to_anchor=(1,1), title='timeframe', loc="lower left")

    if save_fig:
        #plt.tight_layout()
        if output_fn is None:
            output_fn = '../figures/dT_weighted_bymod_'+'-'.join(exps)+'_min-max-shading'
        plt.savefig(output_fn+'.png',dpi=450)
        print(output_fn+'.png saved!')
        plt.savefig(output_fn+'.pdf')
        print(output_fn+'.pdf saved!')
        
        
def fig_geo_partition(var,data_counties,master_params,
                      normalize=True,drop_vars = [],varpart_fn = '../data/climate_proc/var_partitioning_all_bycounty.nc',
                      title=None,
                      save_fig=False,output_fn=None):
    
    # This gets rid of a divide-by-zero warning that comes up from 
    # counties with nan values
    warnings.filterwarnings('ignore') 
    #-----------------------------------------------------------------
    # Setup
    #-----------------------------------------------------------------

    # Load previously calculated variability partitioning per county
    uncs_ds = xr.open_dataset(master_params['proc_data_dir']+master_params['obs_mod']+'/var_partitioning_all_bycounty.nc')
    uncs_ds = uncs_ds.sel(impact=var)
    
    #-----------------------------------------------------------------
    # Get relative uncertainty
    #-----------------------------------------------------------------
    # Join uncertainties into one
    uncs_ds['source'] = [k.title() for k in uncs_ds['source'].values]

    # Get in percent
    if normalize:
        uncs_ds['variance'] = uncs_ds['variance']/uncs_ds['variance'].sum('source')
    
    # Reorder to fit order of stacked barcharts / H&S
    uncs_ds = uncs_ds.sel(source=['Internal','Scenario','Model'])
    
    #-----------------------------------------------------------------
    # Transfer to geopandas geodataframe
    #-----------------------------------------------------------------
    # Turn variance into a wide pandas dataframe
    df = pd.pivot_table(uncs_ds[['FIPS','variance']].to_dataframe().reset_index().drop('county_idx',axis=1),
               index='FIPS',columns=['time','source'],values='variance').reset_index()
    df.columns = [''.join(k) for k in df.columns]
    df = df.reindex(columns=['FIPS',*[v for vs in [[t+s for s in uncs_ds.source.values] for t in uncs_ds.time.values] for v in vs]])

    # Merge variance into data_counties
    data_counties = pd.merge(data_counties,df)
    
    #-----------------------------------------------------------------
    # Figure
    #-----------------------------------------------------------------

    timeframes = uncs_ds.time.values

    var_titles = {'begc':'2010-2039',
                  'midc':'2040-2069',
                  'endc':'2070-2099'}

    cmap = cmocean.cm.speed

    fig,axs=plt.subplots(uncs_ds.dims['source'],len(timeframes),
                        figsize=(10,6),subplot_kw={'projection': ccrs.PlateCarree()})

    for source_idx in np.arange(0,uncs_ds.dims['source']):
        for var_idx in np.arange(0,len(timeframes)):
            if normalize:
                data_counties.plot(column=timeframes[var_idx]+str(uncs_ds.source[source_idx].values),
                                   cmap = cmap,
                                  vmin=0,vmax=1,ax = axs[source_idx,var_idx])
            else:
                data_counties[timeframes[var_idx]+'_'+str(uncs_ds.source[source_idx].values)] = np.log(data_counties[timeframes[var_idx]+'_'+str(uncs_ds.source[source_idx].values)])
                data_counties.plot(column=timeframes[var_idx]+str(uncs_ds.source[source_idx].values),
                                   cmap = cmap,
                                   vmin=0,
                                   vmax=np.log(data_counties[[col for col in data_counties.columns if 'dimp' in col]].max().max()*2),
                                   ax = axs[source_idx,var_idx])
                

            if source_idx == 0:
                axs[source_idx,var_idx].set_title(var_titles[timeframes[var_idx]],fontsize=15)
            if var_idx == 0:
                axs[source_idx,var_idx].text(-0.07, 0.55, uncs_ds.source[source_idx].values, va='bottom', ha='center',
                        rotation='vertical', rotation_mode='anchor',fontsize=15,
                        transform=axs[source_idx,var_idx].transAxes)

            axs[source_idx,var_idx].axis('off')

            axs[source_idx,var_idx].coastlines(resolution='110m')
            axs[source_idx,var_idx].add_feature(cartopy.feature.BORDERS, linestyle='-')
            
            # Add subplot lettering
            axs[source_idx,var_idx].annotate(string.ascii_lowercase[source_idx*len(timeframes)+var_idx]+'.',
                                         (0.01,1.015),
                                         verticalalignment='bottom',
                                         fontweight='bold',
                                         fontsize=16,
                                         xycoords='axes fraction')
    
    if title is not None:
        fig.suptitle(title,fontsize=15)

    # Colobar
    fig.subplots_adjust(right=0.8,wspace=0.05,hspace=-0.1) #and tighten up plots
    cax = fig.add_axes([0.85, 0.15, 0.025, 0.7])
    if normalize: 
        levels = mpl.ticker.MaxNLocator(nbins=256).tick_values(0,1)
        norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        sm = plt.cm.ScalarMappable(cmap=cmap,norm = norm)
        cbar = plt.colorbar(sm,cax=cax,ticks=[0,0.25,0.5,0.75,1])
        cbar.set_label(label='Fractional contribution to local variability',fontsize=12)
    else:
        levels = mpl.ticker.MaxNLocator(nbins=256).tick_values(0,
                                                               np.log(data_counties[[col for col in data_counties.columns if 'dimp' in col]].max().max()*2
                                                                     ))
        norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        sm = plt.cm.ScalarMappable(cmap=cmap,norm = norm)
        cbar = plt.colorbar(sm,cax=cax)
        cbar.set_label(label='Absolute contribution to local variability',fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    
    if save_fig:
        plt.savefig(output_fn+'.png')
        print(output_fn+'.png saved!')
        plt.savefig(output_fn+'.pdf')
        print(output_fn+'.pdf saved!')

    return fig,axs


def figure_partitioned_uncert_panel_bysource(master_params,
                                             varlist=['dmort','dgdp-pp','dyield_tot'],
                                             fnlist = None,
                                             titles = ['All models and runs','16 runs per model','without GFDL-CM3'],
                                             save_fig=True,
                                             output_fn = '../figures/part_uncert_suppdocs'):

    fig,axs = plt.subplots(3,3,
                           figsize=(10,7))#figsize=(12,10)
    
    if fnlist is None:
        fnlist = [master_params['proc_data_dir']+master_params['obs_mod']+'/var_partitioning_all.nc',
                   master_params['proc_data_dir']+master_params['obs_mod']+'/var_partitioning_16runs.nc',
                   master_params['proc_data_dir']+master_params['obs_mod']+'/var_partitioning_noCM3.nc']

    for row_idx in np.arange(0,len(varlist)):
        for col_idx in np.arange(0,len(fnlist)):
            #-----------------------------------------------------
            # Normalized stacked uncertainty plot
            #-----------------------------------------------------
            if row_idx == 0:
                title=titles[col_idx]
            else:
                title=''
            axs[row_idx,col_idx],bars = figure_part_uncert(master_params,
                                                           fn_load=fnlist[col_idx],
                                                            var=varlist[row_idx],ax = axs[row_idx,col_idx],fig = fig,
                                                            title=title,add_legend = False,
                                                            bold_title=True)
            axs[row_idx,col_idx].tick_params(axis='y',which='both',left=False,labelleft=False)
            axs[row_idx,col_idx].set_ylabel('')
            if row_idx == 0 :
                axs[row_idx,col_idx].tick_params(axis='x',which='both',top=True,labeltop=True)
            if row_idx < 2:
                axs[row_idx,col_idx].tick_params(axis='x',which='both',bottom=False,labelbottom=False)

            #-----------------------------------------------------
            # Subplot lettering
            #-----------------------------------------------------
            # Add subplot lettering
            axs[row_idx,col_idx].annotate(string.ascii_lowercase[col_idx*len(varlist)+row_idx]+'.',
                                             (0.01,0.99),
                                             verticalalignment='top',
                                             fontweight='bold',
                                             fontsize=16,
                                             xycoords='axes fraction')

    #-----------------------------------------------------
    # Plot annotations
    #-----------------------------------------------------

    fig.subplots_adjust(left=0.175)

    fig.legend(bars,['model','scenario','internal'],
               bbox_to_anchor=(0.01,0.88),
                    title='uncertainty source', 
                    loc='center left')

    ##### PUT IN HERE THE "MORTALITY", "CORN YIELD" etc. TEXT #####
    # Probably will need to use subplots_adjust to push 
    # everything to the right, then plt.text, with some
    # trial and error to get the location right. 
    # This didn't work: axs[1,0].text(-10,0.6,'Mortality',fontsize=15)
    plt.text(0.15, 0.7775, 'Mortality', fontweight='bold',fontsize=15, transform=fig.transFigure,
             horizontalalignment='right',verticalalignment='center')
    plt.text(0.15, 0.475, 'GDP\nper capita', fontweight='bold',fontsize=15, transform=fig.transFigure,
             horizontalalignment='right',verticalalignment='center')
    plt.text(0.15, 0.475-(0.7775-0.475), 'Corn yields', fontweight='bold',fontsize=15, transform=fig.transFigure,
             horizontalalignment='right',verticalalignment='center')




    if save_fig:
        plt.savefig(output_fn+'.png', bbox_inches = 'tight',facecolor = 'white')
        print(output_fn+'.png saved!')

        plt.savefig(output_fn+'.pdf', bbox_inches = 'tight')
        print(output_fn+'.pdf saved!')
        
        
def figure_bauble_panel(master_params,subset_params,
                        mods = ['CESM1-CAM5-LE','CSIRO-Mk3-6-0-LE','CanESM2-LE','EC-EARTH-LE','GFDL-CM3-LE','GFDL-ESM2M-LE','MPI-ESM-LE'],
                        var_params = {'dmort':['sum_dimp_begc','sum_dimp_midc','sum_dimp_endc'],
                                      'dgdp-pp':['sum_dimp_begc','sum_dimp_midc','sum_dimp_endc'],
                                      'dyield_tot':['dimp_begc_avg','dimp_midc_avg','dimp_endc_avg']},
                        var_titles = ['mortality (deaths/year)',
                                      'gdp per capita ($)',
                                      'corn yields (fraction)'],
                        binwidths = [1250,25,0.0125],
                        save_fig = True,output_fn = '../figures/all_stacked_baubles'):
    impact_projs = dict()
    varlist = [k for k in var_params]
    
    #----------------------------------------------------
    # Load projections
    #----------------------------------------------------
    for impvar in varlist:
        impact_projs[impvar] = dict()
        for mod in mods:
            load_fn = (master_params['impact_data_dir']+master_params['obs_mod']+
                             '/'+re.sub('\_tot','',impvar)+'_tot_'+mod+'_'+master_params['exp_name']+'_'+'-'.join(subset_params.keys())+'_bycounty_CUSA.nc')

            impact_projs[impvar][mod] = xr.open_dataset(load_fn)
            
    #----------------------------------------------------
    # Plot figure
    #----------------------------------------------------
    fig,axs = plt.subplots(3,3,figsize=(12,12))
    for var_idx in np.arange(0,len(varlist)):
        # Setup 
        var_list = var_params[varlist[var_idx]]
        
        if var_idx == 0:
            plot_titles = ['-'.join([y[0:4] for y in l]) for l in [t for k,t in subset_params.items() if k not in ['hist']]]
        else:
            plot_titles = ['','','']

        hdls = plot_hist_bauble_stacked(impact_projs[varlist[var_idx]],
                                 master_params,subset_params,plot_vars=var_list,
                                 binwidth=binwidths[var_idx],fig=fig,axs=axs[var_idx,:],
                                 var_title = var_titles[var_idx],add_legend=False,
                                 plot_titles=plot_titles,ylims=[0,54])

    # Add subplot lettering
    for row_idx in np.arange(0,3):
        for col_idx in np.arange(0,3):
            axs[row_idx,col_idx].annotate(string.ascii_lowercase[3*row_idx+col_idx]+'.',
                                         (0.01,0.98),
                                         verticalalignment='top',
                                         fontweight='bold',
                                         fontsize=16,
                                         xycoords='axes fraction')

    plt.subplots_adjust(right=0.8,hspace=-0.15)
    fig.legend(hdls,[re.sub('\-LE','',k) for k in impact_projs[varlist[var_idx]]],
               borderaxespad=1, 
               bbox_to_anchor=(0.82,0.5), title='large ensemble', 
               loc="center left")
    #plt.tight_layout()

    if save_fig:
        plt.savefig(output_fn+'.png')
        print(output_fn+'.png saved!')
        plt.savefig(output_fn+'.pdf')
        print(output_fn+'.pdf saved!')
        
        
def figure_panel_dTmeans(dms,
                         vmin=0,vmax=16,nlevels=17,
                         save_fig=False,
                         output_fn='../figures/tas_seas_LEs_rcp85-hist_seasavgs_avgacrossruns_2070-2099_1980-2009'):
    ''' Panel maps models vs. seasons of average dTs
    
    '''
    
    #----------------------------------------------------
    # Create figure
    #----------------------------------------------------
    # Make sure the seasons are in the right order
    dms = dms.reindex({'season':['DJF','MAM','JJA','SON']})
    
    # Build figure
    fig = plt.figure(figsize=(15,15))

    plt_idx = 1
    for mod in dms.model.values: #dm_rgrd
        for seas in dms.season:
            # Using LambertConformal; used by USGS for country maps with 33, 45 standard parallels 
            # (source: https://pubs.usgs.gov/gip/70047422/report.pdf)
            ax = plt.subplot(dms.sizes['model'],dms.sizes['season'],plt_idx,
                             projection=ccrs.PlateCarree())#projection=ccrs.LambertConformal(standard_parallels=(33, 45))
            plt_idx = plt_idx+1

            (dms.sel(season=seas,model=mod).tas.
             isel(lon=slice(2,-2),lat=slice(2,-2)). # because of some regridding there are some bad values on the edges
             plot.contourf(vmin=vmin,vmax=vmax,levels=nlevels,
                         cmap=cmocean.cm.matter,transform=ccrs.PlateCarree(),# cmocean.cm.amp
                         add_colorbar=False)) #cbar_kwargs={'label':mod}


            if mod == dms.model[0]:
                ax.set_title(seas.values,fontsize=20)
            else:
                ax.set_title('')

            if seas == dms.season[0]:
                ax.text(-0.07, 0.50,re.sub('\-LE$','',mod), va='bottom', ha='center',
                        rotation='vertical', rotation_mode='anchor',
                        fontsize=12,
                        transform=ax.transAxes)
                #ax.set_ylabel(mod)
            else:
                ax.set_ylabel('')
            ax.set_xlabel('')
        
            # Add coastlines and borders
            ax.coastlines()
            ax.add_feature(cfeature.BORDERS, linestyle='-',linewidth=0.5)
            
            # Add subplot lettering
            ax.annotate(int_to_roman(plt_idx-1).lower()+'.',
                         (0.01,0.98),
                         verticalalignment='top',
                         fontweight='bold',
                         fontsize=16,color='white',
                         xycoords='axes fraction')

    # Add colorbar
    fig.subplots_adjust(right=0.8)
    cax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    cmap = cmocean.cm.matter
    levels = mpl.ticker.MaxNLocator(nbins=nlevels).tick_values(vmin,vmax)
    norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    sm = plt.cm.ScalarMappable(cmap=cmap,norm=norm)
    cbar = plt.colorbar(sm,cax=cax)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(label=r'$\Delta T\ [^\circ C]$',size=20)


    # tighten subplots
    #plt.subplots_adjust(wspace=-0.1, hspace=-0.1)

    if save_fig:
        plt.savefig(output_fn+'.png',dpi=300,facecolor='white')
        print(output_fn+'.png saved!')
        plt.savefig(output_fn+'.pdf',dpi=300,facecolor='white')
        print(output_fn+'.pdf saved!')

        
def figure_dT_bymonth_maps(ds_dT,data_counties,run_idx,var='dtas_midc',
                           title='',
                           vmin=-1.5,vmax=1.5,vs_mean=True,
                           save_fig=False,output_fn=None):
    fig = plt.figure(figsize=(11,3.5))
    if (vmin<0) and (vmax>0):
        cmap = cmocean.cm.balance
    else:
        cmap = cmocean.cm.matter
    month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    
    ds_tmp =  copy.deepcopy(ds_dT[['FIPS','dtas_midc']])
    if vs_mean:
        ds_tmp[var] = ds_tmp[var].isel(run=run_idx) - ds_tmp[var].mean('run')

    for plt_idx in np.arange(0,12):
        ax = plt.subplot(2,6,plt_idx+1,projection=ccrs.PlateCarree())

        df_tmp = ds_tmp[['FIPS','dtas_midc']].isel(month=plt_idx).to_dataframe().drop(['month'],axis=1)
        month_counties = pd.merge(data_counties,df_tmp)

        month_counties.plot(column='dtas_midc',vmin=vmin,vmax=vmax,
                       cmap=cmap,ax=ax,
                       transform=ccrs.PlateCarree())
        ax.coastlines()
        ax.add_feature(cartopy.feature.BORDERS, linestyle='-')
        ax.set_title(month_names[plt_idx])


    fig.subplots_adjust(bottom=0.2)
    #cax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cax = fig.add_axes([0.15,0.05,0.7,0.05])
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap,norm=norm)
    plt.colorbar(sm,cax=cax,label='change in 30-year monthly mean T\nvs. ensemble mean [K]',
                 orientation='horizontal')
    #fig.tight_layout()

    plt.suptitle(title,fontweight='bold',fontsize=14)
    
    if save_fig:
        if output_fn is None:
            raise TypeError('output_fn must be a string')
        plt.savefig(output_fn+'.png',bbox_inches='tight',dpi=400)
        print(output_fn+'.png saved!')
        plt.savefig(output_fn+'.pdf',bbox_inches='tight')
        print(output_fn+'.pdf saved!')



