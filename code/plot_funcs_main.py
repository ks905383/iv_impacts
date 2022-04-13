#------------------- plot_funcs_main.py -------------------
# This file contains the functions needed to create the main
# text figures. Each figure is created using a function called
# wrapper_figure[#], which may call several supporting functions
# (needed to load data or create subplots), which are defined 
# either here or in plot_funcs.py.

import xarray as xr
import xagg as xa
import numpy as np
import geopandas as gpd
import pandas as pd
import os
import re
import glob
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import cartopy
from cartopy import crs as ccrs
import cartopy.feature as cfeature
import cmocean
import regionmask

from sklearn.neighbors import KernelDensity
from xhistogram.xarray import histogram as xhist

from wrapper_funcs import wrapper_var_partitioning
from plot_funcs import plot_hist_bauble_stacked_single


#--------------------------------------------------------------------------
# Figure 1
#--------------------------------------------------------------------------
def load_exposures(master_params):
    exposures = {'dng':'mort','dnh':'gdp-pp','snr':'yields'}
    mods = ['CESM1-CAM5-LE','CanESM2-LE','CSIRO-Mk3-6-0-LE','EC-EARTH-LE','GFDL-CM3-LE','GFDL-ESM2M-LE','MPI-ESM-LE']
    
    #------------------------------------
    # Load impact kdes
    #------------------------------------
    for var in tqdm(exposures):
        fn_var = exposures[var]
        exposures[var] = dict()
        for k in ['hist','endc']:
            exposures[var][k] = dict()
            for mod in mods:
                exposures[var][k][mod] = xr.open_dataset(master_params['impact_data_dir']+master_params['obs_mod']+
                                                         'aux_kdes_'+mod+'_'+fn_var+'_'+k+'.nc').drop('time',errors='ignore')

            # Put run idxs in order to be able to merge them
            if 'run' in exposures[var][k][mod]:
                for mod_idx in np.arange(1,len(mods)):
                    exposures[var][k][mods[mod_idx]]['run'] = (exposures[var][k][mods[mod_idx]]['run'] + 
                                                               exposures[var][k][mods[mod_idx-1]]['run'][-1] + 1)

            # Merge
            if var == 'snr':
                # It's a histogram for snr, instead of kde; 
                # adjust subset variable name accordingly
                exposures[var][k] = xr.merge([v for k,v in exposures[var][k].items()])
            else:
                exposures[var][k] = xr.merge([v for k,v in exposures[var][k].items()])

    #------------------------------------
    # Load tas kdes
    #------------------------------------            
    tass = dict()
    for k in tqdm(['hist','endc']):
        tass[k] = dict()
        for mod in mods:
            tass[k][mod] = xr.open_dataset(master_params['impact_data_dir']+master_params['obs_mod']+
                                           'aux_kdes_tas_'+mod+'_'+k+'.nc')
        # Put run idxs in order to be able to merge them
        if 'run' in tass[k][mod]:
            for mod_idx in np.arange(1,len(mods)):
                tass[k][mods[mod_idx]]['run'] = (tass[k][mods[mod_idx]]['run'] + 
                                                 tass[k][mods[mod_idx-1]]['run'][-1] + 1)
        # Merge
        tass[k] = xr.merge([v for k,v in tass[k].items()])
        
        
    #------------------------------------
    # Remove nan counties
    #------------------------------------  
    # Get rid of nan counties; they mess up the dot product down below
    # (dist vars is defined further down; should probably be defined in setup)
    for var in tqdm(exposures):
        for k in ['hist','endc']:
            subset = np.isnan(exposures[var][k][dist_vars[var]]).sum([d for d in exposures[var][k].dims if d not in ['pix_idx']])==0

            exposures[var][k] = exposures[var][k].isel(pix_idx = subset)

    for k in ['hist','endc']:
        subset = np.isnan(tass[k].kde).sum([d for d in tass[k].dims if d not in ['pix_idx']])==0

        tass[k] = tass[k].isel(pix_idx=subset)
        
    #------------------------------------
    # Return
    #------------------------------------  
    return exposures,tass

def figure1_core(master_params,exposures,tass,
                 exp_colors = {'hist':'tab:blue',
                              'begc':'tab:orange',
                              'midc':'tab:green',
                              'endc':'tab:red'},
                exp_labels = {'hist':'1980-2009','begc':'2010-2039','midc':'2040-2069','endc':'2070-2099'},
                colors = {'dng':'tab:grey',
                          'dnh':'tab:grey',
                          'snr':'tab:grey'},
                colors_CI = {'dng':'lightgrey',
                             'dnh':'lightgrey',
                             'snr':'lightgrey'}):
    
    #-------------------------------------------------------
    # Setup
    #-------------------------------------------------------
    # Temperature x-axis limits
    xlims = [250,315]
    # Dose-response function x values
    Xs = {'dng':np.hstack([xlims[0],np.arange(bins_dng[1]-((bins_dng[1:] - bins_dng[0:-1])/2)[1],
                       bins_dng[-2]+((bins_dng[1:] - bins_dng[0:-1])/2)[1]+1,
                      ((bins_dng[1:] - bins_dng[0:-1]))[1]),xlims[1]]),
          'dnh':np.hstack([xlims[0],np.arange(bins_dnh[1]-((bins_dnh[1:] - bins_dnh[0:-1])/2)[1],
                       bins_dnh[-2]+((bins_dnh[1:] - bins_dnh[0:-1])/2)[1]+1,
                      ((bins_dnh[1:] - bins_dnh[0:-1]))[1]),xlims[1]]),
          'snr':np.hstack([xlims[0],np.unique(np.hstack([v['bounds'] for v in plin_func]))[1:-1],xlims[1]])}

    # Dose-response function y values
    Ys = {'dng':np.hstack([dmgf_dng.values[0],dmgf_dng.values,dmgf_dng.values[-1]]),
          'dnh':np.hstack([dmgf_dnh.values[0],dmgf_dnh.values,dmgf_dnh.values[-1]]),
          'snr':np.array([0,0,0.0057,-0.0625,-0.0625])},
    
    # Dose-response function sources (in-axis caption)
    sources = {'dng':'Deschênes and\nGreenstone (2011)',
               'dnh':'Deryugina and\nHsiang (2014)',
               'snr':'Schlenker and\nRoberts (2009)'}
    
    # Y labels on exposure plots
    ylabs = {'dng':'Annual mortality rate\n[per 100,000]',
             'dnh':'Log annual\nper capita income',
             'snr':'Log annual corn yield'}
    varlist = {'dng':'dmort','dnh':'dgdp-pp','snr':'dyield_tot'}
    
    # Y labels on bar plots
    ylabs_uncert = {'dng':'Variance in mortality\n' r'[$10^7$ deaths$^2$]', #10,000,000s
                    'dnh':'Variance in per-capita income\n' r'[$10^4$ $\$^2$]', #10,000s
                    'snr':'Variance in corn yield change\n' r'[(pct point)$^2$]'}
    
    dist_Xs = {'dng':np.arange(0,1,1/50).reshape(-1,1),
           'dnh':np.arange(-0.0009,0.0007,(0.0016)/50),
           'snr':exposures['snr']['hist']['hist']['outcome_bin']}
    dist_vars = {'dng':'kde',
                 'dnh':'kde',
                 'snr':'hist'}
    dist_scales = {'dng':'tpop', # What to calculate the weighted average using
                   'dnh':'tpop',
                   'snr':'corn_prod'}
    set_exp = {'dng':True,'dnh':True,'snr':False} # Whether to np.exp() the result 
    
    #-------------------------------------------------------
    # Figures
    #-------------------------------------------------------
    fig,axs = plt.subplots(7,6,sharey=False,gridspec_kw={'width_ratios': [1,3,0.5,3,0.5,3],
                                                         'height_ratios':[1,3,0.25,3,0.25,3,1]},
                           figsize=(14,10))#figsize=(12,10)

    ytick_locs = {'dng':[0,0.4,0.8,1.2],
                  'dnh':[-0.0005,0,0.0005],
                  'snr':[-0.06,-0.04,-0.02,0]}
    vert_plot_idxs = [1,3,5]

    # Subplot lettering; in the same grid format as the 
    # axs above, to be able to effectively use the same
    # indexing
    subplot_letters = [[np.nan,'a.',np.nan,'b.',np.nan,'c.',np.nan],
                        ['d.','e.',np.nan,'f.',np.nan,'g.','h.'],
                        [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
                        [np.nan,'i.',np.nan,'j.',np.nan,'k.',np.nan],
                        [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
                        [np.nan,'l.',np.nan,'m.',np.nan,'n.',np.nan]]

    subplot_letters = [[np.nan,'d.',np.nan,np.nan,np.nan,np.nan],
                       ['a.','e.',np.nan,'i.',np.nan,'l.'],
                       [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
                       ['b.','f.',np.nan,'j.',np.nan,'m.'],
                       [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
                       ['c.','g.',np.nan,'k.',np.nan,'n.'],
                       [np.nan,'h.',np.nan,np.nan,np.nan,np.nan]]

    for k in Xs:
        f_idx = [k for k in Xs].index(k)

        #-----------------------------------------------------
        # Outcome distribution plot
        #-----------------------------------------------------
        # Get envelope for the future exp across models, runs
        plot_fill = [(exposures[k]['endc'][dist_vars[k]].min('run').transpose().
                                     dot(exposures[k]['endc'][dist_scales[k]].fillna(0))/exposures[k]['endc'][dist_scales[k]].fillna(0).sum().values),
                     (exposures[k]['endc'][dist_vars[k]].max('run').transpose().
                                     dot(exposures[k]['endc'][dist_scales[k]].fillna(0))/exposures[k]['endc'][dist_scales[k]].fillna(0).sum().values)]
        if set_exp[k]:
            plot_fill = [np.exp(v) for v in plot_fill]
        axs[vert_plot_idxs[f_idx],0].fill_betweenx(np.squeeze(dist_Xs[k]),plot_fill[0],plot_fill[1],
                                                   color='lightcoral',zorder=0,alpha=0.8,
                                                   label='internal & model uncert.')


        # Plot mean / main exposure line 
        for exp in ['hist','endc']:
            if 'run' in exposures[k][exp][dist_vars[k]].dims:
                plot_data = (exposures[k][exp][dist_vars[k]].mean('run').transpose().
                                 dot(exposures[k][exp][dist_scales[k]].fillna(0))/exposures[k][exp][dist_scales[k]].fillna(0).sum().values)
            else:
                plot_data = (exposures[k][exp][dist_vars[k]].transpose().
                                 dot(exposures[k][exp][dist_scales[k]].fillna(0))/exposures[k][exp][dist_scales[k]].fillna(0).sum().values)

            if set_exp[k]:
                plot_data = np.exp(plot_data)

            axs[vert_plot_idxs[f_idx],0].plot(plot_data,dist_Xs[k],label=exp_labels[exp],color=exp_colors[exp],zorder=1)

            # Plot baubles on the singular points to differentiate the distributions
            if k == 'snr':
                axs[vert_plot_idxs[f_idx],0].scatter(plot_data[plot_data>0][0],
                            plot_data.outcome_bin[plot_data>0][0],color=exp_colors[exp],zorder=2)
                axs[vert_plot_idxs[f_idx],0].scatter(plot_data[np.abs(plot_data.outcome_bin).argmin()],0,color=exp_colors[exp],zorder=2)


        axs[vert_plot_idxs[f_idx],0].axhline(0,color='k',linestyle='--')
        axs[vert_plot_idxs[f_idx],0].invert_xaxis()
        axs[vert_plot_idxs[f_idx],0].tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
        axs[vert_plot_idxs[f_idx],0].grid(True,which='major',axis='y')
        axs[vert_plot_idxs[f_idx],0].set_ylabel(ylabs[k])
        if k == 'snr':
            axs[vert_plot_idxs[f_idx],0].set_xscale('log')
            axs[vert_plot_idxs[f_idx],0].set_ylim(-0.065,0.015) #0.01

            axs[vert_plot_idxs[f_idx],0].annotate('(log\nscale)',(0.95,0.98),
                                                    verticalalignment='top',
                                                    horizontalalignment='right',
                                                    xycoords='axes fraction')
        if k == 'dng':
            #axs[vert_plot_idxs[f_idx],0].legend()
            axs[vert_plot_idxs[f_idx],0].legend(bbox_to_anchor=(-0.55,0.85),loc="lower right")
            axs[vert_plot_idxs[f_idx],0].set_title('Example\nexposure',fontweight='bold',rotation=45,fontsize=12)
            axs[vert_plot_idxs[f_idx],0].set_ylim(-0.05,1.2)
        axs[vert_plot_idxs[f_idx],0].set_yticks(ytick_locs[k])

        #-----------------------------------------------------
        # Damage function plot
        #-----------------------------------------------------
        axs[vert_plot_idxs[f_idx],1].plot(Xs[k]-273.15,Ys[k],color=colors[k],linewidth=5)

        axs[vert_plot_idxs[f_idx],1].set_xlim([x-273.15 for x in xlims])
        axs[vert_plot_idxs[f_idx],1].axhline(0,color='k',linestyle='--')
        axs[vert_plot_idxs[f_idx],1].tick_params(axis='x',which='both',bottom=False,labelbottom=False)
        axs[vert_plot_idxs[f_idx],1].tick_params(axis='y',which='both',left=False,labelleft=False)#,right=True,labelright=True)
        axs[vert_plot_idxs[f_idx],1].grid(True,which='major',axis='x')
        axs[vert_plot_idxs[f_idx],1].grid(True,which='major',axis='y')
        axs[vert_plot_idxs[f_idx],1].set_ylim(axs[vert_plot_idxs[f_idx],0].get_ylim())
        axs[vert_plot_idxs[f_idx],1].set_yticks(ytick_locs[k])

        # Text 
        add_cites = True
        if add_cites:
            text_params = [['Deschênes & Greenstone (2011)',0.95,'top'],
                           ['Deryugina & Hsiang (2014)',0.95,'top'],
                           ['Schlenker & Roberts (2009)',0.05,'bottom']] #0.05
            t = axs[vert_plot_idxs[f_idx],1].text(0.125,text_params[f_idx][1],#0.05
                                              text_params[f_idx][0],
                                              horizontalalignment='left',
                                              verticalalignment=text_params[f_idx][2],
                                              transform=axs[vert_plot_idxs[f_idx],1].transAxes,
                                                 bbox={'boxstyle':'round4'})
            t.set_bbox(dict(facecolor='white', alpha=0.5))


        #-----------------------------------------------------
        # Normalized stacked uncertainty plot
        #-----------------------------------------------------
        axs[vert_plot_idxs[f_idx],2].axis('off')
        # Setup 
        if varlist[k] == 'dyield_tot':
            var_list = ['dimp_begc_avg','dimp_midc_avg','dimp_endc_avg']
        else:
            var_list = ['sum_dimp_begc','sum_dimp_midc','sum_dimp_endc']


        # Normalized figure
        if f_idx == 0:
            title = 'Uncertainty by source\n(normalized)'
        else:
            title = ''


        axs[vert_plot_idxs[f_idx],3],bars = figure_part_uncert(master_params,
                                var=varlist[k],ax = axs[vert_plot_idxs[f_idx],3],fig = fig,
                                title=title,add_legend = False,var_list = var_list,
                                bold_title=True)
        axs[vert_plot_idxs[f_idx],3].tick_params(axis='y',which='both',left=False,labelleft=False)
        axs[vert_plot_idxs[f_idx],3].set_ylabel('')
        if f_idx == 0 :
            axs[vert_plot_idxs[f_idx],3].tick_params(axis='x',which='both',top=True,labeltop=True)
        if f_idx < 2:
            axs[vert_plot_idxs[f_idx],3].tick_params(axis='x',which='both',bottom=False,labelbottom=False)


        #-----------------------------------------------------
        # Raw stacked uncertainty plot
        #-----------------------------------------------------
        axs[vert_plot_idxs[f_idx],4].axis('off')
        if f_idx == 0:
            title = 'Uncertainty by source\n(total)'
        else:
            title = ''

        axs[vert_plot_idxs[f_idx],5],bars = figure_part_uncert(master_params,
                                var=varlist[k],ax = axs[vert_plot_idxs[f_idx],5],fig = fig,
                                normalize=False,
                                title=title,add_legend = False,var_list = var_list,
                                bold_title=True)
        axs[vert_plot_idxs[f_idx],5].set_ylabel(ylabs_uncert[k])
        axs[vert_plot_idxs[f_idx],5].yaxis.set_label_position('right')
        axs[vert_plot_idxs[f_idx],5].tick_params(axis='y',which='both',left=False,labelleft=False,
                                 right=True,labelright=True)
        if k == 'dng':
            # Change yticks to 10 million 
            axs[vert_plot_idxs[f_idx],5].get_yaxis().set_major_formatter(
                mpl.ticker.FuncFormatter(lambda x, p: format(int(x/1e7), ',')))
        elif k == 'dnh':
            # Change yticks to 10,000 $^2
            axs[vert_plot_idxs[f_idx],5].get_yaxis().set_major_formatter(
                mpl.ticker.FuncFormatter(lambda x, p: format(int(x/10000), ',')))

        if f_idx == 0 :
            axs[vert_plot_idxs[f_idx],5].tick_params(axis='x',which='both',top=True,labeltop=True)
        if f_idx < 2:
            axs[vert_plot_idxs[f_idx],5].tick_params(axis='x',which='both',bottom=False,labelbottom=False)

    #-----------------------------------------------------
    # Temperature pdes
    #-----------------------------------------------------
    #### Pop-weighted temperature
    # Run mean line
    for exp in ['hist','endc']:
        if 'run' in tass[exp].dims:
            axs[0,1].plot(tass[exp].kdex-273.15,
                      np.exp(tass[exp].kde.mean('run')).dot(tass[exp].tpop.fillna(0))/tass[exp].tpop.sum(),
                      color=exp_colors[exp])
        else:
            axs[0,1].plot(tass[exp].kdex-273.15,
                      np.exp(tass[exp].kde).dot(tass[exp].tpop.fillna(0))/tass[exp].tpop.sum(),
                      color=exp_colors[exp])
    # Plot envelope
    plot_fill = [np.exp(tass[exp].kde.min('run')).dot(tass[exp].tpop.fillna(0))/tass[exp].tpop.sum(),
                 np.exp(tass[exp].kde.max('run')).dot(tass[exp].tpop.fillna(0))/tass[exp].tpop.sum()]
    axs[0,1].fill_between(np.squeeze(tass[exp].kdex-273.15),plot_fill[0],plot_fill[1],color='lightcoral',zorder=0,alpha=0.8)


    axs[0,1].set_xlim((Xs[k][0]-273.15,Xs[k][-1]-273.15))
    axs[0,1].grid(True,which='major',axis='x')
    axs[0,1].tick_params(axis='x',which='both',bottom=False,labelbottom=False,top=True,labeltop=True)
    axs[0,1].tick_params(axis='y',which='both',left=False,labelleft=False)

    axs[0,1].set_title('Dose-response function',fontweight='bold')

    #axs[0,1].text(-19,0.035,'(pop-weighted)')
    #axs[0,1].text(-17.5,0.050,'(pop-weighted)')
    axs[0,1].annotate('(pop-weighted)',(0.11,0.94),
                      verticalalignment='top',
                      xycoords='axes fraction')



    for ax_idx in [0,2,3,4,5]:
        axs[0,ax_idx].axis('off')


    #### Corn yield-weighted temperature
    # Run mean line
    for exp in ['hist','endc']:
        if 'run' in tass[exp].dims:
            axs[6,1].plot(tass[exp].kdex-273.15,
                          np.exp(tass[exp].kde_grwng.mean('run')).dot(tass[exp].corn_prod.fillna(0))/tass[exp].corn_prod.sum(),
                          color=exp_colors[exp])
        else:
            axs[6,1].plot(tass[exp].kdex-273.15,
                  np.exp(tass[exp].kde_grwng).dot(tass[exp].corn_prod.fillna(0))/tass[exp].corn_prod.sum(),
                  color=exp_colors[exp])
    # Plot envelope
    plot_fill = [np.exp(tass[exp].kde_grwng.min('run')).dot(tass[exp].corn_prod.fillna(0))/tass[exp].corn_prod.sum(),
                 np.exp(tass[exp].kde_grwng.max('run')).dot(tass[exp].corn_prod.fillna(0))/tass[exp].corn_prod.sum()]
    axs[6,1].fill_between(np.squeeze(tass[exp].kdex-273.15),plot_fill[0],plot_fill[1],color='lightcoral',zorder=0,alpha=0.8)

    axs[6,1].set_xlim((Xs[k][0]-273.15,Xs[k][-1]-273.15))
    axs[6,1].grid(True,which='major',axis='x')
    axs[6,1].tick_params(axis='y',which='both',left=False,labelleft=False)
    axs[6,1].set_xlabel('Temperature [C]')

    #axs[6,1].text(-19,0.045,'(yield-weighted)')
    #axs[6,1].text(-17.5,0.060,'(yield-weighted)')
    axs[6,1].annotate('(yield-weighted)',(0.11,0.94),
                      verticalalignment='top',
                      xycoords='axes fraction')

    for ax_idx in [0,2,3,4,5]:
        axs[6,ax_idx].axis('off')

    #-----------------------------------------------------
    # Subplot lettering
    #-----------------------------------------------------
    for col_idx in np.arange(0,np.shape(axs)[1]):
        for row_idx in np.arange(0,np.shape(axs)[0]):
            if type(subplot_letters[row_idx][col_idx]) is str:
                axs[row_idx,col_idx].annotate(subplot_letters[row_idx][col_idx],
                                                         (0.01,0.98),
                                                         verticalalignment='top',
                                                         fontweight='bold',
                                                         fontsize=16,
                                                         xycoords='axes fraction')


    #-----------------------------------------------------
    # Plot annotations
    #-----------------------------------------------------
    for ax_idx in np.arange(0,6):
        axs[2,ax_idx].axis('off')
        axs[4,ax_idx].axis('off')
    axs[4,1].axhline(0,linestyle='-',color='k')
    axs[4,1].set_ylim((-1,1))

    fig.legend(bars,['model','scenario','internal'],
               bbox_to_anchor=(0.71,0.08),ncol=3,
                    title='uncertainty source', 
                    loc='upper center')

    ## Row captions
    fig.subplots_adjust(wspace=0,hspace=0,left=0.22)
    plt.text(0.125, 0.7775, 'Mortality', fontweight='bold',fontsize=15, transform=fig.transFigure,
             horizontalalignment='right',verticalalignment='center')
    plt.text(0.125, 0.51, 'GDP\nper capita', fontweight='bold',fontsize=15, transform=fig.transFigure,
             horizontalalignment='right',verticalalignment='center')
    plt.text(0.125, 0.23, 'Corn yields', fontweight='bold',fontsize=15, transform=fig.transFigure,
             horizontalalignment='right',verticalalignment='center')



    return fig



def wrapper_figure1(master_params,
                    save_fig=False,output_fn=None):
    
    exposures,tass = load_exposures(master_params)
    
    fig = figure1_core(master_params,exposures,tass)
    
    if save_fig:
        output_fn = '../figures/fig1_combined'
        plt.savefig(output_fn+'.png', bbox_inches = 'tight',facecolor = 'white')
        print(output_fn+'.png saved!')

        plt.savefig(output_fn+'.pdf', bbox_inches = 'tight')
        print(output_fn+'.pdf saved!')



    
        




#--------------------------------------------------------------------------
# Figure 2
#--------------------------------------------------------------------------

def load_timeseries(master_params,mod):
    # Get list of base files
    fns_base = [fn for fn in glob.glob(master_params['mod_data_dir']+'*/*.nc') if re.search(master_params['mod_search_str'],re.split('\/',fn)[-1])]
    models = [re.split('\/',fn)[-2] for fn in fns_base]
    models = np.unique(models)
    load_fn = fns_base[np.where(models==mod)[0][0]]
    
    # Load temperature 
    ds = xr.open_dataset(load_fn)
    ds = ds.drop(['lat_bnds','lon_bnds'])

    # Aggregate to yearly means
    ds.mean('time') # just to speed up loading, lol
    ds = ds.groupby('time.year').mean()

    # Mask out to USA (the input file is already roughly subset to a bounding box of 
    # CONUS, so this should only subset to CONUS boundaries)
    mask = regionmask.defined_regions.natural_earth_v5_0_0.countries_110.mask(ds)
    mask_idx = regionmask.defined_regions.natural_earth_v5_0_0.countries_110.map_keys("United States of America")
    dsm = ds.where(mask==mask_idx).mean(['lat','lon'])
    
    # Return
    return dsm

def load_impact_change(master_params,mod,data_counties,
                       impact_var='dmort',agg_var='sum_dimp_midc',loc_var='dimp_midc'):
    # Load mortality data
    dmort = xr.open_dataset(master_params['impact_data_dir']+master_params['obs_mod']+
                            '/'+impact_var+'_tot_'+mod+'_rcp85_hist-begc-midc-endc_bycounty_CUSA.nc')

    # Calculate mean 
    dmort['dmort_ref'] = dmort.dimp_midc.mean('run')
    # (or should this be the median run for the total CONUS estimate? can try with both)
    # dmort['dimp_ref'] = dmort.dimp_midc.isel(run=np.abs(dmort.sum_dimp_midc - dmort.sum_dimp_midc.median('run')).argmin())

    # Get idx of least mortality run and most mortality run
    idxs = [dmort[agg_var].argmin().values,dmort[agg_var].argmax().values]

    #-----------------------------------------------------------------
    # Transfer to geopandas geodataframe
    #-----------------------------------------------------------------
    # Turn variance into a wide pandas dataframe
    df_dmorts = pd.pivot_table(dmort[[loc_var,'FIPS','dmort_ref']].
                               isel(run=idxs).to_dataframe().reset_index().drop('county_idx',axis=1),
                               index=['FIPS','dmort_ref'],columns=['run'],values=[loc_var]).reset_index()
    df_dmorts.columns = ['FIPS','dmort_ref','dmort_min','dmort_max']

    # Get relative values
    df_dmorts['ddmort_min'] = df_dmorts['dmort_min'] - df_dmorts['dmort_ref']
    df_dmorts['ddmort_max'] = df_dmorts['dmort_max'] - df_dmorts['dmort_ref']
    df_dmorts = df_dmorts.drop(['dmort_min','dmort_max'],axis=1)

    # Get values as % of reference change
    df_dmorts['ddmort_min_rel'] = df_dmorts['ddmort_min'] / df_dmorts['dmort_ref']
    df_dmorts['ddmort_max_rel'] = df_dmorts['ddmort_max'] / df_dmorts['dmort_ref']


    # Merge variance into data_counties
    data_counties = pd.merge(data_counties,df_dmorts)
    
    # Return
    return data_counties,dmort,idxs

def load_t_change(master_params,data_counties,mod,idxs,agg_var='dtas_midc',loc_var='dtas_midc'):
    # Load temperature change data
    dT = xr.open_dataset(master_params['impact_data_dir']+master_params['obs_mod']+
                         '/dtasmean_tot_'+mod+'_rcp85_hist-begc-midc-endc_bycounty_CUSA.nc')

    # Calculate mean 
    dT['dT_ref'] = dT[agg_var].mean('run')
    # (or should this be the median run for the total CONUS estimate? can try with both)
    # dmort['dimp_ref'] = dmort.dimp_midc.isel(run=np.abs(dmort.sum_dimp_midc - dmort.sum_dimp_midc.median('run')).argmin())

    #-----------------------------------------------------------------
    # Transfer to geopandas geodataframe
    #-----------------------------------------------------------------
    # Turn variance into a wide pandas dataframe
    df_dT = pd.pivot_table(dT[[loc_var,'FIPS','dT_ref']].isel(run=idxs).to_dataframe().reset_index().drop('county_idx',axis=1),
                               index=['FIPS','dT_ref'],columns=['run'],values=[loc_var]).reset_index()
    df_dT.columns = ['FIPS','dT_ref','dimp_min','dimp_max']

    # Get relative values
    df_dT['ddT_min'] = df_dT['dimp_min'] - df_dT['dT_ref']
    df_dT['ddT_max'] = df_dT['dimp_max'] - df_dT['dT_ref']
    df_dT = df_dT.drop(['dimp_min','dimp_max'],axis=1)

    # Get values as % of reference change
    df_dT['ddT_min_rel'] = df_dT['ddT_min'] / df_dT['dT_ref']
    df_dT['ddT_max_rel'] = df_dT['ddT_max'] / df_dT['dT_ref']


    # Merge variance into data_counties
    data_counties = pd.merge(data_counties,df_dT)
    
    # Return data_counties
    return data_counties
    
def figure2_core(dsm,data_counties,mod,dmort,idxs):
    ref_string = 'ensemble mean change'
    letter_pos = (-0.1,0.98)

    fig = plt.figure(figsize=(16,6),facecolor='white')
    # Now create the tiled board
    spec = fig.add_gridspec(4, 13)

    hspecs = [slice(0,2),slice(2,4)]
    wspecs = [slice(0,3),3,slice(4,7),7,slice(8,11),slice(11,13)]

    #-------------------------------------
    # dT time series
    #-------------------------------------
    ax = fig.add_subplot(spec[1:3,wspecs[0]])
    for run_idx in np.arange(0,dsm.dims['run']):
        dsm['tas'].isel(run=run_idx).plot(color='lightgrey',ax=ax)

    l = [None]*2
    l[0] = dsm['tas'].isel(run=idxs[0]).plot(color='tab:blue',ax=ax)
    l[1] = dsm['tas'].isel(run=idxs[1]).plot(color='tab:red',ax=ax)

    # Add 'origin point'
    plt.scatter(1945,dsm['tas'].isel(year=0).mean('run'),facecolor='lightgrey')
    # Add sample lines from origin point to 1980 
    plt.plot([1945,1980],[dsm['tas'].isel(year=0).mean('run'),dsm['tas'].isel(year=0).min('run')+0.11],linestyle='--',color='grey')
    plt.plot([1945,1980],[dsm['tas'].isel(year=0).mean('run'),dsm['tas'].isel(year=0).max('run')-0.11],linestyle='--',color='grey')
    plt.plot([1945,1980],[dsm['tas'].isel(year=0).mean('run'),dsm['tas'].isel(year=0).median('run')],linestyle='--',color='grey')

    # Highlight hist, midc
    ax.axvspan(1980,2009, alpha=0.5, color='beige')
    ax.axvspan(2040,2069, alpha=0.5, color='beige')
    ax.text(1995,290.5,'hist.',ha='center',fontweight='bold')
    ax.text(2055,290.5,'mid-c.',ha='center',fontweight='bold')
    plt.annotate(text='',xy=(1980,290.25), xytext=(2009,290.25), arrowprops=dict(arrowstyle='<->'))
    plt.annotate(text='',xy=(2040,290.25), xytext=(2069,290.25), arrowprops=dict(arrowstyle='<->'))

    # Set xaxis "break" using labels
    ax.set_xlim(1940,2100)
    d = 1.5
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax.plot([0.11,0.15], [0, 0], transform=ax.transAxes, **kwargs)
    #ax.plot([0.13,0.13], [0, 0], transform=ax.transAxes, **kwargs)

    labels = np.arange(1940,2101,40)
    ax.set_xticks(labels)
    labels = [str(k) for k in labels]
    labels[0] = '1845'

    ax.set_xticklabels(labels)
    b = ax.get_xticklabels()

    # Annotations
    ax.set_title(re.sub('\-LE','',mod)+'\nyearly CONUS mean temperature')
    ax.set_xlabel('year')
    ax.set_ylabel('temperature [K]')

    # Legend
    #plt.legend(handles=[blue_line])
    fig.legend(handles=[mlines.Line2D([],[],color='tab:red',label='highest mid-century mortality'),
                         mlines.Line2D([],[],color='tab:blue',label='lowest mid-century mortality')],
               bbox_to_anchor=(0.21,0.24),
                    title='ensemble member with:', 
                    loc='upper center')
    ax.annotate('a.',
                 letter_pos,
                 verticalalignment='top',
                 fontweight='bold',
                 fontsize=16,
                 xycoords='axes fraction')

    #-------------------------------------
    # dT maps
    #-------------------------------------
    ax = fig.add_subplot(spec[hspecs[0],wspecs[2]],projection=ccrs.PlateCarree())
    data_counties.plot(column='ddT_max',vmin=-0.8,vmax=0.8,
                       cmap=cmocean.cm.balance,ax=ax,
                       transform=ccrs.PlateCarree())
    #ax.set_title('Change in mean temperature\nvs. '+ref_string)
    ax.set_title(r'$\bf{change\ in\ temperature}$'+'\n'+r'$\bf{vs. ensemble\ mean}$'+
                 '\n(ensemble member with\nhighest mid-century mortality)')
    ax.coastlines()
    ax.add_feature(cartopy.feature.BORDERS, linestyle='-')
    ax.annotate('b.',
                 letter_pos,
                 verticalalignment='top',
                 fontweight='bold',
                 fontsize=16,
                 xycoords='axes fraction')


    ax = fig.add_subplot(spec[hspecs[1],wspecs[2]],projection=ccrs.PlateCarree())
    data_counties.plot(column='ddT_min',vmin=-0.5,vmax=0.5,
                       cmap=cmocean.cm.balance,ax=ax,
                       transform=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cartopy.feature.BORDERS, linestyle='-')

    ax.set_title('(ensemble member with\nlowest mid-century mortality)')
    ax.annotate('c.',
                 letter_pos,
                 verticalalignment='top',
                 fontweight='bold',
                 fontsize=16,
                 xycoords='axes fraction')

    #-------------------------------------
    # dmort maps
    #-------------------------------------
    ax = fig.add_subplot(spec[hspecs[0],wspecs[4]],projection=ccrs.PlateCarree())
    data_counties.plot(column='ddmort_max',vmin=-10,vmax=10,
                       cmap=cmocean.cm.curl,ax=ax,
                       transform=ccrs.PlateCarree())
    #ax.set_title('Change in mortality\nvs. '+ref_string)
    ax.set_title(r'$\bf{change\ in\ mortality\ rate}$'+'\n'+r'$\bf{vs. ensemble\ mean}$'+
                 '\n(ensemble member with\nhighest mid-century mortality)')
    ax.coastlines()
    ax.add_feature(cartopy.feature.BORDERS, linestyle='-')
    ax.annotate('d.',
                 letter_pos,
                 verticalalignment='top',
                 fontweight='bold',
                 fontsize=16,
                 xycoords='axes fraction')

    ax = fig.add_subplot(spec[hspecs[1],wspecs[4]],projection=ccrs.PlateCarree())
    data_counties.plot(column='ddmort_min',vmin=-10,vmax=10,
                       cmap=cmocean.cm.curl,ax=ax,
                       transform=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cartopy.feature.BORDERS, linestyle='-')
    ax.set_title('\n(ensemble member with\nlowest mid-century mortality)')
    ax.annotate('e.',
                 letter_pos,
                 verticalalignment='top',
                 fontweight='bold',
                 fontsize=16,
                 xycoords='axes fraction')

    #-------------------------------------
    # dmort metrics
    #-------------------------------------
    ax = fig.add_subplot(spec[hspecs[0],wspecs[5]])
    ax.text(0.05,0.5,
            '+'+str(int(np.round(dmort['sum_dimp_midc'].isel(run=idxs[1]).values/100)*100))+'\nannual deaths\nvs. historical',
            ha='left',va='center',fontsize=16)
    ax.axis('off')

    ax = fig.add_subplot(spec[hspecs[1],wspecs[5]])
    ax.text(0.05,0.5,
            str(int(np.round(dmort['sum_dimp_midc'].isel(run=idxs[0]).values/100)*100))+'\nannual deaths\nvs. historical',
            ha='left',va='center',fontsize=16)
    ax.axis('off')

    #-------------------------------------
    # arrows
    #-------------------------------------
    # Left arrow max 
    ax = fig.add_subplot(spec[hspecs[0],wspecs[1]])
    plt.arrow(-0.8,-0.3,1,0.2,color='tab:red',width=0.1)
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.axis('off')

    # Left arrow min 
    ax = fig.add_subplot(spec[hspecs[1],wspecs[1]])
    plt.arrow(-0.8,0.3,1,-0.2,color='tab:blue',width=0.1)
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.axis('off')

    # Right arrow max
    ax = fig.add_subplot(spec[hspecs[0],wspecs[3]])
    plt.arrow(-0.8,0,1,0,color='tab:red',width=0.1)
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.axis('off')

    # Right arrow min
    ax = fig.add_subplot(spec[hspecs[1],wspecs[3]])
    plt.arrow(-0.8,0,1,0,color='tab:blue',width=0.1)
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.axis('off')

    #-------------------------------------
    # Colorbars
    #-------------------------------------
    # Just going to do try to set this up manually

    # Delta T 
    cax = fig.add_axes([0.3525, 0.12, 0.2, 0.02])
    norm = mpl.colors.Normalize(vmin=-0.8, vmax=0.8)
    sm = plt.cm.ScalarMappable(cmap=cmocean.cm.balance,norm=norm)
    plt.colorbar(sm,cax=cax,label='difference in 30-year mean temperature [K]',orientation='horizontal')

    # Delta mort
    cax = fig.add_axes([0.594, 0.12, 0.2, 0.02])
    norm = mpl.colors.Normalize(vmin=-10, vmax=10)
    sm = plt.cm.ScalarMappable(cmap=cmocean.cm.curl,norm=norm)
    plt.colorbar(sm,cax=cax,label=r'difference in mortality rate / 100,000',orientation='horizontal')


    #-------------------------------------
    # General params
    #-------------------------------------
    plt.subplots_adjust(hspace=0)
    

    #-------------------------------------
    # Return
    #-------------------------------------
    return fig
    
    
    
def wrapper_figure2(master_params,data_counties,mod='CESM1-CAM5-LE',
                    save_fig=False,output_fn=None):
    
    # Load constitutent parts
    dsm = load_timeseries(master_params,mod)
    data_counties,dmort,idxs = load_impact_change(master_params,mod,data_counties)
    data_counties = load_t_change(master_params,data_counties,mod,idxs)
    
    # Generate figure
    fig = figure2_core(dsm,data_counties,mod,dmort,idxs)
    
    # Save 
    if save_fig:
        plt.savefig(output_fn+'.png',dpi=400)
        print(output_fn+'.png saved!')
        plt.savefig(output_fn+'.pdf')
        print(output_fn+'.pdf saved!')


#--------------------------------------------------------------------------
# Figure 3
#--------------------------------------------------------------------------
def wrapper_figure3(master_params,subset_params,save_fig=False,fn='../figures/dmort_collapsed_hist_begc-midc-endc'):
    # Load mortality info
    dmorts = dict()
    mods = ['CESM1-CAM5-LE','CSIRO-Mk3-6-0-LE','CanESM2-LE','EC-EARTH-LE','GFDL-CM3-LE','GFDL-ESM2M-LE','MPI-ESM-LE']
    for mod in mods:
        load_fn = (master_params['impact_data_dir']+master_params['obs_mod']+
                   '/dmort_tot_'+mod+'_'+master_params['exp_name']+'_'+'-'.join(subset_params.keys())+'_bycounty_CUSA.nc')

        dmorts[mod] = xr.open_dataset(load_fn)
        
    # Create single-row stacked bauble plot with just the mortality data
    plot_hist_bauble_stacked_single(dmorts,master_params,subset_params,
                                     plot_vars=['sum_dimp_begc','sum_dimp_midc','sum_dimp_endc'],binwidth=1250,
                                     plot_titles=['2010-2039','2040-2069','2070-2099'],
                                     save_fig=save_fig,fn=fn)


#--------------------------------------------------------------------------
# Figure 4
#--------------------------------------------------------------------------
def vals_to_color_triangle(xs,dim='source',col_names = {'r':'internal','g':'scenario','b':'model'}):
    ''' Convert bound values along three relative axes into color triangle colors

    '''
    if type(xs)==xr.core.dataarray.DataArray:
        out_colors = np.array([mpl.colors.to_hex([1,1,1])]*xs.values.shape[1])

        # > 2/3 internal: tab:red
        out_colors[xs.sel({dim:col_names['r']})>(2/3)] = mpl.colors.to_hex([214/255,39/255,40/255])
        # > 2/3 scenario: tab:green
        out_colors[xs.sel({dim:col_names['g']})>(2/3)] = mpl.colors.to_hex([40/255,160/255,44/255])
        # > 2/3 model: tab:blue
        out_colors[xs.sel({dim:col_names['b']})>(2/3)] = mpl.colors.to_hex([31/255,119/255,180/255])

        # 1/3-2/3 internal & 1/3-2/3 scenario: tab:orange
        out_colors[(xs.sel({dim:col_names['r']})>(1/3)) & (xs.sel({dim:col_names['r']})<=(2/3)) & 
                   (xs.sel({dim:col_names['g']})>(1/3)) & (xs.sel({dim:col_names['g']})<=(2/3))] = mpl.colors.to_hex([255/255,187/255,120/255])
        # 1/3-2/3 internal & 1/3-2/3 model: tab:purple
        out_colors[(xs.sel({dim:col_names['r']})>(1/3)) & (xs.sel({dim:col_names['r']})<=(2/3)) & 
                   (xs.sel({dim:col_names['b']})>(1/3)) & (xs.sel({dim:col_names['b']})<=(2/3))] = mpl.colors.to_hex([247/255,182/255,210/255])
        # 1/3-2/3 scenario & 1/3-2/3 model: tab:cyan
        out_colors[(xs.sel({dim:col_names['g']})>(1/3)) & (xs.sel({dim:col_names['g']})<=(2/3)) & 
                   (xs.sel({dim:col_names['b']})>(1/3)) & (xs.sel({dim:col_names['b']})<=(2/3))] = mpl.colors.to_hex([158/255,218/255,229/255])

        # Light blue
        out_colors[(xs.sel({dim:col_names['r']})<(1/3)) & (xs.sel({dim:col_names['g']})<(1/3)) 
                   & (xs.sel({dim:col_names['b']})<=(2/3))] = mpl.colors.to_hex([174/255,199/255,232/255])
        # Light red
        out_colors[(xs.sel({dim:col_names['r']})<=(2/3)) & (xs.sel({dim:col_names['g']})<(1/3)) 
                   & (xs.sel({dim:col_names['b']})<(1/3))] = mpl.colors.to_hex([255/255,152/255,150/255])
        # Light green
        out_colors[(xs.sel({dim:col_names['r']})<(1/3)) & (xs.sel({dim:col_names['g']})<=(2/3)) 
                   & (xs.sel({dim:col_names['b']})<(1/3))] = mpl.colors.to_hex([152/255,223/255,138/255])

        #(this overlaps a bit with the above)
        # where all three are within 0.1, grey
        out_colors[(xs.sel({dim:col_names['r']})>3/12) & (xs.sel({dim:col_names['r']})<5/12) & 
                   (xs.sel({dim:col_names['g']})>3/12) & (xs.sel({dim:col_names['g']})<5/12) &
                   (xs.sel({dim:col_names['b']})>3/12) & (xs.sel({dim:col_names['b']})<5/12)] = mpl.colors.to_hex([0.8,0.8,0.8])

        # White for nans
        out_colors[np.isnan(xs.sel({dim:col_names['r']}))] = mpl.colors.to_hex([1,1,1])
    elif type(xs) in [gpd.geodataframe.GeoDataFrame,pd.core.frame.DataFrame]:
        out_colors = np.array(['white']*len(xs))

        # > 2/3 internal: tab:red
        out_colors[xs[col_names['r']]>(2/3)] = mpl.colors.to_hex([214/255,39/255,40/255])
        # > 2/3 scenario: tab:green
        out_colors[xs[col_names['g']]>(2/3)] = mpl.colors.to_hex([40/255,160/255,44/255])
        # > 2/3 model: tab:blue
        out_colors[xs[col_names['b']]>(2/3)] = mpl.colors.to_hex([31/255,119/255,180/255])

        # 1/3-2/3 internal & 1/3-2/3 scenario: tab:orange
        out_colors[(xs[col_names['r']]>(1/3)) & (xs[col_names['r']]<=(2/3)) & 
                   (xs[col_names['g']]>(1/3)) & (xs[col_names['g']]<=(2/3))] = mpl.colors.to_hex([255/255,187/255,120/255])
        # 1/3-2/3 internal & 1/3-2/3 model: tab:purple
        out_colors[(xs[col_names['r']]>(1/3)) & (xs[col_names['r']]<=(2/3)) & 
                   (xs[col_names['b']]>(1/3)) & (xs[col_names['b']]<=(2/3))] = mpl.colors.to_hex([247/255,182/255,210/255])
        # 1/3-2/3 scenario & 1/3-2/3 model: tab:cyan
        out_colors[(xs[col_names['g']]>(1/3)) & (xs[col_names['g']]<=(2/3)) & 
                   (xs[col_names['b']]>(1/3)) & (xs[col_names['b']]<=(2/3))] = mpl.colors.to_hex([158/255,218/255,229/255])

        # Light blue
        out_colors[(xs[col_names['r']]<(1/3)) & (xs[col_names['g']]<(1/3)) 
                   & (xs[col_names['b']]<=(2/3))] = mpl.colors.to_hex([174/255,199/255,232/255])
        # Light red
        out_colors[(xs[col_names['r']]<=(2/3)) & (xs[col_names['g']]<(1/3)) 
                   & (xs[col_names['b']]<(1/3))] = mpl.colors.to_hex([255/255,152/255,150/255])
        # Light green
        out_colors[(xs[col_names['r']]<(1/3)) & (xs[col_names['g']]<=(2/3)) 
                   & (xs[col_names['b']]<(1/3))] = mpl.colors.to_hex([152/255,223/255,138/255])

        #(this overlaps a bit with the above)
        # where all three are within 0.1, white
        out_colors[(xs[col_names['r']]>3/12) & (xs[col_names['r']]<5/12) & 
                   (xs[col_names['g']]>3/12) & (xs[col_names['g']]<5/12) &
                   (xs[col_names['b']]>3/12) & (xs[col_names['b']]<5/12)] = mpl.colors.to_hex([0.8,0.8,0.8])

        # White for nans
        out_colors[np.isnan(xs[col_names['r']])] = mpl.colors.to_hex([1,1,1])
    
    return out_colors
    


def figure4_core(varpart,data_counties):
    
    time_titles = ['beg-c.\n2010-2039','mid-c.\n2040-2069','end-c.\n2070-2099']
    impact_titles = ['Mortality','GDP per capita','Corn yields']

    fig = plt.figure(figsize=(15,8))
    for impact_idx in np.arange(0,3):
        for time_idx in np.arange(0,3):
            ds_tmp = varpart.isel(impact=impact_idx,time=time_idx)[['variance','FIPS']]
            # Turn scenario, model, internal into color triangle colors
            ds_tmp['colors'] = xr.DataArray(data=vals_to_color_triangle(ds_tmp['variance']),
                                            dims=['county_idx'])
            # Add to geodataframe
            df_tmp = pd.merge(data_counties,
                              (ds_tmp[['colors','FIPS']].to_dataframe().reset_index().drop(['county_idx'],axis=1)),
                              on='FIPS')

            # Plot
            ax = plt.subplot(3,4,impact_idx*4+time_idx+1,
                             projection=ccrs.PlateCarree())
            df_tmp.plot(color=df_tmp.colors,ax=ax)
            ax.coastlines()
            ax.add_feature(cfeature.BORDERS)

            if impact_idx == 0:
                ax.set_title(time_titles[time_idx],fontsize=15)

            if time_idx == 0:
                ax.text(-0.07, 0.55, impact_titles[impact_idx], va='bottom', ha='center',
                            rotation='vertical', rotation_mode='anchor',fontsize=15,
                            transform=ax.transAxes)
                
    return fig



def wrapper_figure4(master_params,data_counties,
                    varpart_suffix='_all_bycounty',
                    save_fig=False,output_fn=None):
    
    # Load partitioned variance
    varpart = xr.open_dataset(master_params['impact_data_dir']+master_params['obs_mod']+'/var_partitioning'+varpart_suffix+'.nc')
    # Normalize into relative partitioned variance
    varpart['variance'] = varpart['variance'] / varpart['variance'].sum('source')
    
    # Plot figure
    fig = figure4_core(varpart,data_counties)
    
    # Save figure
    if save_fig:
        plt.savefig(output_fn+'.png',dpi=300)
        print(output_fn+'.png saved!')
        plt.savefig(output_fn+'.pdf')
        print(output_fn+'.pdf saved!')
