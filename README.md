# The Role of Internal Climate Variability in Climate Impact Projections
Code and data links for Schwarzwald and Lenssen, in preparation.


## Code
All code needed to replicate Schwarzwald and Lenssen is saved in the `code/` directory above. To make the code work locally: 

1. Create a new environment using the provided `iv_impact_env.yaml` file (using the command `conda env create -f environment.yml`)
2. Set all paths to data in the `dir_list.csv` file: 
    - `obs`: where observational data (ERA-INTERIM, in this case) are stored. The code will look for files in a sub-directory named after the data product (i.e., `ERA-INTERIM/`).
    - `mod`: where climate model data (CMIP5, LEs, in this case) are stored. The code will look for files in sub-directories named after each climate model (i.e., `CESM1-CAM5-LE`).
    - `proc`: where processed climate data (climate projections, etc.) will be saved. The code will save files in sub-directories named after the historical data product (i.e., `ERA-INTERIM/`), since all calculations are done on projections of that historical data product. Will be generated by the code if this path doesn't exist.  
    - `impact`: where files for projected climate impacts are saved. The code will save files in sub-directories named after the historical data product (i.e., `ERA-INTERIM/`), since all calculations are done on projections of that historical data product. Will be generated by the code if this path doesn't exist. 
    - `aux`: where certain auxiliary files are saved. Will be generated by the code if this path doesn't exist.
    - `geo`: where county-level shapefiles are stored, including `UScounties_proc.shp`, the generated shapefile that includes all historical county-level impact variables.
    - `pop`: where population, GDP per capita, and corn yield data are stored. 
    - `figs`: where figures are saved. Will be generated by the code if this path doesn't exist. 
3. Run through the code step-by-step using `master_run.ipynb` to generate results and figures. 

_NB: the code directory also includes `create_data_counties.ipynb` which details how mortality, GDP, and corn yields were assigned to a single county file. This is just for reference; the county file (`UScounties_proc.shp`) has already been generated and is available at the data link below._

## Data
The base data needed to replicate this study are available [at this Google Drive link](https://drive.google.com/drive/folders/1S3rRy0muI45WGoknPu5dPAU6Iy23RLmx?usp=sharing). All other files can be generated from these data using the code above. 

The Google Drive link contains: 

1. `climate_raw`: All raw climate data used in this study. Data has been downloaded from the ESGF or the Large Ensemble Archive, and preprocessed by stitching together historical and future climate runs into single files. The path to this directory should be used for both `obs` and `mod` in `dir_list.csv`. 
2. `geo_data`: The raw and processed US county shapefiles used in this study. `UScounties_proc.shp` has been pre-generated; otherwise it can be regenerated through the raw county shapefile in this document and the files in `pop_data` using the `create_data_counties.ipynb` notebook. The path to this directory should be used for `geo` in `dir_list.csv`. 
3. `pop_data`: The raw mortality, GDP, and corn yield databases used in this study. See main text for data citations. These are used to generate `UScounties_proc.shp` (which is also included). The path to this directory should be used for `pop` in `dir_list.csv`.


