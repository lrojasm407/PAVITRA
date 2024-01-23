# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process Pavitra Emissions
@author: lucarojasmendoza
last modified: 2024-01-21
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from shapely.geometry import Point
import matplotlib.pyplot as plt
from cmcrameri import cm
import netCDF4 as nc
import xarray as xr
import process_pavitra_emissions
from process_pavitra_emissions import plot_ground_elevated


#%%
#Define the source CRS and the data corners in lat long
#Modify if needed
source_crs = "+proj=merc +units=m +a=6370000.0 +b=6370000. +lon_0=80.0 +lat_ts=15.0"
CtmGridXo = 39.01822
CtmGridYo = -12.875168

# Calculate corner points
corner = [(CtmGridXo, CtmGridYo)]
# Create a GeoPandas DataFrame
geometry_corner = [Point(x, y) for x, y in corner]
corner_gdf = gpd.GeoDataFrame(geometry=geometry_corner, columns=['geometry'])
# Assign the source CRS
corner_gdf.crs = "EPSG:4326"
corner_gdf = corner_gdf.to_crs(source_crs)


#%%

# Create a geopandas dataframe with 4 corners
# using xo/yo coordinates for WRF output data
data = {
    'x0': -4401000.274,
    'y0': -1394439.5,
    'dx': 27000.0,
    'dy': 27000.0,
    'nx': 327,
    'ny': 225
}
# Calculate corner points
corners = [
    (data['x0'], data['y0']),
    (data['x0'] + data['nx'] * data['dx'], data['y0']),
    (data['x0'] + data['nx'] * data['dx'], data['y0'] + data['ny'] * data['dy']),
    (data['x0'], data['y0'] + data['ny'] * data['dy'])
]

# Create a GeoPandas DataFrame
geometry = [Point(x, y) for x, y in corners]
geo_df = gpd.GeoDataFrame(geometry=geometry, columns=['geometry'])

# Assign the source CRS
geo_df.crs = source_crs

# Convert to target CRS (EPSG:4326)
target_crs ="EPSG:4326"
geo_df = geo_df.to_crs(target_crs)

#%%
# Determine the range of indexes in nc files that fall within the 4 corners.
# This is important to determine how to subset nc files for more efficient handling of the files
# Emission Files contain information for the entire world
# The first step is to extract latitude and latitude values
file_path = "/Volumes/InMAP-shared/PAVITRA_data/Year_Emissions/SMOG_V2_2019_Merged/NH3_anthro_201901-201912_merged.nc"

# Open the NetCDF file using Dask
dataset = nc.Dataset(file_path, 'r', format='NETCDF4')

# Extract values of latitude and longitude
lat_values = np.array(dataset.variables['lat'][:])
lon_values = np.array(dataset.variables['lon'][:])

# Close the dataset when you're done
dataset.close()

# Define the range of longitude and latitude values you're interested in
min_lon = 39.01822
max_lon = 121.23320
min_lat = -12.87517
max_lat = 39.90304

# Find the index of the nearest value to the specified longitudes and latitudes
min_lon_index = np.argmin(np.abs(lon_values - min_lon))
max_lon_index = np.argmin(np.abs(lon_values - max_lon))
min_lat_index = np.argmin(np.abs(lat_values - min_lat))
max_lat_index = np.argmin(np.abs(lat_values - max_lat))

# Print the indexes and values at the selected indexes
print("Index and Value for min longitude:", min_lon_index, lon_values[min_lon_index])
print("Index and Value for max longitude:", max_lon_index, lon_values[max_lon_index])
print("Index and Value for min latitude:", min_lat_index, lat_values[min_lat_index])
print("Index and Value for max latitude:", max_lat_index, lat_values[max_lat_index])

#%%
# Determine the range of indexes in nc files that fall within the 4 corners for VOCs
# The first step is to extract latitude and latitude values
file_path = "/Volumes/InMAP-shared/PAVITRA_data/Year_Emissions/NMVOCs_2015/ALD_anthro_COALESCE_201501-201512.nc"

# Open the NetCDF file using Dask
dataset = nc.Dataset(file_path, 'r', format='NETCDF4')

# Extract values of latitude and longitude
lat_values = np.array(dataset.variables['lat'][:])
lon_values = np.array(dataset.variables['lon'][:])

# Close the dataset when you're done
dataset.close()

# Define the range of longitude and latitude values you're interested in
min_lon = 39.01822
max_lon = 121.23320
min_lat = -12.87517
max_lat = 39.90304

# Find the index of the nearest value to the specified longitudes and latitudes
min_lon_index = np.argmin(np.abs(lon_values - min_lon))
max_lon_index = np.argmin(np.abs(lon_values - max_lon))
min_lat_index = np.argmin(np.abs(lat_values - min_lat))
max_lat_index = np.argmin(np.abs(lat_values - max_lat))

# Print the indexes and values at the selected indexes
print("Index and Value for min longitude:", min_lon_index, lon_values[min_lon_index])
print("Index and Value for max longitude:", max_lon_index, lon_values[max_lon_index])
print("Index and Value for min latitude:", min_lat_index, lat_values[min_lat_index])
print("Index and Value for max latitude:", max_lat_index, lat_values[max_lat_index])

'''
# Modify NC files using Bash. 

** Clip the data to the lat/lon of interest

** Merge ground emissions into one single variable called 'ground'

** Extract energy ('ene') emissions as separate variable (since energy emissions are assigned at layer 2. 

** there is not ufd for VOCs, and there is not trb for other emissions (use WST Instead)'''

#%%%
file_path = "/Volumes/lrojasm/2019_emissions_ground_elevated/BC_anthro_201901-201912_merged_average.nc"

# Open the NetCDF file using Dask
dataset = nc.Dataset(file_path, 'r', format='NETCDF4')

# Iterate through variables in the NetCDF file
for var_name in dataset.variables.keys():
    variable = dataset.variables[var_name]

    # Get the dimensions of the Dask array
    dimensions = variable.dimensions

    # Get the size of each dimension
    dimension_sizes = [len(dataset.dimensions[dim]) for dim in dimensions]

    # Print variable name, dimensions, and their sizes
    print(f"Variable: {var_name}, Dimensions: {dimensions}, Dimension Sizes: {dimension_sizes}")

# Extract values of latitude and longitude
lat_values = dataset.variables['lat'][:]
lon_values = dataset.variables['lon'][:]

# Close the dataset when you're done
dataset.close()

#%%
#Which hight will always result in inputing emissions at a given layer?
grid_wrf=xr.open_dataset('../WRF_INMAP_INDIA/inmap_wrfchem_20171201.ncf')
grid_wrf

#%%
dataset = nc.Dataset('../WRF_INMAP_INDIA/inmap_wrfchem_20171201.ncf', 'r')
# Get the variable names
variable_names = dataset.variables.keys()
list_C = list(variable_names)

summary_table = pd.DataFrame(columns=['Variable', 'Min', 'Max', 'NaN Count', 'Zero Count'])

for i, variable_name in enumerate(list_C):
    # Read the variable from the dataset
    ground_layer = dataset.variables[list_C[i]][0]
    # Compute statistics
    variable_min = np.nanmin(ground_layer)
    variable_max = np.nanmax(ground_layer)
    nan_count = np.isnan(ground_layer).sum()
    zero_count = np.count_nonzero(ground_layer == 0)
    # Append to the summary table
    summary_table = pd.concat([summary_table, pd.DataFrame({'Variable': [variable_name],
                                                            'Min': [variable_min],
                                                            'Max': [variable_max],
                                                            'NaN Count': [nan_count],
                                                            'Zero Count': [zero_count]})],
                              ignore_index=True)

# Show the summary table
summary_table

#%%
layer=1
# Find minimum and maximum values
print('layer 1')
min_value = np.min(dataset.variables['LayerHeights'][layer])
max_value = np.max(dataset.variables['LayerHeights'][layer])
print("Minimum value:", min_value)
print("Maximum value:", max_value)
layer=2
print('layer 2')
min_value = np.min(dataset.variables['LayerHeights'][layer])
max_value = np.max(dataset.variables['LayerHeights'][layer])
print("Minimum value:", min_value)
print("Maximum value:", max_value)
layer=3
print('layer 3')
min_value = np.min(dataset.variables['LayerHeights'][layer])
max_value = np.max(dataset.variables['LayerHeights'][layer])
print("Minimum value:", min_value)
print("Maximum value:", max_value)