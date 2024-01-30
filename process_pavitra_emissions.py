# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process Pavitra Emissions Functions
@author: lucarojasmendoza
last modified: 2023-09-04 on Labor Day
"""

import pandas as pd
import geopandas as gpd
from cmcrameri import cm
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from netCDF4 import Dataset
from shapely.geometry import Point

def plot_ground_elevated(netcdf_file_path, variable_names, borders_gdf_path, target_crs="EPSG:4326", grid_shape=(1057, 1646),
                          x_cell_width=0.05, y_cell_width=0.05, x_origin=39, y_origin=-12.9, columns=2):
    #Obtain borders
    gdf = gpd.read_file(borders_gdf_path)
    gdf = gdf.to_crs(target_crs)

    # Open the netCDF file in read mode
    dataset = nc.Dataset(netcdf_file_path, 'r')

    # Create a figure with subplots arranged in a grid
    fig, axes = plt.subplots(nrows=(len(variable_names) + columns-1) // columns, ncols=columns, figsize=(20, 15))

    # Loop over the variable names and create a subplot for each variable
    for i, variable_name in enumerate(variable_names):
        # Read the variable from the dataset
        variable = dataset.variables[variable_name]

        # Average the data across the first axis
        variable_avg = np.mean(variable, axis=0)

        # Create the x and y coordinate arrays
        x_coords = np.arange(x_origin, x_origin + grid_shape[1] * x_cell_width, x_cell_width)
        y_coords = np.arange(y_origin, y_origin + grid_shape[0] * y_cell_width, y_cell_width)

        # Create the meshgrid of x and y coordinates
        xx, yy = np.meshgrid(x_coords, y_coords)

        # Get the current axis and set the x and y limits based on the grid data
        ax = axes.flat[i]
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())

        # Plot the grid data as an image with the correct x and y scales
        im = ax.imshow(variable_avg[:, :], origin='lower',
                       extent=(x_origin, x_origin + x_cell_width * grid_shape[1],
                               y_origin, y_origin + y_cell_width * grid_shape[0]),
                       cmap=cm.batlow)
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title(variable_name, fontsize=14)
        ax.tick_params(labelsize=14)

        # Add the shapefile plot to the current axis
        gdf.plot(ax=ax, color='none', edgecolor='black')

        # Add the colorbar to the subplot
        cb = plt.colorbar(im, ax=ax, shrink=0.6)
        cb.ax.set_ylabel(variable.units, fontsize=12)
        cb.ax.tick_params(labelsize=12)

    # Close the dataset
    dataset.close()

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()


def process_netCDF_and_save(file_path, variable_names, pollutant, output_path, input_crs="EPSG:4326", target_crs=None,
                             height_elevated=100):
    #extract data
    nc_dataset = Dataset(file_path, "r")
    #splits data in ground and ene and fill is empty values with zeroes
    ground = nc_dataset.variables[variable_names[0]][:]
    ground[ground == -999.0] = 0
    ene = nc_dataset.variables[variable_names[1]][:]
    ene[ene == -999.0] = 0

    latitudes = nc_dataset.variables["lat"][:]
    longitudes = nc_dataset.variables["lon"][:]

    points = []
    for lat in np.array(latitudes):
        for lon in np.array(longitudes):
            point = Point(lon, lat)
            points.append(point)

    gdf = gpd.GeoDataFrame(geometry=points)
    gdf['height'] = 0
    gdf[pollutant] = ground[0].flatten()

    gdf_copy = gdf.copy()
    gdf_copy['height'] = height_elevated
    gdf_copy[pollutant] = ene[0].flatten()

    concatenated_gdf = pd.concat([gdf, gdf_copy], ignore_index=True)

    concatenated_gdf.crs=input_crs

    # If the target CRS is different from the input CRS, convert to the target CRS
    if target_crs:
        concatenated_gdf = concatenated_gdf.to_crs(target_crs)

    # Save the GeoDataFrame to the specified output path with the chosen name
    output_file = output_path # You can change the file format if needed
    concatenated_gdf.to_file(output_file)

    # Return the head of the GeoDataFrame
    return concatenated_gdf.head()


#%%
def increase_resolution_VOC(file_path, output_path):
    #Notice that we are not dividing the values as the units ate kg/m2/s (flux)
    # Read the GeoPandas DataFrame from the file
    gdf = gpd.read_file(file_path)

    #Store new data
    new_rows = []

    for index, row in gdf.iterrows():
        x, y = row['geometry'].x, row['geometry'].y

        # Generate a 5x5 grid around the point
        # Can be generalized later
        for i in range(-2, 3):
            for j in range(-2, 3):
                new_x = x + i * 0.05
                new_y = y + j * 0.05

                # Create a new point geometry
                new_point = Point(new_x, new_y)

                # Create a new row with the original columns
                new_row = {col: row[col] for col in gdf.columns}

                # Update the geometry with the new point
                new_row['geometry'] = new_point

                new_rows.append(new_row)

    # Create a new GeoDataFrame with the generated points
    new_gdf = gpd.GeoDataFrame(new_rows, columns=gdf.columns, crs=gdf.crs)

    # Save the new GeoDataFrame to the output path
    new_gdf.to_file(output_path)

    # Return the head of the GeoDataFrame
    return new_gdf.head()

def process_netCDF_dust_and_save(file_path, variable_names, pollutant, output_path, input_crs="EPSG:4326", target_crs=None):
    #extract data
    #This code works for data with time dimension of 24, where both LAT and LONG have extra dimensions
    nc_dataset = Dataset(file_path, "r")
    #splits data in ground and ene and fill is empty values with zeroes
    ground = nc_dataset.variables[variable_names[0]][:]
    ground[ground == 1e+20] = 0
    ground[ground == 9.969209968386869e+36] = 0
    ground_time_avg = np.mean(ground, axis=0)

    latitudes = nc_dataset.variables["XLAT"][0][:,1]
    longitudes = nc_dataset.variables["XLONG"][0][1,:]

    points = []
    for lat in np.array(latitudes):
        for lon in np.array(longitudes):
            point = Point(lon, lat)
            points.append(point)

    gdf = gpd.GeoDataFrame(geometry=points)
    gdf[pollutant] = ground_time_avg.flatten()

    gdf.crs=input_crs

    # If the target CRS is different from the input CRS, convert to the target CRS
    if target_crs:
        gdf = gdf.to_crs(target_crs)

    # Save the GeoDataFrame to the specified output path with the chosen name
    output_file = output_path # You can change the file format if needed
    gdf.to_file(output_file)

    # Return the head of the GeoDataFrame
    return gdf.head()

def process_netCDF_dust_and_save_2(file_path, variable_names, pollutant, output_path, input_crs="EPSG:4326", target_crs=None):
    #extract data
    #This code works for data with time dimension of 1, processed using montly files
    nc_dataset = Dataset(file_path, "r")
    #splits data in ground and ene and fill is empty values with zeroes
    ground = nc_dataset.variables[variable_names[0]][:]
    ground[ground == 1e+20] = 0
    ground[ground == 9.969209968386869e+36] = 0
    ground_time_avg = np.mean(ground, axis=0)

    latitudes = nc_dataset.variables["XLAT"][:]
    longitudes = nc_dataset.variables["XLONG"][:]

    points = []
    for lat in np.array(latitudes):
        for lon in np.array(longitudes):
            point = Point(lon, lat)
            points.append(point)

    gdf = gpd.GeoDataFrame(geometry=points)
    gdf[pollutant] = ground_time_avg.flatten()

    gdf.crs=input_crs

    # If the target CRS is different from the input CRS, convert to the target CRS
    if target_crs:
        gdf = gdf.to_crs(target_crs)

    # Save the GeoDataFrame to the specified output path with the chosen name
    output_file = output_path # You can change the file format if needed
    gdf.to_file(output_file)

    # Return the head of the GeoDataFrame
    return gdf.head()

def process_netCDF_wrfground_and_save(file_path, variable_names, pollutant, output_path, input_crs="EPSG:4326", target_crs=None):
    #extract data
    nc_dataset = Dataset(file_path, "r")
    ground = nc_dataset.variables[variable_names[0]][:]
    ground[ground == 1e+20] = 0
    ground_time_avg = np.mean(ground, axis=0)

    latitudes = nc_dataset.variables["lat"][:]
    longitudes = nc_dataset.variables["lon"][:]

    points = []
    for lat in np.array(latitudes):
        for lon in np.array(longitudes):
            point = Point(lon, lat)
            points.append(point)

    gdf = gpd.GeoDataFrame(geometry=points)
    gdf[pollutant] = ground_time_avg.flatten()

    gdf.crs=input_crs

    # If the target CRS is different from the input CRS, convert to the target CRS
    if target_crs:
        gdf = gdf.to_crs(target_crs)

    # Save the GeoDataFrame to the specified output path with the chosen name
    output_file = output_path # You can change the file format if needed
    gdf.to_file(output_file)

    # Return the head of the GeoDataFrame
    return gdf.head()

def plot_ground_elevated_shapefile(shapefile_path, pollutant, borders_path, target_crs="EPSG:4326"):
    # Read the shapefile and borders data, and reproject to the target CRS
    gdf = gpd.read_file(shapefile_path)
    gdf = gdf.to_crs(target_crs)

    border_gdf = gpd.read_file(borders_path)
    border_gdf = border_gdf.to_crs(target_crs)

    # Define the figure size and create subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))

    # Plot the pollutant points on the first subplot (height values of 0)
    ground_layer = gdf[gdf['height'] == 0]
    ground_layer.plot(ax=axes[0], marker='o', markersize=50, label='Ground Layer',
                      column=pollutant, cmap=cm.batlow)
    axes[0].set_title("Ground Layer")

    # Plot the pollutant points on the second subplot (height values of 100)
    elevated_layer = gdf[gdf['height'] == 100]
    elevated_layer.plot(ax=axes[1], marker='o', markersize=50, label='Elevated Layer',
                        column=pollutant, cmap=cm.batlow)
    axes[1].set_title("Elevated Layer")

    # Plot the country borders on both subplots
    for ax in axes:
        border_gdf.plot(ax=ax, edgecolor='black', color='none')

    # Set x and y limits based on NH3 point values for both subplots
    for ax in axes:
        x_min, y_min, x_max, y_max = gdf.geometry.total_bounds
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.legend(loc="upper right")

    # Show the plots
    plt.tight_layout()
    plt.show()


def plot_inmap_wrf_bias(gdf, borders_path, target_crs, inmap_column_name, wrf_column_name, bias_column_name):
    # Load and convert 'border_gdf' to the specified CRS
    border_gdf = gpd.read_file(borders_path)
    border_gdf = border_gdf.to_crs(target_crs)

    # Define the color scale range based on the minimum and maximum values of the two columns
    vmin = gdf[inmap_column_name].min()
    vmax = gdf[inmap_column_name].max()
    vmax_bias = max(gdf[bias_column_name].max(), abs(gdf[bias_column_name].min()))

    # Create subplots with 1 row and 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot the specified column from gdf on the left subplot with the 'batlow' colormap
    gdf.plot(column=inmap_column_name, cmap=cm.batlow, legend=True, ax=axes[0], vmin=vmin, vmax=vmax)
    axes[0].set_title(f'{inmap_column_name} InMAP ug/m3')

    # Plot the specified column from gdf on the middle subplot with the 'batlow' colormap
    gdf.plot(column=wrf_column_name, cmap=cm.batlow, legend=True, ax=axes[1], vmin=vmin, vmax=vmax)
    axes[1].set_title(f'{wrf_column_name} WRFCHEM ug/m3')

    # Plot the specified column from gdf on the right subplot with the 'vik' colormap for bias
    gdf.plot(column=bias_column_name, cmap=cm.vik, legend=True, ax=axes[2], vmin=-vmax_bias, vmax=vmax_bias)
    axes[2].set_title('Bias Inmap - WRF')

    # Plot the country borders on all subplots
    for ax in axes:
        border_gdf.plot(ax=ax, edgecolor='black', color='none')

    # Set x and y limits based on point values for all subplots
    for ax in axes:
        x_min, y_min, x_max, y_max = gdf.geometry.total_bounds
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    # Adjust layout
    plt.tight_layout()

    # Show the plots
    plt.show()
