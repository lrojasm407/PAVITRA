# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SGRIDS Functions
@author: lucarojasmendoza
last modified: 2023-09-04
"""

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import box
from time import time
from cmcrameri import cm


def sgrids(dems, target_crs, VariableGridXo, VariableGridYo, VariableGridDxy, Xnests, Ynests, totalpop_threshold, density_threshold,plots=True):
    # Measure execution time
    start_time = time()

    # Convert to target CRS
    dems = dems.to_crs(target_crs)
    dems['dems_area'] = dems.geometry.area

    # Calculate the bounding box of the grid
    xmin = VariableGridXo
    ymin = VariableGridYo
    xmax = xmin + VariableGridDxy * Xnests[0]
    ymax = ymin + VariableGridDxy * Ynests[0]

    # Create a geopandas dataframe with one row for each cell in the grid
    rows = []
    for i in range(Xnests[0]):
        for j in range(Ynests[0]):
            x_left = xmin + i * VariableGridDxy
            y_top = ymax - j * VariableGridDxy
            rows.append({'geometry': box(x_left, y_top - VariableGridDxy, x_left + VariableGridDxy, y_top)})
    gdf = gpd.GeoDataFrame(rows, crs=target_crs)

    # First iteration
    # perform spatial join between gdf and dems
    # create a new geodataframe with all the intersections between gdf and dems
    gdf['index'] = gdf.index
    intersections = gpd.overlay(gdf, dems, how='intersection')
    # Calculate the area of each grid cell, reallocate population and grouped by gdf index
    intersections['area'] = intersections.geometry.area
    intersections['TotalPop_2'] = intersections.apply(lambda x: x['area'] * x['TotalPop'] / x['dems_area'], axis=1)
    intersections['density'] = intersections['TotalPop_2'] / intersections['area']
    grouped = intersections.groupby('index').agg({'TotalPop_2': 'sum', 'density': 'max'}).reset_index()
    # Merge the grouped dataframe back into the gdf dataframe based on the index_right column
    gdf = gdf.merge(grouped, on='index', how='left')
    gdf['TotalPop_2'] = gdf['TotalPop_2'].fillna(0)
    gdf['density'] = gdf['density'].fillna(0)
    gdf.rename(columns={'TotalPop_2': 'TotalPop'}, inplace=True)
    gdf['area'] = gdf.geometry.area
    gdf['length'] = np.sqrt(gdf['area'])

    # Create summary table
    table=sgrids_table(gdf)
    if plots:
        # Create a figure with two subplots
        plot_population_data(intersections, gdf, table)
    else:
        display(table)

    #Following iterations
    for i in range(1, len(Xnests)):
        # Define the number of rows and columns for the new grid
        nrows = Xnests[i]
        # Create an empty list to hold the new polygons
        new_polygons = []
        # Iterate over the polygons in the geometry column
        for polygon in gdf['geometry']:
            # Check the TotalPop and density values for the polygon
            if gdf.loc[gdf['geometry'] == polygon, 'TotalPop'].values[0] > totalpop_threshold or \
                    gdf.loc[gdf['geometry'] == polygon, 'density'].values[0] > density_threshold:
                # Divide the polygon into smaller squares
                xmin, ymin, xmax, ymax = polygon.bounds
                dx = (xmax - xmin) / nrows
                dy = (ymax - ymin) / nrows
                for x in range(nrows):
                    for y in range(nrows):
                        new_polygon = box(xmin + x * dx, ymin + y * dy, xmin + (x + 1) * dx, ymin + (y + 1) * dy)
                        new_polygons.append(new_polygon)
            else:
                # If the condition is not met, add the original polygon to the new list
                new_polygons.append(polygon)

        # Create a new GeoDataFrame from the new polygons
        new_gdf = gpd.GeoDataFrame(geometry=new_polygons, crs=gdf.crs)
        gdf = new_gdf

        # perform spatial join between gdf and dems
        # create a new geodataframe with all the intersections between gdf and dems
        gdf['index'] = gdf.index
        intersections = gpd.overlay(gdf, dems, how='intersection')
        # Calculate the area of each grid cell, reallocate population and grouped by gdf index
        intersections['area'] = intersections.geometry.area
        intersections['TotalPop_2'] = intersections.apply(lambda x: x['area'] * x['TotalPop'] / x['dems_area'], axis=1)
        intersections['density'] = intersections['TotalPop_2'] / intersections['area']
        grouped = intersections.groupby('index').agg({'TotalPop_2': 'sum', 'density': 'max'}).reset_index()
        # Merge the grouped dataframe back into the gdf dataframe based on the index_right column
        gdf = gdf.merge(grouped, on='index', how='left')
        gdf['TotalPop_2'] = gdf['TotalPop_2'].fillna(0)
        gdf['density'] = gdf['density'].fillna(0)
        gdf.rename(columns={'TotalPop_2': 'TotalPop'}, inplace=True)
        gdf['area'] = gdf.geometry.area
        gdf['length'] = np.sqrt(gdf['area'])

        # Create summary table
        table = sgrids_table(gdf)
        if plots:
        # Create a figure with two subplots
            plot_population_data(intersections, gdf, table)
        else:
            display(table)

    # Calculate and print execution time
    end_time = time()
    execution_time = end_time - start_time
    print("Execution time: %.4f seconds" % execution_time)

    return gdf

def sgrids_table(gdf):
    if 'length' not in gdf.columns:
        gdf['length'] = np.sqrt(gdf.geometry.area)

    # Create a list of unique grid lengths
    unique_grid_lengths = gdf['length'].unique()
    # Create a dictionary to store the table data
    table_data = {}
    # Loop over the unique grid lengths and compute the counts
    for grid_length in unique_grid_lengths:
        count = gdf[gdf['length'] == grid_length].shape[0]
        table_data[grid_length] = [count]
    # Create a DataFrame from the dictionary
    table = pd.DataFrame.from_dict(table_data, orient='index', columns=['Number of grids'])
    # Sort the index in descending order
    table = table.sort_index(axis=0, ascending=False)
    # Add the totals row to the DataFrame
    totals = table.sum(axis=0)
    table.loc['totals'] = totals
    return table

def plot_population_data(intersections, gdf, table):
    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    # Plot the second subplot on the left
    intersections.plot(ax=axs[0], column='TotalPop_2', cmap='viridis', edgecolor='black')
    axs[0].set_title('Population intersections')
    # Plot the first subplot on the right
    gdf.plot(ax=axs[1], column='TotalPop', cmap='viridis', edgecolor='black')
    axs[1].set_title('Population redistribution')

    # Hide the default axis
    axs[2].axis('off')
    # Create the table
    # Create the table with minimal column width
    col_width = [0.5] * len(table.columns)
    axs[2].table(cellText=table.values, colLabels=table.columns, rowLabels=table.index,
                 loc='center', cellLoc='center', colWidths=col_width)

    # Show the plot
    plt.show()

    print('TotalPop:', gdf["TotalPop"].sum())


def visualize_data(dems_gdf, target_crs, VariableGridXo, VariableGridYo, VariableGridDxy, Xnests, Ynests):
    # Measure execution time
    start_time = time()

    # Convert to target CRS
    dems_gdf = dems_gdf.to_crs(target_crs)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 12))
    dems_gdf.plot(ax=ax, edgecolor='black', facecolor='none')
    ax.set_xlim(VariableGridXo, VariableGridXo + VariableGridDxy * Xnests[0])
    ax.set_ylim(VariableGridYo, VariableGridYo + VariableGridDxy * Ynests[0])

    # Calculate and print execution time
    end_time = time()
    execution_time = end_time - start_time
    print("Execution time: %.4f seconds" % execution_time)

    # Show the plot
    plt.show()

def plot_with_scalebar(grid_gdf, region_gdf, community_gdf, VariableGridXo,VariableGridYo,VariableGridDxy, x,
                       y, figsize=(12, 12), target_crs=None, title=None):
    if 'length' not in grid_gdf.columns:
        grid_gdf['length'] = np.sqrt(grid_gdf.geometry.area)

    if target_crs:
        grid_gdf = grid_gdf.to_crs(target_crs)
        region_gdf = region_gdf.to_crs(target_crs)
        community_gdf = community_gdf.to_crs(target_crs)

    fig, ax = plt.subplots(figsize=figsize)
    grid_gdf.plot(column='length', ax=ax, edgecolor='gray', categorical=True, legend=True, cmap=cm.hawaii)
    region_gdf.plot(ax=ax, edgecolor='black', facecolor='none')
    community_gdf.plot(ax=ax, edgecolor='red', facecolor='none')

    ax.set_xlim(VariableGridXo, VariableGridXo+VariableGridDxy*x)
    ax.set_ylim(VariableGridYo, VariableGridYo+VariableGridDxy*y)

    ax.set_title(title)
    #scalebar = ScaleBar(1, "m", length_fraction=0.3, location="lower left")
    #ax.add_artist(scalebar)
    plt.show()

def plot_ab617(grid_gdf, region_gdf, community_gdf, figsize=(12, 12),target_crs=None, title=None):
    if target_crs:
        grid_gdf = grid_gdf.to_crs(target_crs)
        region_gdf = region_gdf.to_crs(target_crs)
        community_gdf = community_gdf.to_crs(target_crs)
    num_rows = len(community_gdf.index)

    fig, axs = plt.subplots(3, int(num_rows/3), figsize=figsize, tight_layout=True)
    fig.suptitle(title)

      # Get the number of rows in ab617_gdf

    for i in range(num_rows):
        # Calculate the indices for subplot position
        row_index = i // 5
        col_index = i % 5

        # Create a subplot for the current row
        ax = axs[row_index, col_index]

        # Remove tick marks and labels from both axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # Plot the current row of ab617_gdf without thick marks or values
        grid_gdf.plot(column='length', ax=ax, edgecolor='gray', categorical=True, cmap=cm.hawaii)
        region_gdf.plot(ax=ax, edgecolor='black', facecolor='none')

        # Create a new GeoDataFrame with the desired row
        row_gdf = gpd.GeoDataFrame(community_gdf.iloc[[i]])

        # Plot the current row's geometry
        row_gdf.plot(ax=ax, edgecolor='red', facecolor='none')

        # Set the limits based on the bounding box of the current row's geometry
        ax.set_xlim(row_gdf.total_bounds[0], row_gdf.total_bounds[2])
        ax.set_ylim(row_gdf.total_bounds[1], row_gdf.total_bounds[3])

        # Add a title for the current subplot using the 'code' column from the row
        title = community_gdf['code'].iloc[i]
        ax.set_title(title, fontsize=15)

        # Add a scale bar to the plot
        scalebar = ScaleBar(1, "m", length_fraction=0.25, location="lower right")
        ax.add_artist(scalebar)

    plt.show()



def create_summary_table(grid_gdf, community_gdf, index, index_2=None):
    # Perform the overlay operation
    overlay_gdf = gpd.overlay(grid_gdf, community_gdf, how='intersection')

    # Perform pivot operation on overlay_gdf by length
    length_pivot = overlay_gdf.pivot_table(index=index, columns='length', aggfunc='size', fill_value=0)

    # Calculate the total by community
    length_pivot['Total Cells'] = length_pivot.sum(axis=1)

    # Calculate the population-weighted length
    overlay_gdf['PopulationWeightedLength'] = overlay_gdf['length'] * overlay_gdf['TotalPop']

    # Group by community and calculate the sum of population-weighted length and total population
    summary_table = overlay_gdf.groupby(index).agg({
        'PopulationWeightedLength': 'sum',
        'TotalPop': 'sum',
    }).reset_index()

    # Calculate the population-averaged length scale
    summary_table['PopAveLengthScale'] = (summary_table['PopulationWeightedLength'] / summary_table['TotalPop']).round(
        1)

    # Drop columns
    summary_table = summary_table.drop(['PopulationWeightedLength', 'TotalPop'], axis=1)

    summary_table = summary_table.merge(length_pivot, on=index)

    if index_2:
        summary_table = summary_table.merge(community_gdf[[index, index_2]],on=index)

    return summary_table



