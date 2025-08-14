# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 23:36:26 2024

@author: wuzho
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

# Set global font properties
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14

# Define custom colormapa
color = [(0, 0, 1), (0.3, 0.5, 1), (0, 1, 0), (1, 1, 0), (1, 0.5, 0), (1, 0, 0)]  # Blue to red
colors = color[::-1]
cmap = LinearSegmentedColormap.from_list('CustomMap', colors, N=256)


# Define the function to draw maps
def draw_map(ax, data, column, title):
    LONLIMS = [104.25,104.27]
    LATLIMS = [30.29,30.33]

    # Create a Basemap instance
    m = Basemap(projection='tmerc', lon_0=(LONLIMS[0] + LONLIMS[1]) / 2, lat_0=(LATLIMS[0] + LATLIMS[1]) / 2,
                llcrnrlon=LONLIMS[0], llcrnrlat=LATLIMS[0], urcrnrlon=LONLIMS[1], urcrnrlat=LATLIMS[1], resolution='h',
                ax=ax)

    # Fill the map background and continents
    m.drawmapboundary(fill_color='white')
    m.fillcontinents(color='white', lake_color='lightblue', ax=ax)

    # Draw latitude and longitude lines
    parallels = np.linspace(LATLIMS[0], LATLIMS[1], 5)
    meridians = np.linspace(LONLIMS[0], LONLIMS[1], 5)
    m.drawparallels(parallels, labels=[True, False, False, True], color='gray', fontsize=7, zorder=1, ax=ax, fmt='%.2f',
                    rotation=90)
    m.drawmeridians(meridians, labels=[True, False, False, True], color='gray', fontsize=7, zorder=1, ax=ax, fmt='%.2f')

    # 绘制江苏省边界轮廓
    shp_data.boundary.plot(ax=ax, edgecolor='black', linewidth=1)

    # Plot data points within the shapefile extent
    x, y = m(data.geometry.x, data.geometry.y)
    scatter = m.scatter(x, y, c=data[column], cmap=cmap, s=0.5, zorder=2, ax=ax, vmin=10, vmax=20,marker='s')

    # Set title
    ax.set_title(title, fontsize=14, weight='bold')

    return scatter


# Read the shapefile
shp_data = gpd.read_file('湖边界.shp')

# Read prediction depth data (every 100th row)
data = pd.read_csv('result-.csv')
data = data.iloc[::100, :]

# Convert data['Longitude'] and data['Latitude'] to a GeoDataFrame
data_gdf = gpd.GeoDataFrame(
    data,
    geometry=gpd.points_from_xy(data['Longitude'], data['Latitude'])
)

# Assign CRS to data_gdf if it is None
if data_gdf.crs is None:
    data_gdf.crs = 'EPSG:4326'

# Ensure both data_gdf and shp_data have compatible CRS (if necessary)
data_gdf = data_gdf.to_crs(shp_data.crs)

# Perform spatial join to keep only data points within the shapefile extent
clipped_data = gpd.sjoin(data_gdf, shp_data, predicate='within')

# Create the figure and GridSpec with adjusted aspect ratios
fig = plt.figure(figsize=(10, 8))  # Adjusted figure size
gs = GridSpec(2, 5, width_ratios=[1] * 4 + [0.08], hspace=0.3, wspace=0.2)  # Adjusted width_ratios and wspace

# Set equal aspect ratio for the first four axes
for ax in fig.axes[:4]:
    ax.set_aspect('equal')

# Create axes and draw maps for each model
axes = [fig.add_subplot(gs[i // 2, i % 2 * 2]) for i in range(4)]
scatters = [draw_map(axes[i], clipped_data, col, title) for i, (col, title) in enumerate([
    ('stumpf', 'Stumpf Model'),
    ('log-linear', 'Log Linear Model'),
    ('random_forest_prediction', 'Random Forest Model'),
    ('random_forest_depth', 'RSG-RFR Model')])]


# Add colorbars
for i in range(2):
    cbar_ax = fig.add_subplot(gs[i, 4])
    cbar = fig.colorbar(scatters[i * 2], cax=cbar_ax, orientation='vertical')
    cbar.set_label('Depth (m)', labelpad=10)
    cbar.set_ticks(np.arange(10, 21, 5))
    cbar.ax.set_yticklabels(np.arange(10, 25, 5), fontsize=10)
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('right')
    cbar.ax.yaxis.set_tick_params(labelleft=False, left=False)

# Adjust layout
plt.tight_layout()

plt.savefig("water0001.jpg", dpi=1800)

#plt.show()