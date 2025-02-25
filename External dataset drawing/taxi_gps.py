import numpy as np
from scipy.ndimage import gaussian_filter
import os
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from colors import colors

# Initialize a dictionary to store the latitude and longitude data of each vehicle
track_data = {}

# Read data from a text file
with open('G:\\Trajectory Dataset\\Urban Data Release\\Taxi GPS.txt', 'r') as file:
    for line in file:
        parts = line.strip().split(',')
        if len(parts) >= 5:
            vehicle_id, _, lon, lat, _, _ = parts
            lon, lat = float(lon), float(lat)
            if vehicle_id not in track_data:
                track_data[vehicle_id] = {'lons': [], 'lats': []}
            track_data[vehicle_id]['lons'].append(lon)
            track_data[vehicle_id]['lats'].append(lat)


# Create folders for tracking and heatmap images if they do not exist
tracking_folder = 'G:\\Tracking Dataset\\Urban Data Release\\Taxi GPS\\Tracking'
heatmap_folder = 'G:\\Tracking Dataset\\Urban Data Release\\Taxi GPS\\Heatmaps'

os.makedirs(tracking_folder, exist_ok=True)
os.makedirs(heatmap_folder, exist_ok=True)

# Plot trajectory and corresponding heatmap for each vehicle
for vehicle_id, coords in track_data.items():
    num_points = len(coords['lons'])

    # Process data in batches if the number of points is large
    step = 100
    num_batches = (num_points // step) + (num_points % step != 0)

    for batch in range(num_batches):
        start = batch * step
        end = min((batch + 1) * step, num_points)

        # Plot trajectory
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.plot(coords['lons'][start:end], coords['lats'][start:end], marker='', linestyle='-', color='#854085')
        plt.axis('equal')
        plt.axis('off')

        # Save trajectory plot
        tracking_filename = f'{tracking_folder}\\{vehicle_id}_{batch + 1}.png'
        plt.savefig(tracking_filename, bbox_inches='tight', pad_inches=0, facecolor='white', edgecolor='none')
        plt.close(fig)

        # Create a colormap using the provided color values
        colors = [(r / 255, g / 255, b / 255) for r, g, b in colors]  # Convert RGB values from 0-255 to 0-1
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)

        # Create and save heatmap
        fig, ax = plt.subplots(figsize=(6, 6))
        hist, xedges, yedges = np.histogram2d(coords['lons'][start:end], coords['lats'][start:end], bins=[100, 100])
        hist_smoothed = gaussian_filter(hist, sigma=2.5)
        ax.imshow(hist_smoothed.T, origin='lower', cmap= cmap, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto')
        plt.axis('off')

        # Save heatmap plot
        heatmap_filename = f'{heatmap_folder}\\{vehicle_id}_{batch + 1}.png'
        plt.savefig(heatmap_filename, bbox_inches='tight', pad_inches=0, facecolor='white', edgecolor='none')
        plt.close(fig)

        print(f"Saved tracking and heatmap images for {vehicle_id}, batch {batch + 1}")


