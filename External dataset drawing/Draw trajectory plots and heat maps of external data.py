import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from colors import colors
"""
    This script processes animal tracking data from an external data CSV format file, generating trackmaps and heatmaps
for each animal based on geographic coordinates.
    The data is preprocessed to remove invalid or missing values, sorted by timestamp.
    The trackmap visualizes movement paths, while the heatmap highlights high activity areas using a custom colormap and
Gaussian smoothing.
    All generated images are saved in organized folders.
"""
# Read the CSV file
csv_path = r"G:\Trajectory Dataset\Animal Dataset\Black-legged Kittiwake Rissa tridactyla Middleton Island.csv"
step = 100
df = pd.read_csv(csv_path)

# Remove rows with missing longitude or latitude
df.dropna(subset=['location-long', 'location-lat'], inplace=True)

# Get unique animal identifiers
unique_animals = df['tag-local-identifier'].unique()

# Get the file and folder paths
csv_filename = os.path.basename(csv_path)
base_folder_path = os.path.dirname(csv_path)
trajectory_folder = os.path.join(base_folder_path, csv_filename.split('.')[0], 'trajectories')
heatmap_folder = os.path.join(base_folder_path, csv_filename.split('.')[0], 'heatmaps')

# Create folders for saving trajectories and heatmaps
os.makedirs(trajectory_folder, exist_ok=True)
os.makedirs(heatmap_folder, exist_ok=True)

# Define a custom colormap
cmap = LinearSegmentedColormap.from_list('custom_cmap', [(r/255, g/255, b/255) for r, g, b in colors])

# Process data for each animal
for animal in unique_animals:
    current_animal_data = df[df['tag-local-identifier'] == animal].copy()

    # Skip if no data exists for the animal
    if current_animal_data.empty:
        continue

    # Sort data by timestamp
    current_animal_data.sort_values(by='timestamp', inplace=True)

    # Remove invalid data
    current_animal_data.dropna(subset=['location-long', 'location-lat'], inplace=True)
    current_animal_data = current_animal_data[np.isfinite(current_animal_data['location-long']) & np.isfinite(current_animal_data['location-lat'])]

    # Determine the number of trajectory images to generate
    num_points = len(current_animal_data)
    num_images = max(1, num_points // step + (num_points % step > 0))

    # Generate trajectory plots
    for i in range(num_images):
        fig, ax = plt.subplots(figsize=(6, 6))
        start_idx = i * step
        end_idx = min((i + 1) * step, num_points)

        ax.plot(current_animal_data['location-long'].iloc[start_idx:end_idx].to_numpy(),
                current_animal_data['location-lat'].iloc[start_idx:end_idx].to_numpy(),
                color='#854085')

        ax.axis('off')
        plt.savefig(f'{trajectory_folder}/{animal}_{i + 1}.png', format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    # Generate heat maps
    bins = np.array_split(range(num_points), num_images)
    for i, bin_indices in enumerate(bins, start=1):
        bin_data = current_animal_data.iloc[bin_indices]
        x_edges = np.linspace(bin_data['location-long'].min(), bin_data['location-long'].max(), 100)
        y_edges = np.linspace(bin_data['location-lat'].min(), bin_data['location-lat'].max(), 100)

        hist, xedges, yedges = np.histogram2d(
            bin_data['location-long'], bin_data['location-lat'],
            bins=[x_edges, y_edges]
        )
        hist_smoothed = gaussian_filter(hist, sigma=2.5)

        fig, ax = plt.subplots(figsize=(6, 6))
        cax = ax.imshow(hist_smoothed.T, interpolation='bilinear', cmap=cmap, origin='lower',
                        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto')

        ax.axis('off')

        image_path = f'{heatmap_folder}/{animal}_{i}.png'
        plt.savefig(image_path, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        print(f"Saved {image_path}")
