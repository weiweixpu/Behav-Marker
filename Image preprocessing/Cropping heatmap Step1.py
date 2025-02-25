from PIL import Image
import os
import numpy as np

def crop_png_images(input_folder, output_folder):
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.png'):
            file_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)
            img = Image.open(file_path).convert('RGBA')
            img_array = np.array(img)

            # Create a mask for the region with RGB values of (240, 240, 240)
            mask = np.all(img_array[:, :, :3] == [240, 240, 240], axis=2)

            # Find the bounding box coordinates of the region
            coords = np.argwhere(mask)
            min_row, min_col = np.min(coords, axis=0)
            max_row, max_col = np.max(coords, axis=0)

            # Crop the image to the region
            cropped_img = img.crop((min_col, min_row, max_col + 1, max_row + 1))

            # Save the cropped image
            cropped_img.save(output_path)

# Specify the input and output folders
input_folder = r'E:data\Open Field\Tenth\output1'
output_folder = r'E:\data\Open Field\Tenth\output1'

# Call the function to crop the PNG images
crop_png_images(input_folder, output_folder)