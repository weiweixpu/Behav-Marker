import os
from PIL import Image

def crop_images(input_folder, output_folder):
    # Check if the output folder exists, create it if not
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through all files in the specified folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            # Construct full paths for the input and output files
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Open the image file
            image = Image.open(input_path)

            # Get the dimensions of the image
            width, height = image.size

            # Calculate the new dimensions after cropping
            new_width = width - 4
            new_height = height - 4

            # Crop the image
            cropped_image = image.crop((2, 2, new_width, new_height))

            # Save the cropped image
            cropped_image.save(output_path)

            print(f"Processed {filename}")

# Specify the paths for the input and output folders
input_folder = r"E:data\Open Field\Tenth\output2"
output_folder = r"E:data\Open Field\Tenth\output3"

# Call the function to crop and save images
crop_images(input_folder, output_folder)