import os
from PIL import Image
from util.colors import colors

def calculate_distance(pixel1, pixel2):
    r1, g1, b1 = pixel1
    r2, g2, b2 = pixel2
    return abs(r1 - r2) + abs(g1 - g2) + abs(b1 - b2)


def process_images(input_folder, output_folder, rgb_values):
    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            # Construct full paths to input and output files
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Open the image file and convert it to RGB mode
            image = Image.open(input_path).convert('RGB')
            pixels = image.load()

            # Iterate over each pixel of the image
            width, height = image.size
            for x in range(width):
                for y in range(height):
                    # Get the RGB value of the current pixel
                    r, g, b = pixels[x, y]

                    # Find the corresponding pixel value index
                    if (r, g, b) in rgb_values:
                        index = rgb_values.index((r, g, b))
                    else:
                        # If the pixel value is not in the list, find the closest pixel value
                        closest_pixel = min(rgb_values, key=lambda p: calculate_distance((r, g, b), p))
                        index = rgb_values.index(closest_pixel)

                    # Calculate t and s values
                    t = round(index * 123 / 1021)
                    s = round(t * 1021 / 220)

                    # Get the RGB value to be replaced
                    new_r, new_g, new_b = rgb_values[s]

                    # Replace the RGB value of the current pixel
                    pixels[x, y] = (new_r, new_g, new_b)

            # Save the modified image
            image.save(output_path)


# Specify input folder, output folder and RGB value table
input_folder = r"E:data"
output_folder = r"E:\data\output"

rgb_values = colors

process_images(input_folder, output_folder, rgb_values)