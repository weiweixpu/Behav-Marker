from PIL import Image
import os

def crop_png_folder(folder_path, output_folder_path):
    crop_color = [(255, 128, 0), (255, 128, 1), (255, 129, 2), (255, 129, 1)]
    white_color = [(210, 214, 220), (234, 236, 239), (206, 210, 217), (200, 205, 213),
                   (213, 217, 223), (212, 216, 222), (227, 229, 233), (219, 222, 227),
                   (200, 205, 212), (200, 204, 213), (240, 240, 240), (200, 204, 212),
                   (254, 127, 1), (199, 204, 213),(201, 206, 213)]
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.png'):
            file_path = os.path.join(folder_path, file_name)
            output_file_path = os.path.join(output_folder_path, file_name)
            img = Image.open(file_path).convert('RGB')
            width, height = img.size
            left, right, top, bottom = width, 0, height, 0
            for x in range(width):
                for y in range(height):
                    r, g, b = img.getpixel((x, y))
                    if (r, g, b) in crop_color:
                        left = min(left, x)
                        right = max(right, x)
                        top = min(top, y)
                        bottom = max(bottom, y)
                    elif (r, g, b) in white_color:
                        img.putpixel((x, y), (255, 255, 255))
            if left <= right and top <= bottom:
                cropped_img = img.crop((left, top, right + 1, bottom + 1))
                for x in range(cropped_img.width):
                    for y in range(cropped_img.height):
                        r, g, b = cropped_img.getpixel((x, y))
                        if (r, g, b) in crop_color:
                            cropped_img.putpixel((x, y), (255, 255, 255))
                cropped_img.save(output_file_path)

crop_png_folder(r'E:\data\Open Field\Trajectory plot\Third\Trajectory plot',
                r'E:\data\Open Field\Trajectory plot\Third\Trajectory plot output')


