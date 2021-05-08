from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np


def draw_image(points, width, height, output_file="output.png"):
    # find out image limits determine scaling and translating
    min_x = min(points, key=lambda p: p[0])[0]
    max_x = max(points, key=lambda p: p[0])[0]
    min_y = min(points, key=lambda p: p[1])[1]
    max_y = max(points, key=lambda p: p[1])[1]
    p_width = max_x - min_x
    p_height = max_y - min_y

    width_scale = (width / p_width)
    height_scale = (height / p_height)
    scale = min(width_scale, height_scale)

    # create new image
    image = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(image)

    for point in points:
        x = (point[0] - min_x) * scale
        y = height - (point[1] - min_y) * scale
        draw.point((x, y))

    print("Saving image to: " + output_file)
    image.save(output_file, "PNG")


def plot_fractal(fractal_data, img_size=16, tol=.1, output_file="out.png"):
    img = -np.log(fractal_data + tol)
    plt.figure(figsize=(img_size, img_size))
    plt.axis('off')
    print("Saving image to: " + output_file)
    plt.imshow(img)
    plt.savefig(output_file, bbox_inches='tight')
