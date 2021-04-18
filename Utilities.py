from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
from ctypes import *


def drawImage(points, width, height, output_file="output.png"):
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

    image.save(output_file, "PNG")


def plotFractal(fractal_data, img_size=16, tol=.1, output_file="out.png"):
    img = -np.log(fractal_data + tol)
    plt.figure(figsize=(img_size, img_size))
    plt.axis('off')
    plt.imshow(img)
    plt.savefig(output_file, bbox_inches='tight')


def hammersley(num_samples):
    truncateBits = 0
    value = c_size_t(1)
    numBits = c_size_t(1)
    while value.value < num_samples:
        value.value *= 2
        numBits.value += 1

    samples = list()
    for i in range(num_samples):
        n = c_size_t(i >> truncateBits)
        base = np.float32(1.0) / np.float32(2.0)

        xAxis = np.float32(0)
        while n:
            if n.value & 1:
                xAxis += base
            n = c_size_t(int(n.value / 2))
            base /= np.float32(2)

        # Y axis
        n = c_size_t(i >> truncateBits)
        mask = c_size_t(c_size_t(1).value << numBits.value - 1 - truncateBits)
        yAxis = 0
        base = np.float32(1.0) / np.float32(2.0)
        while mask:
            if n.value & mask.value:
                yAxis += base
            mask = c_size_t(int(mask.value / 2))
            base /= np.float32(2)

        samples.append((xAxis, yAxis))

    return samples
