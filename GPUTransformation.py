import sys
import numpy as np
import pycuda.autoinit  # noqa
import pycuda.gpuarray as gpuarray
from timeit import default_timer as timer
from pycuda.compiler import SourceModule

import KernelCode
import Utilities
import constants


def gpu_ifs_transform(transformation=constants.ifs_fractals["fern"],
                      width=600, height=600, num_points=100000,
                      block_size=64, output_file="gpuOut.png"):
    """
    This function will perform the Iterated Function System (IFS) fractal
    algorithm via CUDA.
    :param block_size: GPU Block Size
    :param transformation: A transformation matrix with 7 columns representing
        [a, b, c, d, e, f, prob] for the IFS function x_(n+1) = ax_n + by_n + e
        and y_(n+1) = cx_n + dy_n + f
    :param width: Width of the image in pixels
    :param height: Height of the image in pixels
    :param num_points: Number of points in fractal
    :param output_file: File to save the image to
    :return: algorithm runtime in seconds
    """
    start = timer()
    # Generate Hammersley sequence
    block = (block_size, 1, 1)

    gpu_x = gpuarray.to_gpu(np.zeros(num_points, np.float32))
    gpu_y = gpuarray.to_gpu(np.zeros(num_points, np.float32))
    data_generation = SourceModule(KernelCode.gpu_hammersley_kernel_code)
    hammersley_func = data_generation.get_function("hammersley")
    hammersley_func(np.int32(num_points), gpu_x, gpu_y, block=block)

    transformation = np.array(transformation, np.float32)
    gpu_transform = gpuarray.to_gpu(transformation)
    rows, cols = transformation.shape

    grid = (num_points, 1, 1)
    mod = SourceModule(KernelCode.ifs_transform_kernel_code, no_extern_c=True)
    ifs_func = mod.get_function("phase1Transform")
    ifs_func(gpu_x, gpu_y, gpu_transform, np.int32(num_points), np.int32(rows),
             block=block, grid=grid, shared=sys.getsizeof(gpu_transform))

    curr_iter = 0
    while curr_iter < 15:
        ifs_func(gpu_x, gpu_y, gpu_transform,
                 np.int32(num_points), np.int32(rows),
                 block=block, grid=grid,
                 shared=sys.getsizeof(gpu_transform))
        curr_iter += 1

    x = gpu_x.get()
    y = gpu_y.get()
    points = list(zip(x, y))
    run_time = timer() - start
    Utilities.draw_image(points, width, height, output_file)
    return run_time


def gpu_divergent_fractal(c=constants.julia_fractals["set1"],
                          iterations=200,
                          divergence_value=10,
                          width=300,
                          block_size=64,
                          output_file="gpuOut.png"):
    """
    GPU implementation of divergent quadratic map 'z = z^2 + c' for nIterations.
    :param block_size: GPU block size
    :param output_file: Filename to save image as
    :param width: Width of image in pixels
    :param c: Complex value representation
    :param iterations: total number of iterations
    :param divergence_value: divergence value for algorithm
    :return: algorithm runtime in seconds
    """
    start = timer()
    height = width
    gpu_data = gpuarray.empty((width, height), np.int32)

    mod = SourceModule(KernelCode.divergent_fractal_kernel_code)
    block = (block_size, 1, 1)
    grid = (width, height, 1)

    func = mod.get_function("computeFractal")
    func(gpu_data, c, np.int32(width), np.int32(height),
         np.int32(iterations),
         np.int32(divergence_value), block=block, grid=grid)
    data = gpu_data.get()
    run_time = timer() - start
    Utilities.plot_fractal(data, width, height, output_file)
    return run_time
