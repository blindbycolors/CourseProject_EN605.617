import sys
import numpy as np
import pycuda.autoinit  # noqa
import pycuda.gpuarray as gpuarray
from timeit import default_timer as timer
from pycuda.compiler import SourceModule

import KernelCode
import Utilities
import constants


def gpuIfsTransform(transformation=constants.ifsFractals["fern"],
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

    gpuX = gpuarray.to_gpu(np.zeros(num_points, np.float32))
    gpuY = gpuarray.to_gpu(np.zeros(num_points, np.float32))
    dataGeneration = SourceModule(KernelCode.gpuHammersley)
    hammersleyFunc = dataGeneration.get_function("hammersley")
    hammersleyFunc(np.int32(num_points), gpuX, gpuY, block=block)

    transformation = np.array(transformation, np.float32)
    gpuTransform = gpuarray.to_gpu(transformation)
    rows, cols = transformation.shape

    grid = (num_points, 1, 1)
    mod = SourceModule(KernelCode.ifsTransform, no_extern_c=True)
    ifsFunc = mod.get_function("phase1Transform")
    ifsFunc(gpuX, gpuY, gpuTransform, np.int32(num_points), np.int32(rows),
         block=block, grid=grid, shared=sys.getsizeof(gpuTransform))

    currIter = 0
    while currIter < 15:
        ifsFunc(gpuX, gpuY, gpuTransform,
             np.int32(num_points), np.int32(rows),
             block=block, grid=grid,
             shared=sys.getsizeof(gpuTransform))
        currIter += 1

    x = gpuX.get()
    y = gpuY.get()
    points = list(zip(x, y))
    Utilities.drawImage(points, width, height, output_file)
    return timer() - start

def gpuDivergentFractal(c=constants.juliaFractals["set1"], iterations=200,
                        divergence_value=4, width=300, block_size=64,
                        output_file="gpuOut.png"):
    """
    GPU implementation of divergent quadratic map 'z = z^2 + c' for nIterations.
    :param block_size: GPU block size
    :param output_file: Filename to save image as
    :param width: Width of image in pixels
    :param c: Complex value representation
    :param iterations: total number of iterations
    :param divergence_value: divergence value for algorithm
    :return: none
    """
    height = width
    gpuData = gpuarray.empty((width, height), np.int32)

    mod = SourceModule(KernelCode.fractalKernelCode)
    block = (block_size, 1, 1)
    grid = (width, height, 1)

    func = mod.get_function("computeFractal")
    func(gpuData, c, np.int32(width), np.int32(height),
         np.int32(iterations),
         np.int32(divergence_value), block=block, grid=grid)
    data = gpuData.get()
    Utilities.plotFractal(data, width, height, output_file)
