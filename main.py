import CPUTransformation
import GPUTransformation
import constants
import pandas as pd
import sys
from os import path
from tabulate import tabulate


def print_help():
    print(
        "To run IFS fractal generation: --ifs {fern, dragon, leaf, pentadentrite, "
        "koch, spiral, tree, twig}")
    print("Optional arguments: --width {width of image}"
          "--height {height of image}"
          "--points {number of points}"
          "--gpu_out {full path for gpu image file}"
          "--cpu_out {full path for cpu image file}"
          "--block {block size for gpu}"
          "--timing {full file path to save run times}")
    print("To run a Julia set fractal: --julia {1-20}")
    print("Optional arguments: --size {size of image width/height}"
          "--iterations {number of total iterations}"
          "--gpu_output {full path for gpu image file}"
          "--cpu_output {full path for cpu image file}"
          "--block {block size for gpu}"
          "--divergence {divergence value}"
          "--timing {full path for file to save run times}")


def save_ifs_times(cpu_time, gpu_time, transformation, timing_file, num_points,
                   block_size, img_width, img_height, cpu_points, gpu_points):
    new_results = {"transformation": [transformation],
                   "points": [num_points],
                   "cpu_time": [cpu_time],
                   "gpu_time": [gpu_time],
                   "img_width": [img_width],
                   "img_height": [img_height],
                   "block_size": [block_size],
                   "cpu_points": [cpu_points],
                   "gpu_points": [gpu_points]}
    new_results = pd.DataFrame.from_dict(new_results)

    if len(timing_file) > 0:
        if path.exists(timing_file):
            print("File exists... Loading in csv...")
            results_table = pd.read_csv(timing_file, index_col=False)
            results_table = results_table.append(new_results)
        else:
            results_table = new_results

        results_table.to_csv(timing_file, index=False)
        print("Saved results to ", timing_file)

    print(tabulate(new_results, headers=new_results.keys()))


def save_julia_times(cpu_time, gpu_time, julia_set, output_file, iterations,
                     size, block):
    new_results_table = {"transformation": [julia_set],
                         "iterations": [iterations],
                         "image_size": [size],
                         "cpu_time_sec": [cpu_time],
                         "gpu_time_sec": [gpu_time],
                         "block_size": [block]}
    new_results_table = pd.DataFrame.from_dict(new_results_table)

    if len(output_file) > 0:
        if path.exists(output_file):
            print("File exists... Loading in csv...")
            results_table = pd.read_csv(output_file, index_col=False)
            results_table = results_table.append(new_results_table)
        else:
            results_table = new_results_table

        results_table.to_csv(output_file, index=False)
        print("Saved results to ", output_file)

    print(tabulate(new_results_table, headers=new_results_table.keys()))


def run_julia(set_to_run, iterations, divergence_val, width, block_size,
              cpu_output_file, gpu_output_file, timing_file, transform_name):
    julia_set = constants.julia_fractals[set_to_run]
    cpu_run_time = \
        CPUTransformation.cpuDivergentFractal(c=julia_set,
                                              iterations=iterations,
                                              divergence_value=divergence_val,
                                              width=width,
                                              output_file=cpu_output_file)
    gpu_run_time = \
        GPUTransformation.gpu_divergent_fractal(c=julia_set,
                                                iterations=iterations,
                                                divergence_value=divergence_val,
                                                width=width,
                                                block_size=block_size,
                                                output_file=gpu_output_file)

    save_julia_times(cpu_time=cpu_run_time,
                     gpu_time=gpu_run_time,
                     julia_set=transform_name,
                     output_file=timing_file,
                     size=width,
                     block=block_size,
                     iterations=iterations)


def run_ifs(transformation, width, height, num_points, gpu_output_file,
            cpu_output_file, timing_file, block, ifs_name):
    cpu_run_time, cpu_total_points = \
        CPUTransformation.cpuIfsTransform(transformation=transformation,
                                          width=width,
                                          height=height,
                                          num_points=num_points,
                                          output_file=cpu_output_file)
    gpu_run_time, gpu_total_points = \
        GPUTransformation.gpu_ifs_transform(transformation=transformation,
                                            width=width,
                                            height=height,
                                            num_points=num_points,
                                            block_size=block,
                                            output_file=gpu_output_file)

    save_ifs_times(cpu_run_time, gpu_run_time, ifs_name, timing_file,
                   num_points, block, width, height, cpu_total_points,
                   gpu_total_points)


def process_julia_runs(i, n):
    if i >= n:
        print("Error: Select a julia set by providing a number 1-20")
        print_help()
        exit(0)
    set_to_run = "set" + sys.argv[i]
    iterations = 200
    divergence_val = 10
    width = 300
    block_size = 64
    gpu_output_file = "gpuOut.png"
    cpu_output_file = "cpuOut.png"
    timing_file = ""
    i += 1
    while i < n:
        try:
            if sys.argv[i] == "--size" or sys.argv[i] == "-s":
                width = int(sys.argv[i + 1])
            elif sys.argv[i] == "--iterations" or sys.argv[i] == "-i":
                iterations = int(sys.argv[i + 1])
            elif sys.argv[i] == "--gpu_output" or sys.argv[i] == "-g":
                gpu_output_file = sys.argv[i + 1]
            elif sys.argv[i] == "--cpu_output" or sys.argv[i] == "-c":
                cpu_output_file = sys.argv[i + 1]
            elif sys.argv[i] == "--block" or sys.argv[i] == "-b":
                block_size = int(sys.argv[i + 1])
            elif sys.argv[i] == "--divergence" or sys.argv[i] == "-d":
                divergence_val = sys.argv[i + 1]
            elif sys.argv[i] == "--timing" or sys.argv[i] == "-t":
                timing_file = sys.argv[i + 1]
            i += 2
        except IndexError:
            print("Index out of range error. The expected number of arguments "
                  "were not provided.")
            continue
        except Exception:
            print("Error:", Exception.args)
            raise

    run_julia(set_to_run, iterations, divergence_val, width, block_size,
              cpu_output_file, gpu_output_file, timing_file, set_to_run)


def process_ifs_runs(i, n):
    if i >= n:
        print("Error: Select a IFS algorithm")
        print_help()
        exit(0)
    print(sys.argv)
    transform_name = sys.argv[i]
    transform = constants.ifs_fractals[transform_name]
    width = 300
    height = 300
    block_size = 64
    num_points = 10000
    gpu_output_file = "gpuOut.png"
    cpu_output_file = "cpuOut.png"
    timing_file = ""
    i += 1
    while i < n:
        try:
            if sys.argv[i] == "--width" or sys.argv[i] == "-s":
                width = int(sys.argv[i + 1])
            elif sys.argv[i] == "--height" or sys.argv[i] == "-h":
                height = int(sys.argv[i + 1])
            elif sys.argv[i] == "--points" or sys.argv[i] == "-p":
                num_points = int(sys.argv[i + 1])
            elif sys.argv[i] == "--gpu_output" or sys.argv[i] == "-g":
                gpu_output_file = sys.argv[i + 1]
            elif sys.argv[i] == "--cpu_output" or sys.argv[i] == "-c":
                cpu_output_file = sys.argv[i + 1]
            elif sys.argv[i] == "--block" or sys.argv[i] == "-b":
                block_size = int(sys.argv[i + 1])
            elif sys.argv[i] == "--timing" or sys.argv[i] == "-t":
                timing_file = sys.argv[i + 1]
            i += 2
        except IndexError:
            print("Index out of range error. The expected number of arguments "
                  "were not provided.")
            continue
        except Exception:
            print("Error:", Exception.args)
            raise

    run_ifs(transform, width, height, num_points, gpu_output_file,
            cpu_output_file, timing_file, block_size, transform_name)


def parse_input_args():
    n = len(sys.argv)
    i = 1
    if i >= n:
        print("Error: A fractal set must be provided.")
        print_help()
        exit(0)
    else:
        if sys.argv[i] == "--julia":
            i += 1
            process_julia_runs(i, n)
        elif sys.argv[i] == "--ifs":
            i += 1
            process_ifs_runs(i, n)
        else:
            print("Unknown option provided")
            print_help()


if __name__ == '__main__':
    parse_input_args()
