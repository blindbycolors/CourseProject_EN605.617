#CUDA Fractal Art Generator

This project implements a CUDA fractal art generator implemented in Python. The
project was tested with Python 3.8.5. Additionally, the project requires and was
tested with the libraries:

- numba==0.53.1
- numpy==1.20.2
- pandas==1.2.4
- Pillow==8.2.0
- pycuda==2021.1
- tabulate==0.8.9
- matplotlib==3.4.1
- matplotlib-inline==0.1.2

Currently, the project can generate a series of the Julia set and IFS fractals.

IFS fractals that are implemented are:
- fern
- dragon
- maple leaf
- pentadentrite
- koch curve
- spiral
- tree
- twig

Julia Fractal sets that are implemented are:
- set 1: 0 + 0.8i
- set 2: 0.37 + 0.1i
- set 3: 0.355 + 0.355i
- set 4: -0.54 + 0.54i
- set 5: -0.4 - 0.5i
- set 6: 0.34 - 0.05i
- set 7: 0.355534 - 0.337292i
- set 8: -1.34882125854492 - 0.454237874348958i
- set 9: -0.79 + 0.15i
- set 10: -0.162 + 1.04i
- set 11: 0.3 - 0.01i
- set 12: -1.476 + 0i
- set 13: -0.12 - 0.77i
- set 14: 0.28 + .008i 

#Running the Software
To generate an implemented IFS fractal:<br>
python3 main.py --ifs {fern, dragon, leaf, pentadentrite, koch, spiral, tree, tree}

Optional arguments:<br>
- --width, -w {width of image}
- -height, -h {height of image}
- --points, -p {number of points}
- --gpu_output, -g {full path for gpu image file}
- --cpu_output, -c {full path for cpu image file}
- --block, -b {block size for gpu}
- --timing, -t {full file path to save run times} 
  - If blank, runtimes will not be saved

To generate an implemented Julia set fractal:<br>
python3 main.py --julia {1-14}

Optional arguments:<br>
- --size, -s {size of image width/height}
    - Note, if the size is too large and not enough memory is allocated to the
      Python stack, a seg fault will occur and kill the python process
- --iterations {number of total iterations}
- --gpu_output {full path for gpu image file}
- --cpu_output {full path for cpu image file}
- --block {block size for gpu}
- --divergence {divergence value}
- --timing {full path for file to save run times}
  - If blank, runtimes will not be saved

<b>Optional arguments can be provided in any order.</b>

#Examples

<h2>Julia Set 2</h2>
![alt text](https://github.com/blindbycolors/CourseProject_EN605.617/blob/main/Julia2.gif)

Additional images can be found in the /images directory.

