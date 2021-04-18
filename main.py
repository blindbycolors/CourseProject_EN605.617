import CPUTransformation
import GPUTransformation
import constants

if __name__ == '__main__':
    CPUTransformation.cpuIfsTransform(output_file="cpuIFSOut.png")
    GPUTransformation.gpuIfsTransform(output_file="gpuIFSOut.png")
    CPUTransformation.cpuDivergentFractal(output_file="cpuJulia.png")
    GPUTransformation.gpuDivergentFractal(output_file="gpuJulia.png")

