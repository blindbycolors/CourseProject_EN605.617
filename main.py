import CPUTransformation
import GPUTransformation
import constants


def runIfsTest():
    resultsTable = {"transformation": list(), "points": list(),
                    "cpuTime": list(), "gpuTime": list()}
    numPoints = [10000, 50000, 100000, 150000, 200000, 300000]
    sizes = [300, 600, 1200]

    for points in numPoints:
        for size in sizes:
            for name, transform in constants.ifsFractals:
                cpuFile = "images/cpu" + name + "_" + points + ".png"
                gpuFile = "images/gpu" + name + "_" + points + ".png"
                cpuTime = CPUTransformation.cpuIfsTransform(
                    transformation=transform,
                    num_points=points,
                    width=size,
                    height=size,
                    output_file=cpuFile)
                gpuTiime = GPUTransformation.gpuIfsTransform(
                    transformation=transform,
                    num_points=points,
                    width=size,
                    height=size,
                    output_file=gpuFile)
                resultsTable["transformation"].append(name)
                resultsTable["points"].append(points)
                resultsTable["cpuTime"].append(cpuTime)
                resultsTable["gpuTime"].append(gpuTiime)

    print(resultsTable)


if __name__ == '__main__':
    runIfsTest()
