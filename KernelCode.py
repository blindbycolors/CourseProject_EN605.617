"""
CUDA kernel code for IFS Fractal Transformations
Algorithm source: https://jo.dreggn.org/home/2010_fractal_flames.pdf
"""
ifs_transform_kernel_code = """
#include <curand.h>
#include <curand_kernel.h>

extern "C"
{
    __device__ void copyToShared(float * transform,
                                 float * sharedTransform, 
                                 int numTransform,
                                 int threadId)
    {
        // Do this on one thread
        if (threadId == 0)
        {
            for (int i = 0; i < numTransform; ++i)
            {
                for (int j = 0; j < 7; ++j)
                {
                    sharedTransform[i * 7 + j] = transform[i * 7 + j];
                }
            }
        }
        __syncthreads();
    }

    __global__ void phase1Transform(float *xPoints,
                                    float *yPoints, 
                                    float *transform,
                                    int numPoints,
                                    int numTransform)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
    
        extern __shared__ float sharedTransform[];
        copyToShared(transform, sharedTransform, numTransform, threadIdx.x);
    
        if (index < numPoints)
        {
            curandState_t state;
    
            curand_init((unsigned long long)clock(), index, 0, &state);
            float random = curand_uniform(&state);
            float currX, currY;
            float randStart = (float) curand_uniform(&state);
            int randPoint = curand(&state) % numPoints;
            currX = xPoints[randPoint];
            currY = yPoints[randPoint];
    
            int i = 0;
            float pSum = 0;
    
            while (i < numTransform && pSum < random)
            {
                // 6th column will always contain probability
                pSum += transform[i * 7 + 6];
    
                if (random <= pSum)
                {
                    float newX = currX * sharedTransform[i * 7 + 0] + currY *
                        sharedTransform[i * 7 + 1] + sharedTransform[i * 7 + 4];
                    currY = currX * sharedTransform[i * 7 + 2] + currY *
                        sharedTransform[i * 7 + 3] + sharedTransform[i * 7 + 5];
                    currX = newX;
                    xPoints[index] = currX;
                    yPoints[index] = currY;
                }
    
                ++i;
            }
        }
    }
}
"""

"""
CUDA kernel code for Julia fractal sets
"""
divergent_fractal_kernel_code = """
    #include <cuComplex.h>
    #include <vector_types.h>
            
    #define TYPEFLOAT
    #ifdef TYPEFLOAT
    #define TYPE  float
    #define cTYPE cuFloatComplex
    #define cMakecuComplex(re, i) make_cuFloatComplex(re, i)
    #endif
    #ifdef TYPEDOUBLE
    #define TYPE  double
    #define cMakecuComplex(re, i) make_cuDoubleComplex(re, i)
    #endif
    
    cTYPE c0;
    __device__ cTYPE fractalFunctor(cTYPE p, cTYPE c)
    {
        return cuCaddf(cuCmulf(p, p), c);
    }
    
    __device__ int evolveComplexPoint(cTYPE z,
                                      cTYPE c,
                                      int maxIterations,
                                      int divergenceVal)
    {
        int it = 0;
    
        while (it <= maxIterations && cuCabsf(z) <= divergenceVal)
        {
            z = fractalFunctor(z, c);
            it++;
        }
        return it;
    }
    
    __device__ cTYPE convertToComplex(int x,
                                      int y,
                                      const int height,
                                      const int width)
    {
        TYPE jx = 1.5 * (x - width / 2) / (0.5 * width);
        TYPE jy = (y - height / 2) / (0.5 * height);
    
        return cMakecuComplex(jx, jy);
    }
    
    __global__ void computeFractal(int *data, cTYPE c, int width, int height,
                                   int maxIterations, int divergenceVal)
    {
        int i =  blockIdx.x * blockDim.x + threadIdx.x;
        int j =  blockIdx.y * blockDim.y + threadIdx.y;
    
        if (i < width && j < height)
        {
            cTYPE p = convertToComplex(i, j, height, width);
            int count = evolveComplexPoint(p, c, maxIterations, divergenceVal);
            data[i * width + j] = count; 
        }
    }
"""

"""
original source for sequential Hammersley sequence generation
https://blog.demofox.org/2017/05/29/when-random-numbers-are-too-random-low-discrepancy-sequences/
"""
gpu_hammersley_kernel_code = """
    __global__ void hammersley(int numSamples, float *xPoints, float *yPoints)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        // figure out how many bits we are working in.
        size_t truncateBits = 0;
        size_t value = 1;
        size_t numBits = 0;
    
        while (value < numSamples)
        {
            value *= 2;
            ++numBits;
        }
    
        // calculate the sample points   
        if (index < numSamples)
        {
            // x axis
            xPoints[index] = 0.0f;
            {
                size_t n = index >> truncateBits;
                float base = 1.0f / 2.0f;
    
                while (n)
                {
                    if (n & 1) xPoints[index] += base;
    
                    n /= 2;
                    base /= 2.0f;
                }
            }
    
            // y axis
            yPoints[index] = 0.0f;
            {
                size_t n = index >> truncateBits;
                size_t mask = size_t(1) << (numBits - 1 - truncateBits);
    
                float base = 1.0f / 2.0f;
    
                while (mask)
                {
                    if (n & mask) yPoints[index] += base;
    
                    mask /= 2;
                    base /= 2.0f;
                }
            }
        }
    }
"""
