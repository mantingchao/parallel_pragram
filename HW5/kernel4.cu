#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#define thread_num 16

__device__ int mandel(float c_re, float c_im, int count)
{
    float z_re = c_re, z_im = c_im;
    int i;
    for (i = 0; i < count; ++i)
    {
        if (z_re * z_re + z_im * z_im > 4.f)
            break;

        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    return i;
}

__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY, int *d_res, int width, int height, int maxIterations)
{
    // To avoid error caused by the floating number, use the following pseudo code
    //
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    float x = lowerX + i * stepX;
    float y = lowerY + j * stepY;
    int index = j * width + i;
    d_res[index] = mandel(x, y, maxIterations);
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE(float upperX, float upperY, float lowerX, float lowerY, int *img, int resX, int resY, int maxIterations)
{

    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    int *d_res;
    int size = resX * resY * sizeof(int);
    cudaMalloc((void **)&d_res, size); // to allocate global memory on device

    dim3 threadsPerBlock(thread_num, thread_num);
    dim3 numBlocks(resX / thread_num, resY / thread_num);
    mandelKernel<<<numBlocks, threadsPerBlock>>>(lowerX, lowerY, stepX, stepY, d_res, resX, resY, maxIterations);

    cudaMemcpy(img, d_res, size, cudaMemcpyDeviceToHost); // device to host

    // free memory
    cudaFree(d_res);
}
