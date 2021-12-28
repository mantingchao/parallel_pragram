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

__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY, int *d_res, int width, int height, int maxIterations, int pitch, int pixelsPerThreads)
{
    // To avoid error caused by the floating number, use the following pseudo code
    for (int p = 0; p < pixelsPerThreads; p++)
    {
        int i = (blockIdx.x * blockDim.x + threadIdx.x) * pixelsPerThreads + p;
        int j = (blockIdx.y * blockDim.y + threadIdx.y);
        // if(p >= height) return;
        // for (int k = i; k < i + pixelsPerThreads; k++)
        // {
        // if(k >= width) continue;
        float x = lowerX + i * stepX;
        float y = lowerY + j * stepY;
        // int index = p * width + k;
        // d_res[index] = mandel(x, y, maxIterations);
        int *row = (int *)((char *)d_res + j * pitch);
        row[i] = mandel(x, y, maxIterations);
        // }
    }
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE(float upperX, float upperY, float lowerX, float lowerY, int *img, int resX, int resY, int maxIterations)
{

    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    int *d_res, *h_res;
    size_t pitch; // 6656
    int pixelsPerThreads = 20;
    int size = resX * resY * sizeof(int);
    cudaHostAlloc((void **)&h_res, size, cudaHostAllocDefault);         // host
    cudaMallocPitch((void **)&d_res, &pitch, resX * sizeof(int), resY); // device
    cudaMemcpy2D(d_res, pitch, h_res, sizeof(float) * resX, sizeof(float) * resX, resY, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(thread_num, thread_num);
    dim3 numBlocks(resX / (thread_num * pixelsPerThreads), resY / thread_num);
    mandelKernel<<<numBlocks, threadsPerBlock>>>(lowerX, lowerY, stepX, stepY, d_res, resX, resY, maxIterations, pitch, pixelsPerThreads);
    cudaDeviceSynchronize();

    cudaMemcpy2D(h_res, resX * sizeof(int), d_res, pitch, resX * sizeof(int), resY, cudaMemcpyDeviceToHost); // device to host
    memcpy(img, h_res, size);

    // free memory
    cudaFree(d_res);
    cudaFreeHost(h_res);
}
