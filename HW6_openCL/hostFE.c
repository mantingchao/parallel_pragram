#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    int filterSize = filterWidth * filterWidth * sizeof(float);
    int imageSize = imageHeight * imageWidth * sizeof(float);

    cl_command_queue commandQueue = clCreateCommandQueue(*context, *device, 0, NULL);

    cl_mem inputBuffer = clCreateBuffer(*context, CL_MEM_READ_ONLY, imageSize, NULL, NULL);
    cl_mem fliterBuffer = clCreateBuffer(*context, CL_MEM_READ_ONLY, filterSize, NULL, NULL);
    cl_mem outputBuffer = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, imageSize, NULL, NULL);

    clEnqueueWriteBuffer(commandQueue, inputBuffer, CL_TRUE, 0, imageSize, inputImage, 0, NULL, NULL);
    clEnqueueWriteBuffer(commandQueue, fliterBuffer, CL_TRUE, 0, filterSize, filter, 0, NULL, NULL);

    cl_kernel kernel = clCreateKernel(*program, "convolution", NULL);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &fliterBuffer);
    clSetKernelArg(kernel, 3, sizeof(cl_int), &imageHeight);
    clSetKernelArg(kernel, 4, sizeof(cl_int), &imageWidth);
    clSetKernelArg(kernel, 5, sizeof(cl_int), &filterWidth);

    size_t globalItemSize = imageHeight * imageWidth;
    size_t localItemSize = 64;
    clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &globalItemSize, &localItemSize, 0, NULL, NULL);
    clEnqueueReadBuffer(commandQueue, outputBuffer, CL_TRUE, 0, imageSize, outputImage, 0, NULL, NULL);
}