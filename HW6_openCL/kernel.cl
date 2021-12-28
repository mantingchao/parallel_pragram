__kernel void convolution(__global float *inputImage, __global float *outputImage, __global float *filter,
                        int imageHeight, int imageWidth, int filterWidth) 
{
    int index = get_global_id(0);
    int halffilterSize = filterWidth / 2;
    float sum = 0.0f;
    int k, l;
    int row = index / imageWidth;
    int col = index % imageWidth;

    for (k = -halffilterSize; k <= halffilterSize; k++)
    {
        for (l = -halffilterSize; l <= halffilterSize; l++)
        { 
            if (row + k >= 0 && row + k < imageHeight &&
                col + l >= 0 && col + l < imageWidth)
            {
                sum += inputImage[(row + k) * imageWidth + col + l] *
                        filter[(k + halffilterSize) * filterWidth +
                                l + halffilterSize];
            }
        }
    }
    outputImage[row * imageWidth + col] = sum;
}
