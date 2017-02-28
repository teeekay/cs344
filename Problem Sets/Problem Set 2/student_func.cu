// Homework 2
// Image Blurring
//
// In this homework we are blurring an image. To do this, imagine that we have
// a square array of weight values. For each pixel in the image, imagine that we
// overlay this square array of weights on top of the image such that the center
// of the weight array is aligned with the current pixel. To compute a blurred
// pixel value, we multiply each pair of numbers that line up. In other words, we
// multiply each weight with the pixel underneath it. Finally, we add up all of the
// multiplied numbers and assign that value to our output for the current pixel.
// We repeat this process for all the pixels in the image.

// To help get you started, we have included some useful notes here.

//****************************************************************************

// For a color image that has multiple channels, we suggest separating
// the different color channels so that each color is stored contiguously
// instead of being interleaved. This will simplify your code.

// That is instead of RGBARGBARGBARGBA... we suggest transforming to three
// arrays (as in the previous homework we ignore the alpha channel again):
//  1) RRRRRRRR...
//  2) GGGGGGGG...
//  3) BBBBBBBB...
//
// The original layout is known an Array of Structures (AoS) whereas the
// format we are converting to is known as a Structure of Arrays (SoA).

// As a warm-up, we will ask you to write the kernel that performs this
// separation. You should then write the "meat" of the assignment,
// which is the kernel that performs the actual blur. We provide code that
// re-combines your blurred results for each color channel.

//****************************************************************************

// You must fill in the gaussian_blur kernel to perform the blurring of the
// inputChannel, using the array of weights, and put the result in the outputChannel.

// Here is an example of computing a blur, using a weighted average, for a single
// pixel in a small image.
//
// Array of weights:
//
//  0.0  0.2  0.0
//  0.2  0.2  0.2
//  0.0  0.2  0.0
//
// Image (note that we align the array of weights to the center of the box):
//
//    1  2  5  2  0  3
//       -------
//    3 |2  5  1| 6  0       0.0*2 + 0.2*5 + 0.0*1 +
//      |       |
//    4 |3  6  2| 1  4   ->  0.2*3 + 0.2*6 + 0.2*2 +   ->  3.2
//      |       |
//    0 |4  0  3| 4  2       0.0*4 + 0.2*0 + 0.0*3
//       -------
//    9  6  5  0  3  9
//
//         (1)                         (2)                 (3)
//
// A good starting place is to map each thread to a pixel as you have before.
// Then every thread can perform steps 2 and 3 in the diagram above
// completely independently of one another.

// Note that the array of weights is square, so its height is the same as its width.
// We refer to the array of weights as a filter, and we refer to its width with the
// variable filterWidth.

//****************************************************************************

// Your homework submission will be evaluated based on correctness and speed.
// We test each pixel against a reference solution. If any pixel differs by
// more than some small threshold value, the system will tell you that your
// solution is incorrect, and it will let you try again.

// Once you have gotten that working correctly, then you can think about using
// shared memory and having the threads cooperate to achieve better performance.

//****************************************************************************

// Also note that we've supplied a helpful debugging function called checkCudaErrors.
// You should wrap your allocation and copying statements like we've done in the
// code we're supplying you. Here is an example of the unsafe way to allocate
// memory on the GPU:
//
// cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols);
//
// Here is an example of the safe way to do the same thing:
//
// checkCudaErrors(cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols));
//
// Writing code the safe way requires slightly more typing, but is very helpful for
// catching mistakes. If you write code the unsafe way and you make a mistake, then
// any subsequent kernels won't compute anything, and it will be hard to figure out
// why. Writing code the safe way will inform you as soon as you make a mistake.

// Finally, remember to free the memory you allocate at the end of the function.

//****************************************************************************
//#define NSIGHT_CUDA_DEBUGGER = 1
//Flag to define Block size
//#define TILE16x8
#ifndef TILE16x8
 #define TILE16x16
#endif

// SHARED_IMAGE_TILE - flag to load image into shared memory tiles with Halos
#define SHARED_IMAGE_TILE

// FOUR_BLOCKS - flag to use four block stamp to create halo rather than trying
// to do edges and corners seperately
#define FOUR_BLOCKS

// SHARED_FILTER - Flag to use the filter in shared memory
#define SHARED_FILTER

// SINGLE_LINE - Flag to cut down clamping function to single line
#define SINGLE_LINE


//Flag to define Block size
//#define TILE16x8
#ifndef TILE16x8
 #define TILE16x16
#endif

// SHARED_IMAGE_TILE - flag to load image into shared memory tiles with Halos
#define SHARED_IMAGE_TILE

// FOUR_BLOCKS - flag to use four block stamp to create halo rather than trying
// to do edges and corners seperately
#define FOUR_BLOCKS

// SHARED_FILTER - Flag to use the filter in shared memory
#define SHARED_FILTER

// SINGLE_LINE - Flag to cut down clamping function to single line
#define SINGLE_LINE


#include "utils.h"
//#include <stdio.h>

/* clamper - function to keep locations inbounds                            */
/* input 2d calculated location and                                         */
/* 2d position marking boundary of array (assuming (0,0) is start of array) */
/* return 1d location of edge if going outside of image                     */
__device__ int clamper(int s_locx, int s_locy, int2 a_bound)
{
#ifdef SINGLE_LINE
 // switched to single statement to try to reduce cost of function
 return(max(min(a_bound.y, s_locy), 0)
    *(a_bound.x+1)+max(min(a_bound.x, s_locx), 0));
#else
 int x = max(min(a_bound.x, s_locx), 0);
 int y = max(min(a_bound.y, s_locy), 0);
 return(y*(a_bound.x+1) + x);
#endif
}

//gaussian_blur
__global__
void gaussian_blur(const unsigned char* const inputChannel,
                   unsigned char* const outputChannel,
                   int numRows, int numCols,
                   const float* const filter, const int filterWidth)
{
    extern __shared__ float s_a[];

    const int2 t_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);
    const int t_1D_pos = t_2D_pos.y * numCols + t_2D_pos.x;
    const int2 t_2D_bound = make_int2(numCols-1,numRows-1);

  //make sure we don't try and access memory outside the image
  //by having any threads mapped there return early
    if ( t_2D_pos.x >= numCols || t_2D_pos.y >= numRows )
        return;

    const int fRadius = filterWidth/2;

#ifdef SHARED_IMAGE_TILE
    const int sCols = blockDim.x + fRadius * 2;
    const int sRows = blockDim.y + fRadius * 2;
    const int sPixels = sRows * sCols;
    const int2 s_2D_pos = make_int2(threadIdx.x+fRadius, threadIdx.y+fRadius);
    const int s_1D_pos = ((fRadius + threadIdx.y) * sCols) + fRadius + threadIdx.x;
    const int2 s_2D_bound = make_int2(sCols-1,sRows-1);
#endif

    const int filterSize = filterWidth * filterWidth;



#ifdef SHARED_FILTER
    int filterOffset = filterSize;
    const int f_1D_pos = threadIdx.y*filterWidth+threadIdx.x;
    const int2 f_2D_pos = make_int2(threadIdx.x,threadIdx.y);
    const int2 f_2D_bound = make_int2(filterWidth-1,filterWidth-1);
    // try to put filter into shared memory -
    //  should be easy, but couldn't get it to work properly (was a block size issue!)
    if (threadIdx.x < filterWidth && threadIdx.y < filterWidth){
        s_a[f_1D_pos] = filter[f_1D_pos];
    }
#else
    int filterOffset = 0;
#endif

#ifdef SHARED_IMAGE_TILE
#ifdef FOUR_BLOCKS
    s_a[filterOffset+s_2D_pos.x-fRadius+(s_2D_pos.y-fRadius)*sCols] =
         inputChannel[clamper(t_2D_pos.x-fRadius,t_2D_pos.y-fRadius,t_2D_bound)];

    s_a[filterOffset+s_2D_pos.x-fRadius+(s_2D_pos.y+fRadius)*sCols] =
         inputChannel[clamper(t_2D_pos.x-fRadius,t_2D_pos.y+fRadius,t_2D_bound)];

    s_a[filterOffset+s_2D_pos.x+fRadius+(s_2D_pos.y-fRadius)*sCols] =
          inputChannel[clamper(t_2D_pos.x+fRadius,t_2D_pos.y-fRadius,t_2D_bound)];

    s_a[filterOffset+s_2D_pos.x+fRadius+(s_2D_pos.y+fRadius)*sCols] =
          inputChannel[clamper(t_2D_pos.x+fRadius,t_2D_pos.y+fRadius,t_2D_bound)];
#else
    // try to fill each section around image tile -can't get to work properly missing case in corner.
    // first fill center of shared array with values from global memory
    s_a[filterOffset + s_1D_pos] = inputChannel[t_1D_pos];
    //s_a[filterOffset+clamper(s_2D_pos.x, s_2D_pos.y,s_2D_bound)] = inputChannel[clamper(t_2D_pos.x,t_2D_pos.y,t_2D_bound)];
    //fill halo directly to left and right of block
    if (threadIdx.x < fRadius){
        //s_a[filterOffset+clamper(s_2D_pos.x-fRadius, s_2D_pos.y, s_2D_bound)] = inputChannel[clamper(t_2D_pos.x-fRadius, t_2D_pos.y, t_2D_bound)];
        s_a[filterOffset+s_1D_pos-fRadius] = inputChannel[clamper(t_2D_pos.x-fRadius, t_2D_pos.y, t_2D_bound)];
        //s_a[filterOffset+clamper(s_2D_pos.x+blockDim.x, s_2D_pos.y, s_2D_bound)] = inputChannel[clamper(t_2D_pos.x+blockDim.x, t_2D_pos.y, t_2D_bound)];
        s_a[filterOffset+s_1D_pos+blockDim.x] = inputChannel[clamper(t_2D_pos.x+blockDim.x, t_2D_pos.y, t_2D_bound)];
    }
    //fill halo directly above and below
    if (threadIdx.y < fRadius){
        //s_a[filterOffset+clamper(s_2D_pos.x, s_2D_pos.y-fRadius, s_2D_bound)] = inputChannel[clamper(t_2D_pos.x, t_2D_pos.y-fRadius, t_2D_bound)];
        s_a[filterOffset+s_1D_pos-(fRadius*sCols)] = inputChannel[clamper(t_2D_pos.x, t_2D_pos.y-fRadius, t_2D_bound)];
        //s_a[filterOffset+clamper(s_2D_pos.x, s_2D_pos.y+blockDim.y,s_2D_bound)] = inputChannel[clamper(t_2D_pos.x, t_2D_pos.y+blockDim.y, t_2D_bound)];
        s_a[filterOffset+s_1D_pos+(blockDim.y*sCols)] = inputChannel[clamper(t_2D_pos.x, t_2D_pos.y+blockDim.y, t_2D_bound)];
    }
    // fill halos at corners - looks like I'm missing something in a corner somewhere here
    if ((threadIdx.x < fRadius) && (threadIdx.y < fRadius)){
        //top left
        s_a[filterOffset+s_1D_pos-fRadius-(fRadius*sCols)] = inputChannel[clamper(t_2D_pos.x-fRadius, t_2D_pos.y-fRadius, t_2D_bound)];
        //bottom left
        s_a[filterOffset+s_1D_pos-fRadius+(blockDim.y*sCols)] = inputChannel[clamper(t_2D_pos.x-fRadius, t_2D_pos.y+fRadius, t_2D_bound)];
        //top right
        s_a[filterOffset+s_1D_pos+blockDim.x-(fRadius*sCols)] = inputChannel[clamper(t_2D_pos.x+blockDim.x, t_2D_pos.y-fRadius, t_2D_bound)];
        //bottom right
        s_a[filterOffset+s_1D_pos+blockDim.x+(blockDim.y*sCols)] = inputChannel[clamper(t_2D_pos.x+blockDim.x, t_2D_pos.y+fRadius, t_2D_bound)];
    }
#endif
#endif
   __syncthreads();


    // NOTE: Be sure to compute any intermediate results in floating point
    // before storing the final result as unsigned char.

    float result = 0.f;
      //For every value in the filter around the pixel (c, r)
    for (int filter_r = -fRadius; filter_r <= fRadius; ++filter_r) {
        for (int filter_c = -fRadius; filter_c <= fRadius; ++filter_c) {
            //Find the global image position for this filter position
            //clamp to boundary of the image
        #ifdef SHARED_IMAGE_TILE
            int image_r = min(max(s_2D_pos.y + filter_r, 0), static_cast<int>(sRows - 1));
            int image_c = min(max(s_2D_pos.x + filter_c, 0), static_cast<int>(sCols - 1));
            float image_value = static_cast<float>(s_a[filterOffset + (image_r * sCols) + image_c]);
        #else
            int image_r = min(max(t_2D_pos.y + filter_r, 0), static_cast<int>(numRows - 1));
            int image_c = min(max(t_2D_pos.x + filter_c, 0), static_cast<int>(numCols - 1));
            float image_value = static_cast<float>(inputChannel[image_r * numCols + image_c]);
        #endif

        #ifdef SHARED_FILTER
        //try using shared memory filter instead of global
            float filter_value = s_a[filter_c + fRadius + (filter_r + fRadius)*filterWidth];
        #else
            float filter_value = filter[filter_c + fRadius + (filter_r + fRadius)*filterWidth];
        #endif
            result += image_value * filter_value;
        }
    }

    outputChannel[t_1D_pos] = result;
}

//This kernel takes in an image represented as a uchar4 and splits
//it into three images consisting of only one color channel each
__global__
void separateChannels(const uchar4* const inputImageRGBA,
                      int numRows,
                      int numCols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
{
  // TODO
  //
  // NOTE: Be careful not to try to access memory that is outside the bounds of
  // the image. You'll want code that performs the following check before accessing
  // GPU memory:
  //
  // if ( absolute_image_position_x >= numCols ||
  //      absolute_image_position_y >= numRows )
  // {
  //     return;
  // }
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);
  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  //make sure we don't try and access memory outside the image
  //by having any threads mapped there return early
  if ( thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows )
    return;

  uchar4 rgba = inputImageRGBA[thread_1D_pos];
  redChannel[thread_1D_pos] = rgba.x;
  greenChannel[thread_1D_pos] = rgba.y;
  blueChannel[thread_1D_pos] = rgba.z;
}

//This kernel takes in three color channels and recombines them
//into one image.  The alpha channel is set to 255 to represent
//that this image has no transparency.
__global__
void recombineChannels(const unsigned char* const redChannel,
                       const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* const outputImageRGBA,
                       int numRows,
                       int numCols)
{
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);

  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  //make sure we don't try and access memory outside the image
  //by having any threads mapped there return early
  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;

  unsigned char red   = redChannel[thread_1D_pos];
  unsigned char green = greenChannel[thread_1D_pos];
  unsigned char blue  = blueChannel[thread_1D_pos];

  //Alpha should be 255 for no transparency
  uchar4 outputPixel = make_uchar4(red, green, blue, 255);

  outputImageRGBA[thread_1D_pos] = outputPixel;
}

unsigned char *d_red, *d_green, *d_blue;
float         *d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const size_t filterWidth)
{

  //allocate memory for the three different channels
  //original
  checkCudaErrors(cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage));

  //TODO:
  //Allocate memory for the filter on the GPU
  //Use the pointer d_filter that we have already declared for you
  //You need to allocate memory for the filter with cudaMalloc
  //be sure to use checkCudaErrors like the above examples to
  //be able to tell if anything goes wrong
  //IMPORTANT: Notice that we pass a pointer to a pointer to cudaMalloc
  checkCudaErrors(cudaMalloc(&d_filter, sizeof(float) * filterWidth * filterWidth));

  //TODO:
  //Copy the filter on the host (h_filter) to the memory you just allocated
  //on the GPU.  cudaMemcpy(dst, src, numBytes, cudaMemcpyHostToDevice);
  //Remember to use checkCudaErrors!
  checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float) * filterWidth * filterWidth, cudaMemcpyHostToDevice));
}

void your_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        unsigned char *d_redBlurred,
                        unsigned char *d_greenBlurred,
                        unsigned char *d_blueBlurred,
                        const int filterWidth)
{
  //TODO: Set reasonable block size (i.e., number of threads per block)
#ifdef TILE16x8
    //found that this works fastest with no shared memory, but blows up if shared memory used
  //const dim3 blockSize(8,16,1);
  const dim3 blockSize(16,8,1);
#else
    // found that 16*16, 20*16 and 20*20 work fine, not 12*12 with shared mem

    const dim3 blockSize(16,16,1);
#endif

  //Compute correct grid size (i.e., number of blocks per kernel launch)
  //from the image size and and block size.
  size_t gridCols = (numCols + blockSize.x - 1) / blockSize.x;
  size_t gridRows = (numRows + blockSize.y - 1) / blockSize.y;

  const dim3 gridSize(gridCols, gridRows, 1);

  //compute size of shared space to move image data into
  int fRad = filterWidth/2;
  //const size_t sGridSz = (gridRows + 2*fRad)*(gridCols + 2*fRad) * sizeof(float);
  // add space for shared mem filter after shared mem image data

#ifdef SHARED_FILTER
    int s_filterSize = filterWidth*filterWidth;
#else
    int s_filterSize = 0;
#endif
#ifdef SHARED_IMAGE_TILE
    int s_imageSize = (gridRows + 2*fRad)*(gridCols + 2*fRad);
#else
    int s_imageSize = 0;
#endif
  const size_t sGridSz = (s_filterSize+s_imageSize) * sizeof(float);

  //TODO: Launch a kernel for separating the RGBA image into different color channels
  // Call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
  // launching your kernel to make sure that you didn't make any mistakes.
  separateChannels<<<gridSize, blockSize>>>(d_inputImageRGBA,
                                            numRows,
                                            numCols,
                                            d_red,
                                            d_green,
                                            d_blue);

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  //TODO: Call your convolution kernel here 3 times, once for each color channel.
  // Again, call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
  // launching your kernel to make sure that you didn't make any mistakes.

  gaussian_blur<<<gridSize, blockSize, sGridSz>>>(d_red,
                                         d_redBlurred,
                                         numRows,
                                         numCols,
                                         d_filter,
                                         filterWidth);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  gaussian_blur<<<gridSize, blockSize, sGridSz>>>(d_green,
                                         d_greenBlurred,
                                         numRows,
                                         numCols,
                                         d_filter,
                                         filterWidth);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  gaussian_blur<<<gridSize, blockSize, sGridSz>>>(d_blue,
                                         d_blueBlurred,
                                         numRows,
                                         numCols,
                                         d_filter,
                                         filterWidth);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // Now we recombine your results. We take care of launching this kernel for you.
  //
  // NOTE: This kernel launch depends on the gridSize and blockSize variables,
  // which you must set yourself.
  recombineChannels<<<gridSize, blockSize>>>(d_redBlurred,
                                             d_greenBlurred,
                                             d_blueBlurred,
                                             d_outputImageRGBA,
                                             numRows,
                                             numCols);

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

}


//Free all the memory that we allocated
//TODO: make sure you free any arrays that you allocated
void cleanup() {
  checkCudaErrors(cudaFree(d_red));
  checkCudaErrors(cudaFree(d_green));
  checkCudaErrors(cudaFree(d_blue));
  checkCudaErrors(cudaFree(d_filter));
}
