/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.
  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  author: Anthony M. Knight
  tony.knight@tknights.com
  date: 23 February 2017
  Got help when things didn't work from github and 
  http://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  Tested and verified on Windows10 with GTX1060
  B:\>c:..\release\hw3 memorial.exr
  Your code ran in: 0.717824 msecs.
  PASS

  B:\>c:..\release\hw3 memorial_large.exr
  Your code ran in: 1.122304 msecs.
  PASS

  B:\>c:..\release\hw3 memorial_png.gold
  Your code ran in: 0.755712 msecs.
  PASS

  B:\>c:..\release\hw3 memorial_png_large.gold
  Your code ran in: 0.849920 msecs.
  PASS

  B:\>c:..\release\hw3 memorial_raw.png
  Your code ran in: 0.745472 msecs.
  PASS

  B:\>c:..\release\hw3 memorial_raw_large.png
  Your code ran in: 1.184768 msecs.
  PASS
*/

#include <limits.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include "utils.h"

//TODO
/*Here are the steps you need to implement
1) find the minimum and maximum value in the input logLuminance channel
store in min_logLum and max_logLum
2) subtract them to find the range
3) generate a histogram of all the values in the logLuminance channel using
the formula: bin = (lum[i] - lumMin) / lumRange * numBins
4) Perform an exclusive scan (prefix sum) on the histogram to get
the cumulative distribution of luminance values (this should go in the
incoming d_cdf pointer which already has been allocated for you)       */

//#define DEBUGGING // turn on to output information
//#define INFORMATION

#define BLOCKSIZE 128 //512

template <unsigned int blockSize>
__device__ void warpMinReduce(volatile float *sdata, unsigned int tId) {
	if (blockSize >= 64) sdata[tId] = min(sdata[tId], sdata[tId + 32]);
	if (blockSize >= 32) sdata[tId] = min(sdata[tId], sdata[tId + 16]);
	if (blockSize >= 16) sdata[tId] = min(sdata[tId], sdata[tId + 8]);
	if (blockSize >= 8) sdata[tId] = min(sdata[tId], sdata[tId + 4]);
	if (blockSize >= 4) sdata[tId] = min(sdata[tId], sdata[tId + 2]);
	if (blockSize >= 2) sdata[tId] = min(sdata[tId], sdata[tId + 1]);
}
template <unsigned int blockSize>
__device__ void warpMaxReduce(volatile float *sdata, unsigned int tId) {
	if (blockSize >= 64) sdata[tId] = max(sdata[tId], sdata[tId + 32]);
	if (blockSize >= 32) sdata[tId] = max(sdata[tId], sdata[tId + 16]);
	if (blockSize >= 16) sdata[tId] = max(sdata[tId], sdata[tId + 8]);
	if (blockSize >= 8) sdata[tId] = max(sdata[tId], sdata[tId + 4]);
	if (blockSize >= 4) sdata[tId] = max(sdata[tId], sdata[tId + 2]);
	if (blockSize >= 2) sdata[tId] = max(sdata[tId], sdata[tId + 1]);
}

template <unsigned int blockSize>
__global__ void reduceMin_kernel(float *d_outMinMax, const float *d_inMinMax, unsigned int arraysize) {

	// float *d_outMinMax : pointer to array of floats that are the maximums 
	// const float *d_inMinMax : array of floats to be scanned for max and mins
	// unsigned int arraysize : size of the array to be scanned

	extern __shared__ float sdata[];  // shared memory to be faster
	unsigned int tId = threadIdx.x;
	unsigned int gId = blockIdx.x*(blockSize * 2) + tId;  // this allows twice number of elements to be loaded in first pull
	unsigned int gridSize = blockSize * 2 * gridDim.x;
	sdata[tId] = FLT_MAX;
	
	while (gId < arraysize) { sdata[tId] = min(sdata[tId], min(d_inMinMax[gId], d_inMinMax[gId + blockSize])); gId += gridSize; }
	__syncthreads();

	if (blockSize >= 512) { if (tId < 256) { sdata[tId] = min(sdata[tId], sdata[tId + 256]); } __syncthreads(); }
	if (blockSize >= 256) { if (tId < 128) { sdata[tId] = min(sdata[tId], sdata[tId + 128]); } __syncthreads(); }
	if (blockSize >= 128) { if (tId < 64) { sdata[tId] = min(sdata[tId], sdata[tId + 64]); } __syncthreads(); }

	if (tId < 32) warpMinReduce<BLOCKSIZE>(sdata, tId); // don't need syncthreads below 32
		
	if (tId == 0) {
		d_outMinMax[blockIdx.x] = sdata[0];
#ifdef DEBUGGING
		printf("Min[%d] = %3.6f,", blockIdx.x, sdata[0]);
#endif
	}
}

template <unsigned int blockSize>
__global__ void reduceMax_kernel(float *d_outMinMax, const float *d_inMinMax, unsigned int arraysize) {

	// float *d_outMinMax : pointer to array of floats that are the maximums 
	// const float *d_inMinMax : array of floats to be scanned for max and mins
	// unsigned int arraysize : size of the array to be scanned

	extern __shared__ float sdata[];  // shared memory to be faster
	unsigned int tId = threadIdx.x;
	unsigned int gId = blockIdx.x*(blockSize * 2) + tId;  // this allows twice number of elements to be loaded in first pull
	unsigned int gridSize = blockSize * 2 * gridDim.x;
	sdata[tId] = FLT_MIN;

	while (gId < arraysize) { sdata[tId] = max(sdata[tId], max(d_inMinMax[gId], d_inMinMax[gId + blockSize])); gId += gridSize; }
	__syncthreads();

	if (blockSize >= 512) { if (tId < 256) { sdata[tId] = max(sdata[tId], sdata[tId + 256]); } __syncthreads(); }
	if (blockSize >= 256) { if (tId < 128) { sdata[tId] = max(sdata[tId], sdata[tId + 128]); } __syncthreads(); }
	if (blockSize >= 128) { if (tId < 64) { sdata[tId] = max(sdata[tId], sdata[tId + 64]); } __syncthreads(); }

	if (tId < 32) warpMaxReduce<BLOCKSIZE>(sdata, tId); // don't need syncthreads below 32

	if (tId == 0) {
		d_outMinMax[blockIdx.x] = sdata[0];
#ifdef DEBUGGING
		printf("Max[%d] = %3.6f,", blockIdx.x, sdata[0]);
#endif
	}
}

__global__ void lumHisto_kernel(unsigned int *d_bins, const float* const d_in, const float lumMin, const float lumRange, const int numBins)
{
	int myId = threadIdx.x + blockDim.x * blockIdx.x;

	int myBin = min((numBins-1), int(numBins * (d_in[myId] - lumMin) / lumRange ) );

	if (myBin < 0 || myBin >= numBins){
		printf("WARNING - out of bounds myBin = %d, pixel = %d, value = %3.6f, lumMin = %3.6f\n", myBin, myId, d_in[myId], lumMin);
		if (myBin < 0) myBin = 0; else myBin = numBins - 1;	
	}
	atomicAdd(&(d_bins[myBin]), (unsigned int) 1);

}

__global__ void atomic_histo(unsigned int* d_histo, const float* const d_inputArray, const float minimum, const float range, const int numBins)
{
	int array_idx = threadIdx.x + blockDim.x*blockIdx.x;
	int bin_idx = int(numBins*(d_inputArray[array_idx] - minimum) / range);

	bin_idx = min(numBins - 1, bin_idx);
	atomicAdd(&(d_histo[bin_idx]), 1);
}


__global__ void lumHistExclusiveScan_kernel(unsigned int *d_out, unsigned int *d_in, int numItems)
{
	extern __shared__ unsigned int s_exScan[];
	int tid = threadIdx.x;

	s_exScan[tid] = (tid > 0) ? d_in[tid - 1] : 0;
	__syncthreads();

	for (int offset = 1; offset <= numItems; offset = offset * 2){
		unsigned int temp = s_exScan[tid];
		unsigned int neighbor = 0;
		if ((tid - offset) >= 0) {
			neighbor = s_exScan[tid - offset];
			__syncthreads();
			s_exScan[tid] = temp + neighbor;
		}
		__syncthreads();
	}
	d_out[tid] = s_exScan[tid];
}



void your_histogram_and_prefixsum(const float* const d_logLuminance,
	unsigned int* const d_cdf,
	float &min_logLum,
	float &max_logLum,
	const size_t numRows,
	const size_t numCols,
	const size_t numBins)
{
	const size_t numPixels = numRows*numCols;
#ifdef INFORMATION
	printf("The Number of pixels is %zd in %zd rows and %zd columns.\n", numPixels, numRows, numCols);
#endif
	// identify if uneven - 
	if (numPixels % 2) printf("this is not even!\n"); //- maybe need to assert here without any code changes - 

	const size_t maxThreadsPerBlock = BLOCKSIZE;

	unsigned int threads = (unsigned int) maxThreadsPerBlock;

	size_t blocks = numPixels / (2 * maxThreadsPerBlock);

	float* d_maxout;
	float* d_maxout1;
	checkCudaErrors(cudaMalloc((void**)&d_maxout, blocks * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_maxout1, sizeof(float)));

	float* d_minout;
	float* d_minout1;
	checkCudaErrors(cudaMalloc((void**)&d_minout, blocks * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_minout1, sizeof(float)));

	reduceMax_kernel<128> << < blocks, threads, threads * sizeof(float) >> >(d_maxout, d_logLuminance, (unsigned int) numPixels);
	reduceMin_kernel<128> << < blocks, threads, threads * sizeof(float) >> >(d_minout, d_logLuminance, (unsigned int) numPixels);

	float * h_maxout, * h_minout;

#ifdef DEBUGGING
	h_maxout = (float*)malloc(blocks * sizeof(float));
	checkCudaErrors(cudaMemcpy(h_maxout, d_maxout, blocks * (size_t)sizeof(float), cudaMemcpyDeviceToHost));
	h_minout = (float*)std::malloc(blocks * sizeof(float));
	checkCudaErrors(cudaMemcpy(h_minout, d_minout, blocks * sizeof(float), cudaMemcpyDeviceToHost));
	std::cout << "h_maxout after first run[ ";
	for (int i = 0; i < blocks; i++) {
		std::cout << h_maxout[i] << ",";
	}
	std::cout << "]" << std::endl;
	free(h_maxout);

	std::cout << "h_minout after first run[ ";
	for (int i = 0; i < blocks; i++) {
		std::cout << h_maxout[i] << ",";
	}
	std::cout << "]" << std::endl;
	free(h_minout);
#endif

	reduceMax_kernel<128> << < 1, threads, threads * sizeof(float) >> >(d_maxout1, d_maxout, blocks);
	reduceMin_kernel<128> << < 1, threads, threads * sizeof(float) >> >(d_minout1, d_minout, blocks);

	h_maxout = (float*)std::malloc(sizeof(float));
	h_minout = (float*)std::malloc(sizeof(float));
	
	checkCudaErrors(cudaMemcpy(h_maxout, d_maxout1, sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_minout, d_minout1, sizeof(float), cudaMemcpyDeviceToHost));

	min_logLum = h_minout[0];
	max_logLum = h_maxout[0];
	float range_logLum = max_logLum - min_logLum;

#ifdef INFORMATION
	printf("min is %3.6f, max is %3.6f, range is %3.6f\n", min_logLum, max_logLum, range_logLum);
#endif

	checkCudaErrors(cudaFree(d_maxout));
	checkCudaErrors(cudaFree(d_maxout1));
	checkCudaErrors(cudaFree(d_minout));
	checkCudaErrors(cudaFree(d_minout1));
	free(h_maxout);
	free(h_minout);

/* 3) generate a histogram of all the values in the logLuminance channel using
the formula: bin = (lum[i] - lumMin) / lumRange * numBins */

	threads = maxThreadsPerBlock;
	blocks = numPixels / maxThreadsPerBlock;
	unsigned int * d_lumBins;

	checkCudaErrors(cudaMalloc(&d_lumBins, numBins * sizeof(unsigned int)) ) ;
	checkCudaErrors(cudaMemset(d_lumBins, 0 ,  numBins * sizeof(unsigned int)));

	lumHisto_kernel << <blocks, threads>> >
		(d_lumBins, d_logLuminance, min_logLum, range_logLum, (int) numBins);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

#ifdef DEBUGGING
	unsigned int* h_lumBins = (unsigned int*)std::malloc(numBins * sizeof(unsigned int));
	checkCudaErrors(cudaMemcpy(h_lumBins, d_lumBins, numBins * sizeof(unsigned), cudaMemcpyDeviceToHost));

	std::cout << "d_lumBins[ ";
	for (int i = 0; i < numBins; i++) {
		std::cout << h_lumBins[i] << ",";
	}
	std::cout << "]" << std::endl;

	free(h_lumBins);
#endif
	
	/*	4) Perform an exclusive scan(prefix sum) on the histogram to get
		the cumulative distribution of luminance values(this should go in the
		incoming d_cdf pointer which already has been allocated for you) */
	
	size_t mTpBlock = 1024;  //just happens to be the number of bins

	blocks = numBins / mTpBlock; // 1 in this case
	threads = (unsigned int) numBins;
	size_t channelSize = mTpBlock * sizeof(unsigned int);
	
	lumHistExclusiveScan_kernel << <blocks, threads, channelSize >> > (d_cdf, d_lumBins, (int) numBins);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

#ifdef DEBUGGING
	unsigned int* h_cdf = (unsigned int*)std::malloc(numBins * sizeof(unsigned int));
	checkCudaErrors(cudaMemcpy(h_cdf, d_cdf, numBins * sizeof(unsigned), cudaMemcpyDeviceToHost));

	std::cout << "cdf[ ";
	for (int i = 0; i < numBins; i++) {
		std::cout << h_cdf[i] << ",";
	}
	std::cout << "]" << std::endl;

	free(h_cdf);
#endif
	checkCudaErrors(cudaFree(d_lumBins));
}