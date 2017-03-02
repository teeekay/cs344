//
// This reduction code worked most of the time, but failed on memorial_large.exr
// produced :
//   min is -3.109206, max is 2.350199, range is 5.459405
// should be :
//   min is -3.122315, max is 2.350199, range is 5.472514
// debugged how it failed - error due to running min and max reduction in a loop,
// but not re-initializing num_elem_in, when running second loop.
//  num_elem_in had been shrinking.
// Took a lot of debugging code, but realized issue was with the input when 
// I switched the order of the min max reductions and got the wrong answer for 
// the other operation.

#include <cuda_runtime.h>
#include "utils.h"

#define MAX_THREADS_PER_BLOCK 32 //1024 //?

__global__ void reduce_min_max_kernel(float* d_out, float* d_in, int array_len, bool use_min)
	/*
	Does a reduction of d_in into one final d_out array, where d_out is
	an array with length of the number of blocks in the kernel.
	Then only expectation is that blocks and threads are one dimensional.
	\Params:
	* array_len - length of d_in array
	* use_min - boolean to use minimum reduction operator if true, else use maximium.
	*/
{

	// Set up shared memory
	extern __shared__ float input_array[];

	int global_idx = blockIdx.x*blockDim.x + threadIdx.x;
	int th_idx = threadIdx.x;

	// If this thread index takes us outside of the array, fill it with
	// the first value of the actual global array.
	if (global_idx >= array_len) input_array[th_idx] = d_in[0];
	else input_array[th_idx] = d_in[global_idx];
	
	__syncthreads(); // syncs up all threads within the block.

					 // Do reduction in shared memory. All elements in input_array are
					 // filled (some with duplicate values from first element of global
					 // input array which wont effect final result).

	for (int neighbor = 1; neighbor<=blockDim.x/2; neighbor *=2){
		int skip = 2 * neighbor;
		if ((th_idx % skip) == 0) {
			if ((th_idx + neighbor) < blockDim.x) {
				if (use_min) input_array[th_idx] = min(input_array[th_idx], input_array[th_idx + neighbor]);
				else input_array[th_idx] = max(input_array[th_idx], input_array[th_idx + neighbor]);
			}
		}
		
		__syncthreads();
	}

	// only thread 0 writes result for this block to d_out:

	if (th_idx == 0) {
		d_out[blockIdx.x] = input_array[0];
	}
}

void reduce_min_max(const float* const d_input_array, unsigned num_elem_in, float& minimum,	float& maximum)
	/*
	Split up array into blocks of MAX_THREADS_PER_BLOCK length each, and
	reduce (find extrema) of each block, writing the output of the block
	to a new d_out array. Then the new d_out array becomes the input array
	to perform reduction on, until the length of the d_out array is 1 element
	and extremum is found.
	*/
{

	// We can't change original array, so copy it here on device so that
	// we can modify it:

	float * d_in;

	
	const unsigned int num_elem_store = num_elem_in;
	
	for (int min_or_max = 0; min_or_max < 2; min_or_max++)
	{
		num_elem_in = num_elem_store;
		checkCudaErrors(cudaMalloc((void**)&d_in, num_elem_in * sizeof(float)));
		checkCudaErrors(cudaMemcpy(d_in, d_input_array, num_elem_in * sizeof(float), cudaMemcpyDeviceToDevice));
				
		int nthreads = MAX_THREADS_PER_BLOCK;
		int num_elem_out = (num_elem_in - 1) / MAX_THREADS_PER_BLOCK + 1;
		int nblocks = num_elem_out;

		while (true) {

			float * d_out;
						
			checkCudaErrors(cudaMalloc((void**)&d_out, num_elem_out * sizeof(float)));
			reduce_min_max_kernel << <nblocks, nthreads, nthreads * sizeof(float) >> >
				(d_out, d_in, num_elem_in, (bool)min_or_max);
			checkCudaErrors(cudaFree(d_in));
			
			if (num_elem_out <= 1) {  //check this
				
				// Copy output to h_out
				float* h_out = (float*)malloc(sizeof(float));
				if (min_or_max) {
					checkCudaErrors(cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
					minimum = h_out[0];
				}
				else {
					checkCudaErrors(cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
					maximum = h_out[0];
				}
				free(h_out);

				checkCudaErrors(cudaFree(d_out));
				break;
			}

			// Now output array becomes new input array:
			num_elem_in = num_elem_out;
			num_elem_out = (num_elem_in - 1) / MAX_THREADS_PER_BLOCK + 1;
			nblocks = num_elem_out;
			d_in = d_out;
			
		}

	}
	return;
}

__global__ void shmem_reduce_kernel_min(float * d_out, const float * d_in, int arraysize)
{
	extern __shared__ float sdata[];
	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	int tid = threadIdx.x;
	if (myId > arraysize)
		sdata[tid] = FLT_MAX;
	else
		sdata[tid] = d_in[myId];
	__syncthreads(); // make sure entire block is loaded!
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) 
	{
		if (tid < s) 
		{
			sdata[tid] = min(sdata[tid], sdata[tid + s]);
		}
		__syncthreads(); 
	}
	if (tid == 0) d_out[blockIdx.x] = sdata[0];
}

__global__ void shmem_reduce_kernel_max(float * d_out, const float * d_in, int arraysize) 
{
	extern __shared__ float sdata[];
	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	int tid = threadIdx.x;
	if (myId > arraysize)
		sdata[tid] = FLT_MIN;
	else
		sdata[tid] = d_in[myId];
	
	__syncthreads(); // make sure entire block is loaded!
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) 
		{
			sdata[tid] = max(sdata[tid], sdata[tid + s]);
		}
		__syncthreads(); 
	}
	if (tid == 0) d_out[blockIdx.x] = sdata[0];
}

int nextpower(int in) {
	if (in > 1024 || in < 1) {
		printf("Array Size %d out of bounds!\n");
		return(in);
	}
	if (in > 512) return 1024;
	if (in > 256) return 512;
	if (in > 128) return 256;
	if (in > 64) return 128;
	if (in > 32) return 64;
	if (in > 16) return 32;
	if (in > 8) return 16;
	if (in > 4) return 8;
	if (in > 2) return 4;
	if (in == 1) return 2;
	return(in);
}


//void reduce_min_max2(float * d_outmin, float * d_outmax, const float* const d_in, float * d_intermin, float * d_intermax, const size_t numRows, const size_t numCols) {
void reduce_min_max2(float& d_outmin, float& d_outmax, const float* const d_in, const size_t numRows, const size_t numCols) {
	const int maxThreadsPerBlock = 1024; // BLOCKSIZE;
	float * d_out;
	checkCudaErrors(cudaMalloc((void**)&d_out, sizeof(float)));
	float* h_out = (float*)malloc(sizeof(float));
	int arraysize = numRows*numCols;
	int threads = maxThreadsPerBlock; // launch one thread for each block in prev step
	int blocks = numRows*numCols / maxThreadsPerBlock;
	blocks = nextpower(blocks);

	float * d_intermax;
	checkCudaErrors(cudaMalloc((void**)&d_intermax, blocks * sizeof(float)));
	
	shmem_reduce_kernel_max << <blocks, threads, threads * sizeof(float) >> > (d_intermax, d_in, arraysize);
	
	threads = blocks; // launch one thread for each block in prev step actually move up to next power of 2
	
	
	blocks = 1;
	
	shmem_reduce_kernel_max << <blocks, threads, threads * sizeof(float) >> > (d_out, d_intermax, arraysize/maxThreadsPerBlock);
	checkCudaErrors(cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
	d_outmax = h_out[0];

	threads = maxThreadsPerBlock;
	blocks = numRows*numCols / maxThreadsPerBlock;
	blocks = nextpower(blocks);
	float * d_intermin;
	checkCudaErrors(cudaMalloc((void**)&d_intermin, blocks * sizeof(float)));
	
	shmem_reduce_kernel_min << <blocks, threads, threads * sizeof(float) >> > (d_intermin, d_in, arraysize);
	threads = blocks; // launch one thread for each block in prev step
	blocks = 1;
	printf("launching %d threads in last reduction of single block\n", threads);
	shmem_reduce_kernel_min << < blocks, threads, threads * sizeof(float) >> > (d_out, d_intermin, arraysize/maxThreadsPerBlock);
	checkCudaErrors(cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
	d_outmin = h_out[0];

	checkCudaErrors(cudaFree(d_intermin));

	free(h_out);
	checkCudaErrors(cudaFree(d_intermax));
	checkCudaErrors(cudaFree(d_out));
}