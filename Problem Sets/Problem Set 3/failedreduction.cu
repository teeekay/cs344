//
// This reduction code works most of the time, but failed on memorial_large.exr
// did not debug how it fails
//
// produces :
//   min is -3.109206, max is 2.350199, range is 5.459405
// should be :
//   min is -3.122315, max is 2.350199, range is 5.472514


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

	for (int neighbor = 1; neighbor<=blockDim.x; neighbor *=2){
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

	float* d_in;
	for (int min_or_max = 0; min_or_max < 2; min_or_max++)
	{
		checkCudaErrors(cudaMalloc((void**)&d_in, num_elem_in * sizeof(float)));
		checkCudaErrors(cudaMemcpy(d_in, d_input_array, num_elem_in * sizeof(float),
			cudaMemcpyDeviceToDevice));

		int nthreads = MAX_THREADS_PER_BLOCK;
		int num_elem_out = (num_elem_in - 1) / MAX_THREADS_PER_BLOCK + 1;
		int nblocks = num_elem_out;

		//unsigned iloop = 0;
		while (true) {

			float* d_out;
			checkCudaErrors(cudaMalloc((void**)&d_out, num_elem_out * sizeof(float)));

			reduce_min_max_kernel << <nblocks, nthreads, nthreads * sizeof(float) >> >
				(d_out, d_in, num_elem_in, min_or_max);

			checkCudaErrors(cudaFree(d_in));

			if (num_elem_out <= 1) {  //check this
				// Copy output to h_out
				float* h_out = (float*)malloc(sizeof(float));
				checkCudaErrors(cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
				if(min_or_max == 0)
					maximum = h_out[0];
				else
					minimum = h_out[0];

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
	unsigned int arrayLen = (unsigned int)numPixels;

	reduce_min_max(d_logLuminance, arrayLen, min_logLum, max_logLum); //fails to find for memorial_large.exr

	float range_logLum = max_logLum - min_logLum;
	printf("min is %3.6f, max is %3.6f, range is %3.6f\n", min_logLum, max_logLum, range_logLum);
}
