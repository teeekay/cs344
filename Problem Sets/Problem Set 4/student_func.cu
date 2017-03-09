//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>
#include <stdio.h>
#include <cuda_runtime.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */
//#define DEBUGGING1	
// bitmasks
#define b0  0x00000001
#define b1  0x00000002
#define b2  0x00000004
#define b3  0x00000008
#define b4  0x00000010
#define b5  0x00000020
#define b6  0x00000040
#define b7  0x00000080
#define b8  0x00000100
#define b9  0x00000200
#define b10 0x00000400
#define b11 0x00000800
#define b12 0x00001000
#define b13 0x00002000
#define b14 0x00004000
#define b15 0x00008000
#define b16 0x00010000
#define b17 0x00020000
#define b18 0x00040000
#define b19 0x00080000
#define b20 0x00100000
#define b21 0x00200000
#define b22 0x00400000
#define b23 0x00800000
#define b24 0x01000000
#define b25 0x02000000
#define b26 0x04000000
#define b27 0x08000000
#define b28 0x10000000
#define b29 0x20000000
#define b30 0x40000000
#define b31 0x80000000
unsigned int bMasks[32] = { b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16, b17, b18, b19, b20, b21, b22, b23, b24, b25, b26, b27, b28, b29, b30, b31 };

__global__ void lsbHisto_kernel(unsigned int* d_binHistogram, unsigned int numBins, unsigned int* const d_inVals, const size_t numElems) {

	//1) loop from 0 to biggest value;
	//2) perform check to see if the value is 
	unsigned int bMasks[32] = { b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16, b17, b18, b19, b20, b21, b22, b23, b24, b25, b26, b27, b28, b29, b30, b31 };
	extern __shared__ unsigned int s_vals[];

	if (numBins > 32) numBins = 32;
	unsigned int tIdx = threadIdx.x;
	unsigned int gIdx = blockIdx.x * blockDim.x + threadIdx.x;

	if (gIdx < numElems) 
	{
		s_vals[tIdx] = d_inVals[gIdx];
		for (int i = 0; i < numBins; i++) {
			if ((bMasks[i]&s_vals[tIdx]) == bMasks[i]) 
			{
				atomicAdd(&(d_binHistogram[i]), 1);

			}
		}
	}
#ifdef DEBUGGING2	
	if (blockIdx.x < 1) {
		printf("d_inVals[ %d ] = %d == %d , checkval j= %d .\n", gIdx, s_vals[tIdx], d_inVals[gIdx], j);
		printf("%d - %d , %d, %d ==> %d, %d, %d .\n", s_vals[tIdx], bMasks[0], bMasks[1], bMasks[2], s_vals[tIdx] & bMasks[0], s_vals[tIdx] & bMasks[1], s_vals[tIdx] & bMasks[2]);
	}
#endif
}

__global__ void incSumScan_kernel(unsigned int* d_outVals, unsigned int* d_inVals, size_t numVals)
{
	unsigned int tIdx = threadIdx.x;
	unsigned int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
	extern __shared__  unsigned int s_incScan[];
	if (gIdx >= numVals) return;
	
	s_incScan[tIdx] = d_inVals[tIdx];
	__syncthreads();

	for (int offset = 1; offset <= numVals; offset = offset * 2)
	{
		unsigned int temp = s_incScan[tIdx];
		unsigned int neighbor = 0;
		if (tIdx >= offset ) {
			neighbor = s_incScan[tIdx - offset];
			__syncthreads();
			s_incScan[tIdx] = temp + neighbor;
		}
		__syncthreads();
	}
	d_outVals[tIdx] = s_incScan[tIdx];
}

//first part of inclusive sum scan of an array larger than a single block.
__global__ void incSumScanB1_kernel(unsigned int* d_outVals, unsigned int* d_inVals, size_t numVals, unsigned int* d_blockOffset, unsigned int valOffset)
{
	unsigned int tIdx = threadIdx.x;
	unsigned int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
	extern __shared__  unsigned int s_incScan[];
	if (gIdx >= numVals) return;
	
	//if it is the first element of a block then we need to add the offset to it.
	s_incScan[tIdx] = (tIdx == 0)? d_inVals[gIdx] + valOffset: d_inVals[gIdx];
	
//	if (tIdx == 0) printf("gIdx =  %d,  d_inVals[ %d ] = %d , s_incScan[ %d ] = %d ,  valOffset = %d .\n", gIdx, gIdx, d_inVals[gIdx], tIdx, s_incScan[tIdx], valOffset);
	__syncthreads();

	//for (int offset = 1; offset <= numVals; offset = offset * 2)
	for (int offset = 1; offset <= blockDim.x; offset = offset * 2)
	{
		unsigned int temp = s_incScan[tIdx];
		unsigned int neighbor = 0;
		if (tIdx >= offset) {
			neighbor = s_incScan[tIdx - offset];
			__syncthreads();
			s_incScan[tIdx] = temp + neighbor;
		}
		__syncthreads();
	}
	d_outVals[gIdx] = s_incScan[tIdx];

	//now set the cumulative sum for this block in the the blockoffsetarray
	if ((tIdx + 1) == blockDim.x)
	{
		if ((blockIdx.x + 1) < gridDim.x)
		{
			d_blockOffset[blockIdx.x + 1] = s_incScan[tIdx]; //this will still need to be summed with other blocks
		}
	}
//	if (gIdx < 10 || gIdx > (numVals - 10)) printf("gIdx =  %d,  d_inVals[ %d ] = %d, d_outvals[ %d ] = %d , s_incScan[ %d ] = %d ,  valOffset = %d .\n",
//		 gIdx, gIdx, d_inVals[gIdx], gIdx, d_outVals[gIdx], tIdx, s_incScan[tIdx], valOffset);
	
}

//finishes the multi-part sumScan of an array larger than blockSize -
__global__ void incSumScanB2_kernel(unsigned int* d_outVals, unsigned int* d_inVals, size_t numVals, unsigned int* d_blockOffset)
{
//	unsigned int tIdx = threadIdx.x;
	unsigned int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
	extern __shared__  unsigned int s_incScan[];
	if (gIdx >= numVals) return;

	d_outVals[gIdx] = ( blockIdx.x > 0) ? d_inVals[gIdx] + d_blockOffset[blockIdx.x]: d_inVals[gIdx];

}

__global__ void arraySet_kernel(unsigned int* d_vals, unsigned int value, size_t num_vals) 
{
//	tIdx = threadIdx.x;
	unsigned int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (gIdx < num_vals) d_vals[gIdx] = value;
}


__global__ void getPredicate_kernel(unsigned int * d_inVal, unsigned int * d_predVal, unsigned int numElems, unsigned int bitMask)
{

	unsigned int gIdx = blockIdx.x * blockDim.x + threadIdx.x;

	if (gIdx < numElems) 
	{
		// if bitmask matches inputvale then assign 1 to the position otherwise set to 0
		// we'll need to run an inclusive scan later to get the position
		d_predVal[gIdx] = ((d_inVal[gIdx] & bitMask) == bitMask) ? 1 : 0;
		//d_npredVal[gIdx] = ((d_inVal[gIdx] & bitMask) == bitMask) ? 0 : 1;
	}
}

__global__ void swapLocations_kernel(unsigned int * d_outVals, unsigned int * d_inVals,
									 unsigned int * d_outPos, unsigned int * d_inPos,
									 unsigned int * d_swapPred, /*unsigned int * d_swapnPred,*/
									 unsigned int numElems, unsigned int bitmask)
{

	unsigned int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int offset = d_swapPred[numElems-1];
	int swapmove;
	__syncthreads();

	if (gIdx < numElems)
	{
		//unsigned int swapmove = ((d_inVals[gIdx] & bitmask) == bitmask) ? d_swapPred[gIdx]-1 : (gIdx - (d_swapPred[gIdx]-1))+offset-1;
		if ((d_inVals[gIdx] & bitmask) == bitmask)
		{
			swapmove = d_swapPred[gIdx] - 1;
			//if (gIdx < 10 || gIdx >(numElems - 10)) printf("gIdx = %d, swapmove = %d .\n", gIdx, swapmove);
			//if (swapmove < 0) swapmove = 0;
		}
		else
		{
			swapmove = (gIdx - (d_swapPred[gIdx] - 1)) + offset-1;
			//if (gIdx < 10 || gIdx >(numElems - 10)) printf("gIdx = %d, swapmove = %d, offset = %d .\n", gIdx, swapmove, offset);
			//if (swapmove < 0) swapmove = 0;
		}


		d_outVals[swapmove] = d_inVals[gIdx];
		d_outPos[swapmove] = d_inPos[gIdx];
//		if (gIdx < 10 || gIdx > (numElems - 10)) {
//			printf("gIdx = %d , bitmask = %08x , offset= %d, swapmove = %d , d_inVals[gIdx] = %d, d_inPos[gIdx] = %d .\n ",
//				gIdx, bitmask, offset, swapmove, d_inVals[gIdx], d_inPos[gIdx]);
//		}
	}
}

__global__ void swapVals_kernel(unsigned int * d_newArray, unsigned int * d_oldArray, unsigned int numElems)
{
	unsigned int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (gIdx < numElems)
	{
		d_newArray[gIdx] = d_oldArray[gIdx];
	}
}

__global__ void reverseSort_kernel(unsigned int * d_newArray, unsigned int * d_oldArray, unsigned int numElems)
{
	unsigned int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (gIdx < numElems)
	{
		d_newArray[gIdx] = d_oldArray[(numElems - 1)- gIdx];
	}
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 

	//inputPos holds original position.
	//outputPos holds the location when resorted by Val


#ifdef DEBUGGING1	  
	std::cout << "Sort of " << numElems << " Elements through " << 8*sizeof(unsigned int)<< " loops." << std::endl;
#endif

	
	unsigned int threadsperblock = 32;

	//Assign Histogram in device
	unsigned int *d_binHistogram;
	//for 32bit integers
	unsigned int numBins = 32; 
	checkCudaErrors(cudaMalloc(&d_binHistogram, numBins*sizeof(unsigned int)));

	//set histogram values to zero	- faster than memcpy?
	dim3 blockSize = { threadsperblock, 1, 1 };
	dim3 gridSize = { (numBins + blockSize.x - 1) / (blockSize.x), 1, 1 };
	arraySet_kernel << <gridSize, blockSize >> > (d_binHistogram, (unsigned int)0, numBins);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	blockSize = { threadsperblock, 1, 1 };
	gridSize = { ((unsigned int)numElems + blockSize.x - 1) / blockSize.x, 1, 1 };
	
//	std::cout << "blocks = " << gridSize.x << " when using " << blockSize.x << " threads per block ." << std::endl;
	
	lsbHisto_kernel << <gridSize, blockSize, blockSize.x*sizeof(unsigned int) >> > (d_binHistogram, numBins, d_inputVals, numElems);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());


	unsigned int * h_binHistogram = (unsigned int*)std::malloc(numBins * sizeof(unsigned int));
	checkCudaErrors(cudaMemcpy(h_binHistogram, d_binHistogram, numBins * sizeof(unsigned int), cudaMemcpyDeviceToHost));
#ifdef DEBUGGING1	
	std::cout << "h_binHistogram [ ";
	for (unsigned int i = 0; i < numBins-1; i++) {
		std::cout << h_binHistogram[i] << ",";
	}
	std::cout << h_binHistogram[numBins-1] << "]" << std::endl;
#endif


// don't need to add these up - only do one at a time.
//	incSumScan_kernel<< < 1, numBins, numBins * sizeof(unsigned int) >> > (d_binHistogram, d_binHistogram, numBins);
//	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

#ifdef DEBUGGING1	
//	h_binHistogram = (unsigned int*)std::malloc(numBins * sizeof(unsigned int));
//	checkCudaErrors(cudaMemcpy(h_binHistogram, d_binHistogram, numBins * sizeof(unsigned int), cudaMemcpyDeviceToHost));
//	std::cout << "h_binHistogram [ ";
//	for (unsigned int i = 0; i < numBins - 1; i++) {
//		std::cout << h_binHistogram[i] << ",";
//	}
//	std::cout << h_binHistogram[numBins - 1] << "]" << std::endl;

//	free(h_binHistogram);
#endif

	threadsperblock = 1024;
	blockSize = { threadsperblock, 1, 1 };
	gridSize = { ((unsigned int)numElems + blockSize.x - 1) / blockSize.x, 1, 1 };

//	std::cout << "Doing inclusive sumscan in " << gridSize.x << " blocks of " << blockSize.x << " threads." << std::endl;

	unsigned int * d_blockOffsets;
	checkCudaErrors(cudaMalloc(&d_blockOffsets, gridSize.x * sizeof(unsigned int)));

	unsigned int * d_predicates; //, * d_npredicates;
	checkCudaErrors(cudaMalloc(&d_predicates, numElems * sizeof(unsigned int)));
	//checkCudaErrors(cudaMalloc(&d_npredicates, numElems * sizeof(unsigned int)));

	//for (int maskPtr = 0; maskPtr < 32; maskPtr++) //should be to 32
	for (int maskPtr = 0; maskPtr < 32; maskPtr++) //should be to 32
	{
		if (h_binHistogram[maskPtr] > 0)  //don't bother if no elements to be sorted - everything will stay the same
		{
			//rad_sort(d_inputVals, d_outputVals, d_inputPos, d_outputPos, d_predicates, d_npredicates, numElems);
			// is npredicate == gIdx - d_predicates[gIdx] + d_binHistogram[bMasks[maskPtr]]

			arraySet_kernel << <1, gridSize>> > (d_blockOffsets, (unsigned int) 0, gridSize.x);
			cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

			// run predicate on input vals and put result in outputPos
			getPredicate_kernel << <gridSize, blockSize >> > (d_inputVals, d_predicates, numElems, bMasks[maskPtr]);
			cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

			// run inclusive scans on each block, putting Offset total for that block in d_blockOffsets
			incSumScanB1_kernel << < gridSize, blockSize, blockSize.x * sizeof(unsigned int) >> > (d_predicates, d_predicates, numElems, d_blockOffsets, 0);
			cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
			// run inclusive scan on d_blockoffsets
			incSumScan_kernel << < 1, gridSize, gridSize.x * sizeof(unsigned int) >> > (d_blockOffsets, d_blockOffsets, gridSize.x);
			cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

			//finish the sumscan accounting for all blocks
			incSumScanB2_kernel << < gridSize, blockSize >> > (d_predicates, d_predicates, numElems, d_blockOffsets);
			cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
#ifdef DEBUGGING1	
			unsigned int * h_blockOffsets = (unsigned int*)std::malloc(gridSize.x * sizeof(unsigned int));
			checkCudaErrors(cudaMemcpy(h_blockOffsets, d_blockOffsets, gridSize.x * sizeof(unsigned int), cudaMemcpyDeviceToHost));
			std::cout << "h_blockOffsets [ ";
			for (unsigned int i = 0; i < gridSize.x - 1; i++) {
				std::cout << h_blockOffsets[i] << ",";
			}
			std::cout << h_blockOffsets[gridSize.x - 1] << "]" << std::endl;

			free(h_blockOffsets);
#endif
			//do the gathering moving values and positions into new locations on the output arrays
			swapLocations_kernel << < gridSize, blockSize >> > (d_outputVals, d_inputVals, d_outputPos, d_inputPos, d_predicates, numElems, bMasks[maskPtr]);
			cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

			//now move all them back to input locations so we can do it again on next loop
			swapVals_kernel << < gridSize, blockSize >> > (d_inputVals, d_outputVals, numElems);
			cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
			swapVals_kernel << < gridSize, blockSize >> > (d_inputPos, d_outputPos, numElems);
			cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
//			std::cout << "Got to end of loop " << maskPtr << std::endl;
		}
		else
		{
//			printf("skipping loop %d because there are no matches on this bitmask.\n", maskPtr);
		}
	}

	threadsperblock = 1024;
	blockSize = { threadsperblock, 1, 1 };
	gridSize = { ((unsigned int)numElems + blockSize.x - 1) / blockSize.x, 1, 1 };
	//I may have sorted the wrong way!
	reverseSort_kernel << < gridSize, blockSize >> > (d_outputPos, d_inputPos, numElems);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	reverseSort_kernel << < gridSize, blockSize >> > (d_outputVals, d_inputVals, numElems);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

#ifdef DEBUGGING1
	unsigned int * h_Vals = (unsigned int*)std::malloc(numElems * sizeof(unsigned int));
	checkCudaErrors(cudaMemcpy(h_Vals, d_outputVals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	unsigned int * h_Poss = (unsigned int*)std::malloc(numElems * sizeof(unsigned int));
	checkCudaErrors(cudaMemcpy(h_Poss, d_outputPos, numElems * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	std::cout << "Pos, Val, OrigPos \n";
	for (unsigned int i = 0; i < numElems; i++) {
		std::cout << i <<","<< h_Vals[i] << "," << h_Poss[i] << "," << std::endl;
	}
	

	free(h_Vals);
	free(h_Poss);
#endif

	free(h_binHistogram);
	checkCudaErrors(cudaFree(d_binHistogram));
	checkCudaErrors(cudaFree(d_blockOffsets));
}