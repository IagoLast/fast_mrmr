#include "histogram.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <cstdlib>

#define WARP_SIZE	32		// Threads per warp.
#define WARPS_PER_BLOCK 6	// 2.1 max 8 blocks per sm and max 48 warps per sm.
#define THREADS_PER_BLOCK 192 // Warp
#define MAX_BINS 255		// 1KB shared mem per warp. (1 bin = 4 bytes)

//#define THREADS_PER_BLOCK 255
#define MAX_BINS_BLOCK 255


inline __device__ void addByte(byte data, histogram sharedHisto) {
	atomicAdd(&sharedHisto[data], 1);
}

inline __device__ void addWord(histogram sharedHisto, uint fourValuesX) {
	#pragma unroll 4
	for (byte i = 0; i < 4; i++) {
		addByte((byte) (fourValuesX >> (i * 8)), sharedHisto);
	}
}

__global__ void naiveHistoKernel_block(t_data * data_vector, histogram histo,
		const unsigned int datasize) {

	__shared__ unsigned int  sharedHistogram[MAX_BINS_BLOCK];
	sharedHistogram[threadIdx.x] = 0;
	__syncthreads();

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;
	while (i < datasize/16) {
		uint4 fourValuesX = ((uint4 *) data_vector)[i];
		addWord(sharedHistogram, fourValuesX.x);
		addWord(sharedHistogram, fourValuesX.y);
		addWord(sharedHistogram, fourValuesX.z);
		addWord(sharedHistogram, fourValuesX.w);
		i += offset;
	}

	__syncthreads();
// Problema si tenemos mas bins que threads por bloque...
	atomicAdd(&(histo[threadIdx.x]), sharedHistogram[threadIdx.x]);

}

__global__ void naiveHistoKernel_warp(t_data * data_vector, histogram histo,
		const unsigned int datasize) {

	__shared__ unsigned int sharedHistogram[MAX_BINS * WARPS_PER_BLOCK];
	const int tid = threadIdx.x + blockIdx.x * blockDim.x;
	const int wid = threadIdx.x / WARP_SIZE;

// Inicializar la memoria a cero
	for (int i = threadIdx.x; i < MAX_BINS * WARPS_PER_BLOCK; i +=
			THREADS_PER_BLOCK) {
		sharedHistogram[i] = 0;
	}
	__syncthreads();

// Calcular la suma en el histograma local
	int i = tid;
	int offset = blockDim.x * gridDim.x;
	while (i < datasize / 16) {
		uint4 fourValuesX = ((uint4 *) data_vector)[i];
		addWord(&sharedHistogram[MAX_BINS * wid], fourValuesX.x);
		addWord(&sharedHistogram[MAX_BINS * wid], fourValuesX.y);
		addWord(&sharedHistogram[MAX_BINS * wid], fourValuesX.z);
		addWord(&sharedHistogram[MAX_BINS * wid], fourValuesX.w);
		i += offset;
	}
	__syncthreads();

// Reducir el Histograma.
	for (int i = threadIdx.x; i < MAX_BINS; i += THREADS_PER_BLOCK) {
		uint acum = 0;
		for (uint j = 0; j < WARPS_PER_BLOCK; j++) {
			acum += sharedHistogram[MAX_BINS * j + i];
		}
		atomicAdd(&histo[i], acum);
	}

}

extern "C" void histogramNaive(t_data * d_vector, histogram d_hist,
		unsigned int datasize, int blocks) {
//	naiveHistoKernel_block<<<blocks, THREADS_PER_BLOCK>>>(d_vector, d_hist, datasize);
	naiveHistoKernel_warp<<<blocks, THREADS_PER_BLOCK>>>(d_vector, d_hist,datasize);
}
