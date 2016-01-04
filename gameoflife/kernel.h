#ifndef __KERNEL_H__
#define __KERNEL_H__

#include "cuda_runtime.h"

#define checkCudaRet(r, m, f, l) if((r) != cudaSuccess) { fprintf(stderr, "Error: %s: %s (%s:%d)\n", m, cudaGetErrorString(r), f, l); exit(-1); } 

__global__ void update_map_d(bool *map_d, bool *updated_map_d, unsigned int rows, unsigned int cols, int block_width, int block_height);
__device__ unsigned int mod(int a, int m);

#endif