/**
 * @file compare.cu
 * @brief cuda arrayの比較の実装
 * @author HIKARU KONDO
 * @date 2021/09/10
 */
#include "transpose.cuh"
#include <stdio.h>
#include "cuda.h"

#define BLOCKDIM 256

/**
 * TODO Doc
**/


template<typename T>
  __global__ void transpose_kernel(T *x, T *y, int size, const int *index_array) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) { return; }
    int transpose_idx = index_array[idx];
    y[transpose_idx] = x[idx];
  }

template<typename T>
  void transpose(T *x, T *y, int size, const int *index_array) {
    dim3 blockDim(BLOCKDIM);
    dim3 gridDim((size + blockDim.x - 1) / blockDim.x);
    int *_index_array;
    cudaMalloc(&_index_array, size * sizeof(int));
    cudaMemcpy(_index_array, index_array, size * sizeof(int), cudaMemcpyHostToDevice);
    transpose_kernel <<< gridDim, blockDim >>> (x, y, size, _index_array); 
    cudaFree(_index_array);
  }

void floatTranspose(float *x, float *y, int size, const int *index_array) {
  transpose(x, y, size, index_array);
}

void doubleTranspose(double *x, double *y, int size, const int *index_array) {
  transpose(x, y, size, index_array);
}
