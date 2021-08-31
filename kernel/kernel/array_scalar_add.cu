/**
 * @file compare.cu
 * @brief cuda arrayの比較の実装
 * @author HIKARU KONDO
 * @date 2021/08/31
 */
#include <stdio.h>
#include "cuda.h"

#define BLOCKDIM 256

template<typename T>
__global__ void arrayAddScalar(T *array, T *resArray, T scalar, int size) {
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= size) { return; }
  resArray[idx] = array[idx] + scalar;
}

extern "C" {
  void floatArrayScalarAdd(float *array, float *resArray, float scalar, int size) {
    dim3 blockDim(BLOCKDIM);
    dim3 gridDim((size + blockDim.x - 1) / blockDim.x);
    arrayAddScalar<<< gridDim, blockDim >>> (array, resArray, scalar, size);
  }

  void doubleArrayScalarAdd(double *array, double *resArray, double scalar, int size) {
    dim3 blockDim(BLOCKDIM);
    dim3 gridDim((size + blockDim.x - 1) / blockDim.x);
    arrayAddScalar<<< gridDim, blockDim >>> (array, resArray, scalar, size);
  }
}
