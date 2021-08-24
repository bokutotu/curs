/**
 * @file compare.cu
 * @brief element wise product 
 * @author HIKARU KONDO
 * @date 2021/08/24
 */
#include <stdio.h>
#include "cuda.h"

#define BLOCKDIM 256


template<typename T>
__global__ void element_wise_product(T *arrayA, T *arrayB, T *resArray, int size) {
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= size) { return ; }
  resArray[idx] = arrayA[idx] * arrayB[idx];
}

template<typename T>
__global__ void element_wise_devide(T *arrayA, T *arrayB, T *resArray, int size) {
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= size) { return ; }
  resArray[idx] = arrayA[idx] / arrayB[idx];
}

extern "C" {
  void float_element_wise_product(float *arrayA, float *arrayB, float *resArray, int size) {
    dim3 blockDim(BLOCKDIM);
    dim3 gridDim((size + blockDim.x - 1) / blockDim.x);
    element_wise_product<<< gridDim, blockDim>>> (arrayA, arrayB, resArray, size);
  }

  void float_element_wise_devide(float *arrayA, float *arrayB, float *resArray, int size) {
    dim3 blockDim(BLOCKDIM);
    dim3 gridDim((size + blockDim.x - 1) / blockDim.x);
    element_wise_devide<<< gridDim, blockDim >>> (arrayA, arrayB, resArray, size);
  }
}
