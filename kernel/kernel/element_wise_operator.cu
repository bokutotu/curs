/**
 * @file compare.cu
 * @brief element wise product 
 * @author HIKARU KONDO
 * @date 2021/08/24
 */

#include "element_wise_operator.cuh"

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

void double_element_wise_product(double *arrayA, double *arrayB, double *resArray, int size) {
  dim3 blockDim(BLOCKDIM);
  dim3 gridDim((size + blockDim.x - 1) / blockDim.x);
  element_wise_product<<< gridDim, blockDim>>> (arrayA, arrayB, resArray, size);
}

void double_element_wise_devide(double *arrayA, double *arrayB, double *resArray, int size) {
  dim3 blockDim(BLOCKDIM);
  dim3 gridDim((size + blockDim.x - 1) / blockDim.x);
  element_wise_devide<<< gridDim, blockDim >>> (arrayA, arrayB, resArray, size);
}
