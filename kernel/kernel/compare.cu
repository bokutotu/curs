/**
 * @file compare.cu
 * @brief cuda arrayの比較の実装
 * @author HIKARU KONDO
 * @date 2021/07/19
 */
#include <stdio.h>
#include "cuda.h"

#define BLOCKDIM 256

/**
 * @def 
 * Macro to compare against arrays on the GPU
 * @fn
 * Macro to compare against arrays on the GPU
 * @param (comareArrayA) Pointer to the beginning of the array to be compared
 * @param (comareArrayB) Pointer to the beginning of the array to be compared
 * @param (resArray) Pointer to an array to record the result of the comparison.
 * @param (size) Number of elements in the array
 * @detail 
 */
#define COMPARE(FUNCTION, OPERATOR)                                                 \
    template<typename T>                                                            \
    __global__                                                                      \
    void FUNCTION(T *compareArrayA, T *compareArrayB, T *resArray, int size) {  \
        unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;                   \
        if (idx >= size) { return; }                                                \
        resArray[idx] = (compareArrayA[idx] OPERATOR compareArrayB[idx]);           \
    }
COMPARE(equal, ==)
COMPARE(negativeEqual, !=)
COMPARE(greater, >)
COMPARE(less, <)
COMPARE(greaterEqual, >=)
COMPARE(lessEqual, <=)

/**
 * @def 
 * Macro to call compare kernel on host
 * @fn
 * Macro to call compare kernel on device
 * @param (comareArrayA) Pointer to the beginning of the array to be compared
 * @param (comareArrayB) Pointer to the beginning of the array to be compared
 * @param (resArray) Pointer to an array to record the result of the comparison.
 * @param (size) Number of elements in the array
 * @detail 
 */
#define IMLP_COMPARE_FN(FUNCTION, KERNEL, TYPE)                                          \
    void FUNCTION(TYPE *compareArrayA, TYPE *compareArrayB, TYPE *resArray, int size) {  \
      dim3 blockDim(BLOCKDIM);                                                           \
      dim3 gridDim((size + blockDim.x - 1) / blockDim.x);                                \
      KERNEL <<< gridDim, blockDim >>> (compareArrayA, compareArrayB, resArray, size);   \
    }                                                                                    
IMLP_COMPARE_FN(_equalFloat, equal, float)
IMLP_COMPARE_FN(_equalInt, equal, int)
IMLP_COMPARE_FN(_negativeEqualFloat, negativeEqual, float)
IMLP_COMPARE_FN(_negativeEqualInt, negativeEqual, int)
IMLP_COMPARE_FN(_lessFloat, less, float)
IMLP_COMPARE_FN(_lessInt, less, int)
IMLP_COMPARE_FN(_greaterFloat, greater, float)
IMLP_COMPARE_FN(_greaterInt, greater, int)
IMLP_COMPARE_FN(_lessEqualFloat, lessEqual, float)
IMLP_COMPARE_FN(_lessEqualInt, lessEqual, int)
IMLP_COMPARE_FN(_greaterEqualFloat, greaterEqual, float)
IMLP_COMPARE_FN(_greaterEqualInt, greaterEqual, int)

/** 
 * C inter face
 */
extern "C" {
  void equalFloat(float *compareArrayA, float *compareArrayB, float *resArray, int size) {
    _equalFloat(compareArrayA, compareArrayB, resArray, size);
  }

  void equalInt(int *compareArrayA, int *compareArrayB, int *resArray, int size) {
    _equalInt(compareArrayA, compareArrayB, resArray, size);
  }
  
  void negativeEqualFloat(float *compareArrayA, float *compareArrayB, float *resArray, int size) {
    _negativeEqualFloat(compareArrayA, compareArrayB, resArray, size);
  }
  
  void negativeEqualInt(int *compareArrayA, int *compareArrayB, int *resArray, int size) {
    _negativeEqualInt(compareArrayA, compareArrayB, resArray, size);
  }

  void lessFloat(float *compareArrayA, float *compareArrayB, float *resArray, int size) {
    _lessFloat(compareArrayA, compareArrayB, resArray, size);
  }

  void lessInt(int *compareArrayA, int *compareArrayB, int *resArray, int size) {
    _lessInt(compareArrayA, compareArrayB, resArray, size);
  }
}
