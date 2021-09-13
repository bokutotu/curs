/**
 * @file compare.cu
 * @brief cuda arrayの比較の実装
 * @author HIKARU KONDO
 * @date 2021/07/19
 */

#include "compare.cuh"

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
IMLP_COMPARE_FN(equalFloat, equal, float)
IMLP_COMPARE_FN(equalInt, equal, int)
IMLP_COMPARE_FN(negativeEqualFloat, negativeEqual, float)
IMLP_COMPARE_FN(negativeEqualInt, negativeEqual, int)
IMLP_COMPARE_FN(lessFloat, less, float)
IMLP_COMPARE_FN(lessInt, less, int)
IMLP_COMPARE_FN(greaterFloat, greater, float)
IMLP_COMPARE_FN(greaterInt, greater, int)
IMLP_COMPARE_FN(lessEqualFloat, lessEqual, float)
IMLP_COMPARE_FN(lessEqualInt, lessEqual, int)
IMLP_COMPARE_FN(greaterEqualFloat, greaterEqual, float)
IMLP_COMPARE_FN(greaterEqualInt, greaterEqual, int)
