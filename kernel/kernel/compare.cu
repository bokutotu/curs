/**
 * @file compare.cu
 * @brief cuda arrayの比較の実装
 * @author HIKARU KONDO
 * @date 2021/07/19
 */
#include "cuda.h"

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
    void FUNCTION(T *compareArrayA, T *compareArrayB, bool *resArray, int size) {   \
        unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;                   \
        if (idx >= size) { return; }                                                \
        resArray[idx] = (compareArrayA[idx] OPERATOR compareArrayB[idx]);           \
    }
COMPARE(equal, ==)
COMPARE(negativeEqual, !=)
COMPARE(grater, >)
COMPARE(less, <)
COMPARE(greaterEqual, >=)
COMPARE(lessEqual, <=)
