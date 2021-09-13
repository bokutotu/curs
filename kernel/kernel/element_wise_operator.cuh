#pragma once

#ifdef __cplusplus
extern "C" {
#endif
void float_element_wise_product(float *arrayA, float *arrayB, float *resArray, int size);
void float_element_wise_devide(float *arrayA, float *arrayB, float *resArray, int size);
void double_element_wise_product(double *arrayA, double *arrayB, double *resArray, int size);
void double_element_wise_devide(double *arrayA, double *arrayB, double *resArray, int size);

#ifdef __cplusplus
}
#endif
