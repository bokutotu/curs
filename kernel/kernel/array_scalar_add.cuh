#pragma once

#ifdef __cplusplus
extern "C" {
#endif
void floatArrayScalarAdd(float *array, float *resArray, float scalar, int size);
void doubleArrayScalarAdd(double *array, double *resArray, double scalar, int size);

#ifdef __cplusplus
}
#endif
