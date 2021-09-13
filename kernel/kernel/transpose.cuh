#pragma once

#ifdef __cplusplus
extern "C" {
#endif
void floatTranspose(float *x, float *y, int size, const int *index_array);
void doubleTranspose(double *x, double *y, int size, const int *index_array);
#ifdef __cplusplus
}
#endif
