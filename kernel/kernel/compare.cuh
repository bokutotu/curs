#pragma once


#ifdef __cplusplus
extern "C" {
#endif

void equalFloat(float *compareArrayA, float *compareArrayB, float *resArray, int size);
void equalInt(int *compareArrayA, int *compareArrayB, int *resArray, int size);
void negativeEqualFloat(float *compareArrayA, float *compareArrayB, float *resArray, int size);
void negativeEqualInt(int *compareArrayA, int *compareArrayB, int *resArray, int size);
void lessFloat(float *compareArrayA, float *compareArrayB, float *resArray, int size);
void lessInt(int *compareArrayA, int *compareArrayB, int *resArray, int size);
void greaterFloat(float *compareArrayA, float *compareArrayB, float *resArray, int size);
void greaterInt(int *compareArrayA, int *compareArrayB, int *resArray, int size);
void lessEqualFloat(float *compareArrayA, float *compareArrayB, float *resArray, int size);
void lessEqualInt(int *compareArrayA, int *compareArrayB, int *resArray, int size);
void greaterEqualFloat(float *compareArrayA, float *compareArrayB, float *resArray, int size);
void greaterEqualInt(int *compareArrayA, int *compareArrayB, int *resArray, int size);
#ifdef __cplusplus
}
#endif
