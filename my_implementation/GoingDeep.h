#ifndef GOING_DEEP
#define GOING_DEEP

#include <stdlib.h>

/* LinearRegression
 *
 *y = a * x + b
 * needs to find a and b
 *
 * */

float MSE(float *array, float a, float b, size_t size);
float *LinearRegression(int rows, int n, float X[rows][n], float *y,
                        size_t train_count);
float rand_float();
void LR_generate_data(float X[][2], float y[], int num_samples);
void ln_print(int n, float X[][n], int rows, int cols);
float LR_cost();
float *LogisticRegression(int rows, int n, float X[rows][n], float *y,
                        size_t train_count);
void matrixTranspose(int cols, int rows, float X[rows][cols], float result[cols][rows]);
void matrixMultiply(size_t rowsA, size_t colsA, size_t rowsB, size_t colsB, float A[rowsA][colsA], float B[rowsB][colsB], float result[rowsA][colsB]);

#endif // !GOING_DEEP
