#ifndef GOING_DEEP
#define GOING_DEEP

#include <stdlib.h>

/* LinearRegression
 *
 *y = a * x + b
 * needs to find a and b
 *
 * */

float ln_MSE(float *array, float a, float b, size_t size);
float *ln_gradient_descent(float *train, size_t train_count);
float dotProduct(float *a, float *b, int size);
void matrixMultiplication(float *matrix, float *vector, float *result, int rows,
                          int cols);
float rand_float();

#endif // !GOING_DEEP
