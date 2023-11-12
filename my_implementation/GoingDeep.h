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
float *ln_gradient_descent(int rows, int n, float X[rows][n], float *y);
float rand_float();
void ln_print(int n, float X[][n], int rows, int cols);

#endif // !GOING_DEEP
