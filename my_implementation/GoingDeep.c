#include "GoingDeep.h"
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

float ln_MSE(float *array, float a, float b, size_t size) {
  float error = 0.0f;
  for (size_t i = 0; i < size; ++i) {
    float x = array[i];
    float y = array[i + 1];

    float pred = a * x + b;
    float d = pred - y;
    error += d * d;
  }
  return error /= size;
}

void ln_print(int n, float X[][n], int rows, int cols) {
  printf("Array Structure:\n");
  printf("[ \n");

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      printf("%8.2f ", X[i][j]);
      printf("%i", j);
    }
    printf("\n");
  }

  printf("]\n");
}

float *ln_gradient_descent(int rows, int n, float X[rows][n], float *y) {
  /*
   *
   * first getting the X_b
   * second dot product of X_b and theta
   * third transpose X_b
   * fourth dot product of third and second
   * fifth multiply by 2/m
   *
   * */

  // Getting X_b

  int cols = n + 1;
  float X_b[rows][cols];

  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      X_b[i][j] = X[i][j];
      X_b[i][cols - 1] = 1.0f;
    }
  }

  ln_print(cols, X_b, rows, cols);
}

float rand_float() { return (float)rand() / (float)RAND_MAX; }
