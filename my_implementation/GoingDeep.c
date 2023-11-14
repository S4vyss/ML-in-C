#include "GoingDeep.h"
#include <assert.h>
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
  printf("X_b Structure:\n");
  printf("[ \n");

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      printf("%8.2f ", X[i][j]);
    }
    printf("\n");
  }

  printf("]\n");
}

float *ln_gradient_descent(int rows, int n, float X[rows][n], float *y,
                           size_t train_count) {
  /*
   *
   * first getting the X_b done
   * second dot product of X_b and theta done
   * third transpose X_b done
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

  float theta[2] = {rand_float(), rand_float()};

  assert((sizeof(theta) / sizeof(theta[0])) == cols);

  float X_b_dot_theta[rows];

  for (size_t i = 0; i < rows; ++i) {
    X_b_dot_theta[i] = 0.0f;
  }

  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      X_b_dot_theta[i] += X_b[i][j] * theta[j];
    }
  }

  float X_b_T[cols][rows];

  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      X_b_T[j][i] = 0.0f;
    }
  }

  for (size_t i = 0; i < cols; ++i) {
    for (size_t j = 0; j < rows; ++j) {
      X_b_T[i][j] = X_b[j][i];
    }
  }

  int T_rows = sizeof(X_b_T) / sizeof(X_b_T[0]);
  int T_cols = sizeof(X_b_T[0]) / sizeof(X_b_T[0][0]);
  int X_b_dot_rows = sizeof(X_b_dot_theta) / sizeof(X_b_dot_theta[0]);

  ln_print(T_cols, X_b_T, T_rows, T_cols);

  assert((sizeof(X_b_dot_theta) / sizeof(X_b_dot_theta[0])) == T_cols);

  // (T_rows x T_cols) * (X_b_dot_rows x X_b_dot_cols) = ()
  float T_dot_product[T_rows];

  for (size_t i = 0; i < T_rows; ++i) {
    T_dot_product[i] = 0.0f;
  }

  for (size_t i = 0; i < T_rows; ++i) {
    for (size_t j = 0; j < T_cols; ++j) {
      T_dot_product[i] += X_b_T[i][j] * X_b_dot_theta[j];
    }
  }

  for (size_t i = 0; i < 2; ++i) {
    theta[i] = (2 / 10) * T_dot_product[i];
  }

  for (size_t i = 0; i < T_rows; ++i) {
    printf("%f\n", T_dot_product[i]);
  }
}
float rand_float() { return (float)rand() / (float)RAND_MAX; }
