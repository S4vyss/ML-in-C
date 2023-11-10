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

void matrixMultiplication(float *matrix, float *vector, float *result, int rows,
                          int cols) {
  for (int i = 0; i < rows; ++i) {
    result[i] = 0.0f;
    for (int j = 0; j < cols; ++j) {
      result[i] += matrix[i * cols + j] * vector[j];
    }
  }
}

float dotProduct(float *a, float *b, int size) {
  float result = 0.0;
  for (int i = 0; i < size; ++i) {
    result += a[i] * b[i];
  }
  return result;
}

float *ln_gradient_descent(float *train, size_t train_count) {
  float *theta = malloc(2 * sizeof(float));
  theta[0] = rand_float();
  theta[1] = rand_float();

  float eta = 0.1;
  int rows = 3;
  int cols = 2;

  float X_b[rows * (cols + 1)];
  float y[rows];

  for (int i = 0; i < rows; ++i) {
    X_b[i * (cols + 1)] = 1.0;
    for (int j = 0; j < cols; ++j) {
      X_b[i * (cols + 1) + j + 1] = train[i * cols + j];
    }
    y[i] = train[i * cols + cols];
  }

  for (int i = 0; i < 100 * 1000; ++i) {
    float X_b_dot_theta[rows];
    matrixMultiplication(X_b, theta, X_b_dot_theta, rows, cols);

    float gradients[rows];
    for (int i = 0; i < rows; ++i) {
      gradients[i] = X_b_dot_theta[i] - y[i];
    }

    float X_b_T_dot_gradients[cols];
    float X_b_T[cols * rows];

    matrixMultiplication(X_b_T, gradients, X_b_T_dot_gradients, cols, rows);

    for (int i = 0; i < cols; ++i) {
      theta[i] = theta[i] - eta * (2 / train_count * X_b_T_dot_gradients[i]);
    }
  }
  return theta;
}

float rand_float() { return (float)rand() / (float)RAND_MAX; }
