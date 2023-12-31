#include "GoingDeep.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

float MSE(float *array, float a, float b, size_t size) {
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

float *LinearRegression(int rows, int n, float X[rows][n], float *y,
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

  int const cols = n + 1;
  float X_b[rows][cols];
  float eta = 0.1f;

  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      X_b[i][j] = X[i][j];
    }
    X_b[i][cols - 1] = 1.0f;
  }

  float *theta = (float *)malloc((n + 1) * sizeof(float));

  for (int i = 0; i < n + 1; ++i) {
    theta[i] = rand_float();
  }

  assert((sizeof(theta) / sizeof(theta[0])) == cols);

  for (size_t i = 0; i < 1000; ++i) {

    float X_b_dot_theta[rows];
    float X_b_T[cols][rows];

    for (size_t i = 0; i < rows; ++i) {
      X_b_dot_theta[i] = 0.0f;
      for (size_t j = 0; j < cols; ++j) {
        X_b_dot_theta[i] += X_b[i][j] * theta[j];
      }
    }

    for (size_t i = 0; i < cols; ++i) {
      for (size_t j = 0; j < rows; ++j) {
        X_b_T[i][j] = 0.0f;
        X_b_T[i][j] = X_b[j][i];
      }
    }

    int T_rows = sizeof(X_b_T) / sizeof(X_b_T[0]);
    int T_cols = sizeof(X_b_T[0]) / sizeof(X_b_T[0][0]);
    int X_b_dot_rows = sizeof(X_b_dot_theta) / sizeof(X_b_dot_theta[0]);

    assert((sizeof(X_b_dot_theta) / sizeof(X_b_dot_theta[0])) == T_cols);

    // (T_rows x T_cols) * (X_b_dot_rows x X_b_dot_cols) = ()
    float T_dot_product[T_rows];

    for (size_t i = 0; i < T_rows; ++i) {
      T_dot_product[i] = 0.0f;
      for (size_t j = 0; j < T_cols; ++j) {
        T_dot_product[i] += X_b_T[i][j] * (X_b_dot_theta[j] - y[j]);
      }
    }

    float gradients[T_rows];

    for (size_t i = 0; i < T_rows; ++i) {
      gradients[i] = (2 / (float)train_count) * T_dot_product[i];
      theta[i] = theta[i] - eta * gradients[i];
    }
  }
  float temp = theta[0];
  theta[0] = theta[1];
  theta[1] = temp;

  return theta;
}

void LR_generate_data(float X[][2], float y[][1], int num_samples) {
  for (int i = 0; i < num_samples; ++i) {
    X[i][0] = 1.0f;

    X[i][1] = rand_float();

    y[i][0] = (i < num_samples / 2) ? 0.0f : 1.0f;
  }
}

float* LogisticRegression(int rows, int n, float X[rows][n + 1], float y[rows][1],
                          size_t train_count) {

  /*
   *
   * First get the prediction, sigmoid(xT * theta)
   * Then get the cost
   * Create global transpose and matrix multiply functions (need to reuse them
   * alot)
   *
   */

  float theta[n + 1][1];
  // (rows x n + 1) * (n + 1 x 1)

  float prediction[rows][1];

  for (int i = 0; i < n + 1; ++i) {
    theta[i][0] = rand_float();
  }

  matrixMultiply(rows, n + 1, n + 1, 1, X, theta, prediction);

  for (int i = 0; i < rows; ++i) {
      prediction[i][0] = sigmoidf(prediction[i][0]);
  }

  for (int i = 0; i < rows; ++i) {
      printf("X_theta_dot[%d]: %f\n", i, prediction[i][0]);
  }

  return 0;
}

float LR_cost() {}

void matrixMultiply(size_t rowsA, size_t colsA, size_t rowsB, size_t colsB,
                    float A[rowsA][colsA], float B[rowsB][colsB],
                    float dotProduct[rowsA][colsB]) {

  // (rowsA x colsA) * (rowsB x colsB)
  assert(rowsB == colsA);

  for (size_t i = 0; i < rowsA; ++i) {
    for (size_t j = 0; j < colsB; ++j) {
      dotProduct[i][j] = 0.0f;
      for (size_t k = 0; k < colsA; ++k) {
        dotProduct[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

void matrixTranspose(int cols, int rows, float X[rows][cols],
                     float result[cols][rows]) {
  for (size_t i = 0; i < cols; ++i) {
    for (size_t j = 0; j < rows; ++j) {
      result[i][j] = X[j][i];
    }
  }
}

float rand_float() { return (float)rand() / (float)RAND_MAX; }
float sigmoidf(float x) { return 1.f / (1.f + expf(-x)); }
