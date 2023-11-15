#include "GoingDeep.h"
#include <stdio.h>
#include <time.h>

int main() {

  srand(time(0));
  size_t N = 100;

  float X[N][1];
  float y[N];

  for (int i = 0; i < N; ++i) {
    X[i][0] = 2 * rand_float();
    y[i] = 4 + 3 * X[i][0] + rand_float();
  }

  int rows = sizeof(X) / sizeof(X[0]);
  int cols = sizeof(X[0]) / sizeof(X[0][0]);

  float result[2] = ln_gradient_descent(rows, cols, X, y, N);

  printf("%f, %f\n", result[0], result[1]);

  // float *result =
  //     ln_gradient_descent(X, y, rows, cols, train_count, 100 * 1000);

  // printf("%f, %f\n", result[0], result[1]);

  // printf("%f\n", ln_MSE(train, result[0], result[1], train_count));
  return 0;
}
