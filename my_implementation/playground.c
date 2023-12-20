#include "GoingDeep.h"
#include <stdio.h>
#include <time.h>

int main() {

  srand(time(0));

  int num_samples = 10;
  int n = 1;
  float X[num_samples][n + 1];
  float y[num_samples][1];

  LR_generate_data(X, y, num_samples);

  printf("Sample Data:\n");
  for (int i = 0; i < num_samples; ++i) {
    printf("X[%d]: [%.2f, %.2f], y[%d]: %.2f\n", i, X[i][0], X[i][1], i, y[i][0]);
  }

  printf("\nMultiplication\n");
  LogisticRegression(num_samples, n, X, y, num_samples);

  return 0;
}

int main2() {
  size_t N = 100;

  float X[N][1];
  float y[N];

  for (int i = 0; i < N; ++i) {
    X[i][0] = 2 * rand_float();
    y[i] = 4 + 3 * X[i][0] + rand_float();
  }

  int rows = sizeof(X) / sizeof(X[0]);
  int cols = sizeof(X[0]) / sizeof(X[0][0]);

  float *result = LinearRegression(rows, cols, X, y, N);

  printf("%f, %f\n", result[0], result[1]);
}
