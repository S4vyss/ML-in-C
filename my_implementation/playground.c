#include "GoingDeep.h"
#include <stdio.h>
#include <time.h>

float train[] = {
    2, 4, 6, 12, 10, 20,
};

size_t train_count = sizeof(train) / sizeof(train[0]);

int main() {

  srand(time(0));
  float *result = ln_gradient_descent(train, train_count);

  printf("%f\n", ln_MSE(train, result[0], result[1], train_count));
  return 0;
}
