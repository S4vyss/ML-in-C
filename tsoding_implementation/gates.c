#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float sigmoidf(float x) { return 1.f / (1.f + expf(-x)); }

typedef float sample[3];

// OR - GATE
sample or_train[] = {
    {0, 0, 0},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 1},
};

// AND - GATE
sample and_train[] = {
    {0, 0, 0},
    {1, 0, 0},
    {0, 1, 0},
    {1, 1, 1},
};

// NAND - GATE
sample nand_train[] = {
    {0, 0, 1},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 0},
};

sample *train = or_train;
size_t train_count = 4;

float randomFloat(void) { return (float)rand() / (float)RAND_MAX; }

float loss(float w1, float w2, float bias) {
  float error = 0.0f;
  for (size_t i = 0; i < train_count; ++i) {
    float a = train[i][0];
    float b = train[i][1];
    float y = train[i][2];
    float pred = sigmoidf(a * w1 + b * w2 + bias);
    error += (pred - y) * (pred - y);
  }
  error /= train_count;
  return error;
}

int main2() {
  // XOR = (x|y) & ~(x&y)

  for (size_t x = 0; x < 2; ++x) {
    for (size_t y = 0; y < 2; ++y) {
      printf("%zu ^ %zu = %zu\n", x, y, (x | y) & (~(x & y)));
    }
  }

  return 0;
}

int main() {
  srand(time(0));

  float w1 = randomFloat();
  float w2 = randomFloat();
  float b = randomFloat();

  float rate = 1e-1;
  float eps = 1e-1;

  for (size_t i = 0; i < 100 * 1000; ++i) {
    float cost = loss(w1, w2, b);
    // printf("w1 = %f, w2 = %f, b = %f, c = %f\n", w1, w2, b, cost);
    // printf("%f\n", cost);
    float dw1 = (loss(w1 + eps, w2, b) - cost) / eps;
    float dw2 = (loss(w1, w2 + eps, b) - cost) / eps;
    float db = (loss(w1, w2, b + eps) - cost) / eps;

    w1 -= rate * dw1;
    w2 -= rate * dw2;
    b -= rate * db;
  }

  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      printf("%zu | %zu = %f\n", i, j, sigmoidf(i * w1 + j * w2 + b));
    }
  }

  return 0;
}
