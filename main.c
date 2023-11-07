#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float data[][3] = {
    {1, 2, 6},
    {3, 4, 14},
    {5, 6, 22},
    {7, 8, 30},
};

float randomFloat(void) { return (float)rand() / (float)RAND_MAX; }

float loss(float w) {
  float error = 0.0f;
  size_t train_count = sizeof(data) / sizeof(data[0]);
  for (size_t i = 0; i < train_count; ++i) {
    float a = data[i][0];
    float b = data[i][1];
    float y = data[i][2];
    float pred = (a + b) * w;
    error += (y - pred) * (y - pred);
  }
  error /= train_count;
  return error;
}

int main() {
  srand(time(NULL));
  float w = randomFloat() * 10.0f;
  double best_loss = INFINITY;
  float best_weight = 0.0f;

  for (int i = 0; i < 1000; ++i) {
    float loss_value = loss(w);
    float eps = 1e-3;
    float lr = 1e-3;

    float dcost = (loss(w + eps) - loss(w)) / eps;
    w -= lr * dcost;

    printf("Loss: %f for w = %f. Best loss: %f\n", loss_value, w, best_loss);

    if (loss_value < best_loss) {
      best_loss = loss_value;
      best_weight = w;
    } else {
      break;
    }
  }
  printf("==================\n");
  printf("Final loss: %f. Final weight: %f.", best_loss, best_weight);
  return 0;
}
