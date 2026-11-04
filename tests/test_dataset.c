#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <omp.h>
#include "dataset.h"

int main() {
    int num_threads = omp_get_max_threads();
    printf("=== Unit Test: Dataset Loader (OpenMP Threads: %d) ===\n\n", num_threads);

    float *images = (float *)malloc(MNIST_TRAIN_SAMPLES * MNIST_IMAGE_SIZE * sizeof(float));
    assert(images != NULL);

    DatasetTiming timing;
    int status = load_and_normalize_images("data/train_images.bin", images, MNIST_TRAIN_SAMPLES, &timing);
    assert(status == 0 && "Image loading failed!");

    for (long long i = 0; i < (long long)MNIST_TRAIN_SAMPLES * MNIST_IMAGE_SIZE; i++) {
        assert(images[i] >= 0.0f && images[i] <= 1.0f && "Pixel value out of [0.0, 1.0] bounds!");
    }
    printf("  [OK] 47,040,000 pixels normalized within [0.0, 1.0] (Read: %.2f ms, Norm: %.2f ms)\n",
           timing.read_time_ms, timing.norm_time_ms);

    float *labels = (float *)malloc(MNIST_TRAIN_SAMPLES * MNIST_NUM_CLASSES * sizeof(float));
    assert(labels != NULL);

    status = load_and_encode_labels("data/train_labels.bin", labels, MNIST_TRAIN_SAMPLES);
    assert(status == 0 && "Label loading failed!");

    for (int i = 0; i < MNIST_TRAIN_SAMPLES; i++) {
        float sum = 0.0f;
        int active_count = 0;
        for (int c = 0; c < MNIST_NUM_CLASSES; c++) {
            float val = labels[i * MNIST_NUM_CLASSES + c];
            assert((val == 0.0f || val == 1.0f) && "One-hot values must be 0.0 or 1.0");
            sum += val;
            if (val == 1.0f) active_count++;
        }
        assert(fabs(sum - 1.0f) < 1e-6f && "One-hot row does not sum to 1.0!");
        assert(active_count == 1 && "Exactly one class must be active!");
    }
    printf("  [OK] 60,000 one-hot label vectors verified\n");

    int expected[5] = {5, 0, 4, 1, 9};
    for (int i = 0; i < 5; i++) {
        assert(labels[i * MNIST_NUM_CLASSES + expected[i]] == 1.0f && "Label sequence mismatch!");
    }
    printf("  [OK] Ground-truth header sequence verified (5, 0, 4, 1, 9)\n\n");

    free(images);
    free(labels);
    return 0;
}
