#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <omp.h>
#include "dataset.h"

int main() {
    int num_threads = omp_get_max_threads();
    printf("=== Running Unit Test: Dataset Loader ===\n");
    printf("Active OpenMP Threads: %d\n\n", num_threads);

    printf("[1/3] Testing train_images.bin normalization...\n");
    float *images = (float *)malloc(MNIST_TRAIN_SAMPLES * MNIST_IMAGE_SIZE * sizeof(float));
    assert(images != NULL);

    DatasetTiming timing;
    int status = load_and_normalize_images("data/train_images.bin", images, MNIST_TRAIN_SAMPLES, &timing);
    assert(status == 0 && "Image loading failed!");

    // Verify bounds: every pixel must be in [0.0, 1.0]
    for (long long i = 0; i < (long long)MNIST_TRAIN_SAMPLES * MNIST_IMAGE_SIZE; i++) {
        assert(images[i] >= 0.0f && images[i] <= 1.0f && "Pixel value out of [0.0, 1.0] bounds!");
    }
    printf("  ? All 47,040,000 pixels strictly within [0.0, 1.0]\n");
    printf("    -> C-calculated Read Time: %.2f ms\n", timing.read_time_ms);
    printf("    -> C-calculated Norm Time: %.2f ms\n", timing.norm_time_ms);
    printf("    -> C-calculated Total:     %.2f ms\n", timing.total_time_ms);

    // 2. Test Label One-Hot Encoding
    printf("[2/3] Testing train_labels.bin one-hot encoding...\n");
    float *labels = (float *)malloc(MNIST_TRAIN_SAMPLES * MNIST_NUM_CLASSES * sizeof(float));
    assert(labels != NULL);

    status = load_and_encode_labels("data/train_labels.bin", labels, MNIST_TRAIN_SAMPLES);
    assert(status == 0 && "Label loading failed!");

    // Verify one-hot: each row must sum to exactly 1.0
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
    printf("  ? All 60,000 one-hot vectors valid\n");

    // 3. Ground-Truth Check for first 5 labels (5, 0, 4, 1, 9)
    printf("[3/3] Checking ground-truth sequence...\n");
    int expected[5] = {5, 0, 4, 1, 9};
    for (int i = 0; i < 5; i++) {
        assert(labels[i * MNIST_NUM_CLASSES + expected[i]] == 1.0f && "Label sequence mismatch!");
    }
    printf("  ? First 5 labels match ground-truth (5, 0, 4, 1, 9)\n");

    free(images);
    free(labels);
    printf("\n>>> ALL DATASET UNIT TESTS PASSED SUCCESSFULLY! <<<\n");
    return 0;
}
