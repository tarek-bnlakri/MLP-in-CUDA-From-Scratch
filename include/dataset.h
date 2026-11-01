#ifndef DATASET_H
#define DATASET_H

#define MNIST_IMAGE_SIZE    784     // 28 * 28 pixels
#define MNIST_NUM_CLASSES   10      // Digits 0 to 9
#define MNIST_TRAIN_SAMPLES 60000
#define MNIST_TEST_SAMPLES  10000

typedef struct {
    double read_time_ms;   // Time spent in fread()
    double norm_time_ms;   // Time spent in OpenMP normalization loop
    double total_time_ms;  // Sum of read and normalization times
} DatasetTiming;

int load_and_normalize_images(const char *filepath, float *output, int num_images, DatasetTiming *timing);
int load_and_encode_labels(const char *filepath, float *output, int num_labels);

#endif // DATASET_H
