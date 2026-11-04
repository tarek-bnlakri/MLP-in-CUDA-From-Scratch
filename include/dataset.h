#ifndef DATASET_H
#define DATASET_H

#ifdef __cplusplus
extern "C" {
#endif

#define MNIST_IMAGE_SIZE    784
#define MNIST_NUM_CLASSES   10
#define MNIST_TRAIN_SAMPLES 60000
#define MNIST_TEST_SAMPLES  10000

typedef struct {
    double read_time_ms;
    double norm_time_ms;
    double total_time_ms;
} DatasetTiming;

int load_and_normalize_images(const char *filepath, float *output, int num_images, DatasetTiming *timing);
int load_and_encode_labels(const char *filepath, float *output, int num_labels);

#ifdef __cplusplus
}
#endif

#endif
