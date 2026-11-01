#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "dataset.h"

int load_and_normalize_images(const char *filepath, float *output, int num_images, DatasetTiming *timing) {
    size_t total_pixels = (size_t)MNIST_IMAGE_SIZE * num_images;

    unsigned char *raw_bytes = (unsigned char *)malloc(total_pixels);
    if (!raw_bytes) {
        fprintf(stderr, "Error: Memory allocation failed for raw_bytes (%s)\n", filepath);
        return -1;
    }

    FILE *f = fopen(filepath, "rb");
    if (!f) {
        fprintf(stderr, "Error: Couldn't open file (%s)\n", filepath);
        free(raw_bytes);
        return -1;
    }

    // 1. Measure File Reading (fread)
    double t_read_start = omp_get_wtime();
    size_t read_bytes = fread(raw_bytes, 1, total_pixels, f);
    double t_read_end = omp_get_wtime();
    fclose(f);

    if (read_bytes != total_pixels) {
        fprintf(stderr, "Error: Expected %zu bytes but read %zu from %s\n", total_pixels, read_bytes, filepath);
        free(raw_bytes);
        return -1;
    }

    // 2. Measure Normalization (OpenMP Math Loop)
    const float inv_255 = 1.0f / 255.0f;
    double t_norm_start = omp_get_wtime();
    #pragma omp parallel for schedule(dynamic,10000)
    for (size_t pixel = 0; pixel < total_pixels; pixel++) {
        output[pixel] = (float)raw_bytes[pixel] * inv_255;
    }
    double t_norm_end = omp_get_wtime();

    // 3. Store timings if caller requested them
    if (timing != NULL) {
        timing->read_time_ms  = (t_read_end - t_read_start) * 1000.0;
        timing->norm_time_ms  = (t_norm_end - t_norm_start) * 1000.0;
        timing->total_time_ms = timing->read_time_ms + timing->norm_time_ms;
    }

    free(raw_bytes);
    return 0;
}

int load_and_encode_labels(const char *filepath, float *y, int num_labels) {
    unsigned char *raw_labels = (unsigned char *)malloc(num_labels);
    if (!raw_labels) {
        fprintf(stderr, "Error: Memory allocation failed for raw_labels (%s)\n", filepath);
        return -1;
    }

    FILE *f = fopen(filepath, "rb");
    if (!f) {
        fprintf(stderr, "Error: Couldn't open file (%s)\n", filepath);
        free(raw_labels);
        return -1;
    }

    size_t read_bytes = fread(raw_labels, 1, num_labels, f);
    fclose(f);

    if (read_bytes != (size_t)num_labels) {
        fprintf(stderr, "Error: Expected %d bytes but read %zu from %s\n", num_labels, read_bytes, filepath);
        free(raw_labels);
        return -1;
    }

    #pragma omp parallel for
    for (int i = 0; i < num_labels; i++) {
        int value = raw_labels[i];
        for (int s = 0; s < MNIST_NUM_CLASSES; s++) {
            y[i * MNIST_NUM_CLASSES + s] = (s == value) ? 1.0f : 0.0f;
        }
    }

    free(raw_labels);
    return 0;
}
