#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "dataset.h"

int main(int argc, char *argv[]) {
    size_t total_pixels = (size_t)MNIST_TRAIN_SAMPLES * MNIST_IMAGE_SIZE;
    float *images = (float *)malloc(total_pixels * sizeof(float));
    if (!images) {
        fprintf(stderr, "Failed to allocate memory for images\n");
        return 1;
    }

    int thread_counts[] = {1, 2, 4, 6, 8, 12, 16, 20, 24};
    int num_configs = sizeof(thread_counts) / sizeof(thread_counts[0]);

    char csv_path[512] = "benchmarks/cpu_benchmarks/benchmark_results.csv";
    if (argc > 1 && argv[1][0] != '\0') {
        const char *arg = argv[1];
        if (strstr(arg, "benchmarks/cpu_benchmarks/") == arg) {
            snprintf(csv_path, sizeof(csv_path), "%s", arg);
        } else if (strstr(arg, "benchmarks/") == arg) {
            snprintf(csv_path, sizeof(csv_path), "benchmarks/cpu_benchmarks/%s", arg + 11);
        } else {
            snprintf(csv_path, sizeof(csv_path), "benchmarks/cpu_benchmarks/%s", arg);
        }
        if (strlen(csv_path) < 4 || strcmp(csv_path + strlen(csv_path) - 4, ".csv") != 0) {
            strncat(csv_path, ".csv", sizeof(csv_path) - strlen(csv_path) - 1);
        }
    }

    FILE *csv_file = fopen(csv_path, "w");
    if (csv_file) {
        fprintf(csv_file, "threads,read_ms,norm_ms,total_ms,total_speedup,norm_speedup,throughput_mpix_s\n");
    }

    printf("========================================================================================================\n");
    printf("   OpenMP Scaling Benchmark: Disk Read (fread) vs Pixel Normalization (uint8 -> fp32)\n");
    printf("========================================================================================================\n");
    printf("%-8s | %-10s | %-10s | %-18s | %-10s | %-10s | %-12s\n",
           "Threads", "Read (ms)", "Norm (ms)", "Total Time (ms)", "Total S/U", "Norm S/U", "Mpix/s");
    printf("--------------------------------------------------------------------------------------------------------\n");

    omp_set_num_threads(1);
    load_and_normalize_images("data/train_images.bin", images, MNIST_TRAIN_SAMPLES, NULL);

    double baseline_total = 0.0;
    double baseline_norm  = 0.0;

    for (int i = 0; i < num_configs; i++) {
        int threads = thread_counts[i];
        omp_set_num_threads(threads);

        int iters = 5;
        double sum_read_ms = 0.0;
        double sum_norm_ms = 0.0;

        for (int iter = 0; iter < iters; iter++) {
            DatasetTiming timing;
            if (load_and_normalize_images("data/train_images.bin", images, MNIST_TRAIN_SAMPLES, &timing) != 0) {
                fprintf(stderr, "Error loading dataset\n");
                free(images);
                return 1;
            }
            sum_read_ms += timing.read_time_ms;
            sum_norm_ms += timing.norm_time_ms;
        }

        double avg_read_ms  = sum_read_ms / iters;
        double avg_norm_ms  = sum_norm_ms / iters;
        double avg_total_ms = avg_read_ms + avg_norm_ms;

        if (threads == 1) {
            baseline_total = avg_total_ms;
            baseline_norm  = avg_norm_ms;
        }

        double total_speedup = baseline_total / avg_total_ms;
        double norm_speedup  = baseline_norm / avg_norm_ms;
        double throughput    = 47.04 / (avg_total_ms / 1000.0);

        char total_detail[32];
        snprintf(total_detail, sizeof(total_detail), "%.1f+%.1f = %.2f", avg_read_ms, avg_norm_ms, avg_total_ms);

        printf("%-8d | %-10.2f | %-10.2f | %-18s | %-9.2fx | %-9.2fx | %-12.1f\n",
               threads, avg_read_ms, avg_norm_ms, total_detail,
               total_speedup, norm_speedup, throughput);

        if (csv_file) {
            fprintf(csv_file, "%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.1f\n",
                    threads, avg_read_ms, avg_norm_ms, avg_total_ms,
                    total_speedup, norm_speedup, throughput);
        }
    }

    if (csv_file) {
        fclose(csv_file);
        printf("--------------------------------------------------------------------------------------------------------\n");
        printf("[OK] Results saved to: %s\n", csv_path);
    }
    printf("========================================================================================================\n");

    free(images);
    return 0;
}
