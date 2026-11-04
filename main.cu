#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <sys/stat.h>
#include <time.h>
#include <cuda_runtime.h>
#include "dataset.h"
#include "layers.cuh"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

#define TOTAL_RAW_TRAIN 60000
#define TRAIN_SAMPLES   50000
#define VAL_SAMPLES     10000
#define TEST_SAMPLES    10000

#define N_INPUT         784
#define N_HIDDEN        128
#define N_CLASSES       10

#define BATCH_SIZE      100
#define EVAL_BATCH_SIZE 1000
#define NUM_EPOCHS      10
#define LEARNING_RATE   0.08f

typedef struct {
    float loss;
    float acc;
} EvalMetrics;

static void shuffle_indices(int* arr, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

static EvalMetrics evaluate_metrics(const float* d_x, const float* d_y, int total_samples,
                                    const float* d_w1, const float* d_b1, const float* d_w2, const float* d_b2,
                                    float* d_z1, float* d_a1, float* d_z2, float* d_p,
                                    float* d_loss, int* d_correct) {
    CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_correct, 0, sizeof(int)));

    dim3 block2d(16, 16);
    dim3 block1d(256);

    for (int offset = 0; offset < total_samples; offset += EVAL_BATCH_SIZE) {
        int cur_batch = (offset + EVAL_BATCH_SIZE <= total_samples)
                        ? EVAL_BATCH_SIZE
                        : (total_samples - offset);

        dim3 grid_l1((N_HIDDEN + 15) / 16, (cur_batch + 15) / 16);
        dim3 grid_relu((cur_batch * N_HIDDEN + 255) / 256);
        dim3 grid_l2((N_CLASSES + 15) / 16, (cur_batch + 15) / 16);
        dim3 grid_out_2d((N_CLASSES + 15) / 16, (cur_batch + 15) / 16);
        dim3 grid_batch_1d((cur_batch + 255) / 256);

        const float* batch_x = d_x + (offset * N_INPUT);
        const float* batch_y = d_y + (offset * N_CLASSES);

        k_forward_linear<<<grid_l1, block2d>>>(batch_x, d_w1, d_b1, d_z1, cur_batch, N_INPUT, N_HIDDEN);
        k_relu<<<grid_relu, block1d>>>(d_z1, d_a1, cur_batch * N_HIDDEN);
        k_forward_linear<<<grid_l2, block2d>>>(d_a1, d_w2, d_b2, d_z2, cur_batch, N_HIDDEN, N_CLASSES);
        k_softmax<<<grid_out_2d, block2d>>>(d_z2, d_p, cur_batch, N_CLASSES);

        k_accumulate_loss<<<grid_batch_1d, block1d>>>(d_p, batch_y, d_loss, cur_batch, N_CLASSES);
        k_count_correct<<<grid_batch_1d, block1d>>>(d_p, batch_y, d_correct, cur_batch, N_CLASSES);
        CUDA_CHECK_LAST();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_loss = 0.0f;
    int h_correct = 0;
    CUDA_CHECK(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_correct, d_correct, sizeof(int), cudaMemcpyDeviceToHost));

    EvalMetrics m;
    m.loss = h_loss / (float)total_samples;
    m.acc  = ((float)h_correct / (float)total_samples) * 100.0f;
    return m;
}

static void save_model_weights(const char* filepath, const float* d_w1, const float* d_b1,
                               const float* d_w2, const float* d_b2) {
    mkdir("models", 0777);
    FILE* f = fopen(filepath, "wb");
    if (!f) {
        fprintf(stderr, "Error: Could not open %s for writing\n", filepath);
        return;
    }

    float* h_w1 = (float*)malloc(N_INPUT * N_HIDDEN * sizeof(float));
    float* h_b1 = (float*)malloc(N_HIDDEN * sizeof(float));
    float* h_w2 = (float*)malloc(N_HIDDEN * N_CLASSES * sizeof(float));
    float* h_b2 = (float*)malloc(N_CLASSES * sizeof(float));

    if (!h_w1 || !h_b1 || !h_w2 || !h_b2) {
        fclose(f);
        free(h_w1); free(h_b1); free(h_w2); free(h_b2);
        return;
    }

    CUDA_CHECK(cudaMemcpy(h_w1, d_w1, N_INPUT * N_HIDDEN * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b1, d_b1, N_HIDDEN * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_w2, d_w2, N_HIDDEN * N_CLASSES * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b2, d_b2, N_CLASSES * sizeof(float), cudaMemcpyDeviceToHost));

    fwrite(h_w1, sizeof(float), N_INPUT * N_HIDDEN, f);
    fwrite(h_b1, sizeof(float), N_HIDDEN, f);
    fwrite(h_w2, sizeof(float), N_HIDDEN * N_CLASSES, f);
    fwrite(h_b2, sizeof(float), N_CLASSES, f);

    fclose(f);
    free(h_w1); free(h_b1); free(h_w2); free(h_b2);
}

static void load_model_weights(const char* filepath, float* d_w1, float* d_b1,
                               float* d_w2, float* d_b2) {
    FILE* f = fopen(filepath, "rb");
    if (!f) {
        fprintf(stderr, "Error: Could not open checkpoint %s\n", filepath);
        exit(EXIT_FAILURE);
    }

    float* h_w1 = (float*)malloc(N_INPUT * N_HIDDEN * sizeof(float));
    float* h_b1 = (float*)malloc(N_HIDDEN * sizeof(float));
    float* h_w2 = (float*)malloc(N_HIDDEN * N_CLASSES * sizeof(float));
    float* h_b2 = (float*)malloc(N_CLASSES * sizeof(float));

    if (!h_w1 || !h_b1 || !h_w2 || !h_b2) {
        fclose(f);
        exit(EXIT_FAILURE);
    }

    if (fread(h_w1, sizeof(float), N_INPUT * N_HIDDEN, f) != N_INPUT * N_HIDDEN ||
        fread(h_b1, sizeof(float), N_HIDDEN, f) != N_HIDDEN ||
        fread(h_w2, sizeof(float), N_HIDDEN * N_CLASSES, f) != N_HIDDEN * N_CLASSES ||
        fread(h_b2, sizeof(float), N_CLASSES, f) != N_CLASSES) {
        fprintf(stderr, "Error: Corrupted checkpoint file %s\n", filepath);
        fclose(f);
        free(h_w1); free(h_b1); free(h_w2); free(h_b2);
        exit(EXIT_FAILURE);
    }
    fclose(f);

    CUDA_CHECK(cudaMemcpy(d_w1, h_w1, N_INPUT * N_HIDDEN * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1, h_b1, N_HIDDEN * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w2, h_w2, N_HIDDEN * N_CLASSES * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2, h_b2, N_CLASSES * sizeof(float), cudaMemcpyHostToDevice));

    free(h_w1); free(h_b1); free(h_w2); free(h_b2);
}

int main(int argc, char* argv[]) {
    srand(1337);

    mkdir("benchmarks", 0777);
    char csv_path[512] = "benchmarks/training_log.csv";
    if (argc > 1 && argv[1][0] != '\0') {
        if (strchr(argv[1], '/') != NULL) {
            snprintf(csv_path, sizeof(csv_path), "%s", argv[1]);
        } else {
            snprintf(csv_path, sizeof(csv_path), "benchmarks/%s", argv[1]);
        }
        if (strlen(csv_path) < 4 || strcmp(csv_path + strlen(csv_path) - 4, ".csv") != 0) {
            strncat(csv_path, ".csv", sizeof(csv_path) - strlen(csv_path) - 1);
        }
    }

    const char* weights_path = "models/mnist_mlp_weights.bin";

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("===================================================================================\n");
    printf("   MNIST Neural Network Training Engine - %s\n", prop.name);
    printf("   Architecture: 784 (Input) -> 128 (ReLU) -> 10 (Softmax)\n");
    printf("   Train: %d | Validation: %d | Unseen Test: %d (Locked)\n",
           TRAIN_SAMPLES, VAL_SAMPLES, TEST_SAMPLES);
    printf("   Batch Size: %d | Epochs: %d | Learning Rate: %.2f\n",
           BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE);
    printf("===================================================================================\n\n");

    float *h_train_images = (float*)malloc(TOTAL_RAW_TRAIN * N_INPUT * sizeof(float));
    float *h_train_labels = (float*)malloc(TOTAL_RAW_TRAIN * N_CLASSES * sizeof(float));
    float *h_test_images  = (float*)malloc(TEST_SAMPLES * N_INPUT * sizeof(float));
    float *h_test_labels  = (float*)malloc(TEST_SAMPLES * N_CLASSES * sizeof(float));

    if (!h_train_images || !h_train_labels || !h_test_images || !h_test_labels) {
        fprintf(stderr, "Host memory allocation failed!\n");
        exit(EXIT_FAILURE);
    }

    if (load_and_normalize_images("data/train_images.bin", h_train_images, TOTAL_RAW_TRAIN, NULL) != 0 ||
        load_and_encode_labels("data/train_labels.bin", h_train_labels, TOTAL_RAW_TRAIN) != 0 ||
        load_and_normalize_images("data/test_images.bin", h_test_images, TEST_SAMPLES, NULL) != 0 ||
        load_and_encode_labels("data/test_labels.bin", h_test_labels, TEST_SAMPLES) != 0) {
        fprintf(stderr, "Failed to load MNIST dataset files!\n");
        exit(EXIT_FAILURE);
    }

    float *d_train_x, *d_train_y, *d_val_x, *d_val_y, *d_test_x, *d_test_y;
    CUDA_CHECK(cudaMalloc(&d_train_x, TRAIN_SAMPLES * N_INPUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_train_y, TRAIN_SAMPLES * N_CLASSES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_val_x,   VAL_SAMPLES   * N_INPUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_val_y,   VAL_SAMPLES   * N_CLASSES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_test_x,  TEST_SAMPLES  * N_INPUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_test_y,  TEST_SAMPLES  * N_CLASSES * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_train_x, h_train_images,
                          TRAIN_SAMPLES * N_INPUT * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_train_y, h_train_labels,
                          TRAIN_SAMPLES * N_CLASSES * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_val_x, h_train_images + (TRAIN_SAMPLES * N_INPUT),
                          VAL_SAMPLES * N_INPUT * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_val_y, h_train_labels + (TRAIN_SAMPLES * N_CLASSES),
                          VAL_SAMPLES * N_CLASSES * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_test_x, h_test_images,
                          TEST_SAMPLES * N_INPUT * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_test_y, h_test_labels,
                          TEST_SAMPLES * N_CLASSES * sizeof(float), cudaMemcpyHostToDevice));

    free(h_train_images);
    free(h_train_labels);
    free(h_test_images);
    free(h_test_labels);

    float *d_w1, *d_b1, *d_w2, *d_b2;
    CUDA_CHECK(cudaMalloc(&d_w1, N_INPUT * N_HIDDEN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b1, N_HIDDEN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w2, N_HIDDEN * N_CLASSES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b2, N_CLASSES * sizeof(float)));

    dim3 b_init(16, 16);
    dim3 g_init1((N_HIDDEN + 15) / 16, (N_INPUT + 15) / 16);
    k_init_weights_he<<<g_init1, b_init>>>(d_w1, N_HIDDEN, N_INPUT, 42ULL);
    CUDA_CHECK_LAST();

    dim3 g_init2((N_CLASSES + 15) / 16, (N_HIDDEN + 15) / 16);
    k_init_weights_he<<<g_init2, b_init>>>(d_w2, N_CLASSES, N_HIDDEN, 84ULL);
    CUDA_CHECK_LAST();

    CUDA_CHECK(cudaMemset(d_b1, 0, N_HIDDEN * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_b2, 0, N_CLASSES * sizeof(float)));

    float *d_batch_x, *d_batch_y, *d_z1, *d_a1, *d_z2, *d_p;
    float *d_dz2, *d_dw2, *d_db2, *d_da1, *d_dz1, *d_dw1, *d_db1;
    int   *d_indices, *d_correct;
    float *d_epoch_loss;

    CUDA_CHECK(cudaMalloc(&d_batch_x, BATCH_SIZE * N_INPUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_batch_y, BATCH_SIZE * N_CLASSES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_z1,      EVAL_BATCH_SIZE * N_HIDDEN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_a1,      EVAL_BATCH_SIZE * N_HIDDEN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_z2,      EVAL_BATCH_SIZE * N_CLASSES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p,       EVAL_BATCH_SIZE * N_CLASSES * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_dz2,     BATCH_SIZE * N_CLASSES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dw2,     N_HIDDEN * N_CLASSES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db2,     N_CLASSES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_da1,     BATCH_SIZE * N_HIDDEN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dz1,     BATCH_SIZE * N_HIDDEN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dw1,     N_INPUT * N_HIDDEN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db1,     N_HIDDEN * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_indices,    TRAIN_SAMPLES * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_correct,    sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_epoch_loss, sizeof(float)));

    int* h_indices = (int*)malloc(TRAIN_SAMPLES * sizeof(int));
    for (int i = 0; i < TRAIN_SAMPLES; i++) h_indices[i] = i;

    dim3 block2d(16, 16);
    dim3 block1d(256);

    dim3 grid_gather_x((N_INPUT + 15) / 16, (BATCH_SIZE + 15) / 16);
    dim3 grid_gather_y((N_CLASSES + 15) / 16, (BATCH_SIZE + 15) / 16);
    dim3 grid_l1((N_HIDDEN + 15) / 16, (BATCH_SIZE + 15) / 16);
    dim3 grid_l2((N_CLASSES + 15) / 16, (BATCH_SIZE + 15) / 16);
    dim3 grid_out_2d((N_CLASSES + 15) / 16, (BATCH_SIZE + 15) / 16);
    dim3 grid_relu((BATCH_SIZE * N_HIDDEN + 255) / 256);
    dim3 grid_batch_1d((BATCH_SIZE + 255) / 256);
    dim3 grid_dw2((N_CLASSES + 15) / 16, (N_HIDDEN + 15) / 16);
    dim3 grid_db2((N_CLASSES + 255) / 256);
    dim3 grid_da1((N_HIDDEN + 15) / 16, (BATCH_SIZE + 15) / 16);
    dim3 grid_dw1((N_HIDDEN + 15) / 16, (N_INPUT + 15) / 16);
    dim3 grid_db1((N_HIDDEN + 255) / 256);
    dim3 grid_up_w1((N_INPUT * N_HIDDEN + 255) / 256);
    dim3 grid_up_b1((N_HIDDEN + 255) / 256);
    dim3 grid_up_w2((N_HIDDEN * N_CLASSES + 255) / 256);
    dim3 grid_up_b2((N_CLASSES + 255) / 256);

    const int num_train_batches = TRAIN_SAMPLES / BATCH_SIZE;

    EvalMetrics init_val = evaluate_metrics(d_val_x, d_val_y, VAL_SAMPLES,
                                            d_w1, d_b1, d_w2, d_b2,
                                            d_z1, d_a1, d_z2, d_p,
                                            d_epoch_loss, d_correct);
    printf("Initial Validation: Loss = %.4f (ln(10) = 2.3026) | Acc = %.2f%%\n\n",
           init_val.loss, init_val.acc);

    FILE* csv_fp = fopen(csv_path, "w");
    if (csv_fp) {
        fprintf(csv_fp, "epoch,train_loss,train_acc,val_loss,val_acc,epoch_ms,is_best\n");
    }

    printf("------------------------------------------------------------------------------------------------\n");
    printf("  Epoch  |  Train Loss  |  Train Acc  |   Val Loss   |   Val Acc    |  Epoch Time  | Checkpoint\n");
    printf("---------+--------------+-------------+--------------+--------------+--------------+------------\n");

    float best_val_acc  = -1.0f;
    float best_val_loss = FLT_MAX;
    int   best_epoch    = 0;

    for (int epoch = 1; epoch <= NUM_EPOCHS; epoch++) {
        struct timespec epoch_start, epoch_end;
        clock_gettime(CLOCK_MONOTONIC, &epoch_start);

        shuffle_indices(h_indices, TRAIN_SAMPLES);
        CUDA_CHECK(cudaMemcpy(d_indices, h_indices, TRAIN_SAMPLES * sizeof(int), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemset(d_epoch_loss, 0, sizeof(float)));
        CUDA_CHECK(cudaMemset(d_correct, 0, sizeof(int)));

        for (int step = 0; step < num_train_batches; step++) {
            int offset = step * BATCH_SIZE;

            k_gather_batch<<<grid_gather_x, block2d>>>(d_train_x, d_batch_x, d_indices + offset, BATCH_SIZE, N_INPUT);
            k_gather_batch<<<grid_gather_y, block2d>>>(d_train_y, d_batch_y, d_indices + offset, BATCH_SIZE, N_CLASSES);

            k_forward_linear<<<grid_l1, block2d>>>(d_batch_x, d_w1, d_b1, d_z1, BATCH_SIZE, N_INPUT, N_HIDDEN);
            k_relu<<<grid_relu, block1d>>>(d_z1, d_a1, BATCH_SIZE * N_HIDDEN);
            k_forward_linear<<<grid_l2, block2d>>>(d_a1, d_w2, d_b2, d_z2, BATCH_SIZE, N_HIDDEN, N_CLASSES);
            k_softmax<<<grid_out_2d, block2d>>>(d_z2, d_p, BATCH_SIZE, N_CLASSES);

            k_accumulate_loss<<<grid_batch_1d, block1d>>>(d_p, d_batch_y, d_epoch_loss, BATCH_SIZE, N_CLASSES);
            k_count_correct<<<grid_batch_1d, block1d>>>(d_p, d_batch_y, d_correct, BATCH_SIZE, N_CLASSES);

            k_cross_entropy_backward<<<grid_out_2d, block2d>>>(d_p, d_batch_y, d_dz2, BATCH_SIZE, N_CLASSES);
            k_compute_dw<<<grid_dw2, block2d>>>(d_a1, d_dz2, d_dw2, BATCH_SIZE, N_HIDDEN, N_CLASSES);
            k_compute_db<<<grid_db2, block1d>>>(d_dz2, d_db2, BATCH_SIZE, N_CLASSES);
            k_compute_da<<<grid_da1, block2d>>>(d_dz2, d_w2, d_da1, BATCH_SIZE, N_HIDDEN, N_CLASSES);
            k_relu_backward<<<grid_relu, block1d>>>(d_z1, d_da1, d_dz1, BATCH_SIZE * N_HIDDEN);
            k_compute_dw<<<grid_dw1, block2d>>>(d_batch_x, d_dz1, d_dw1, BATCH_SIZE, N_INPUT, N_HIDDEN);
            k_compute_db<<<grid_db1, block1d>>>(d_dz1, d_db1, BATCH_SIZE, N_HIDDEN);

            k_sgd_update<<<grid_up_w1, block1d>>>(d_w1, d_dw1, LEARNING_RATE, N_INPUT * N_HIDDEN);
            k_sgd_update<<<grid_up_b1, block1d>>>(d_b1, d_db1, LEARNING_RATE, N_HIDDEN);
            k_sgd_update<<<grid_up_w2, block1d>>>(d_w2, d_dw2, LEARNING_RATE, N_HIDDEN * N_CLASSES);
            k_sgd_update<<<grid_up_b2, block1d>>>(d_b2, d_db2, LEARNING_RATE, N_CLASSES);

            CUDA_CHECK_LAST();
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        clock_gettime(CLOCK_MONOTONIC, &epoch_end);
        double epoch_ms = (epoch_end.tv_sec - epoch_start.tv_sec) * 1000.0 +
                          (epoch_end.tv_nsec - epoch_start.tv_nsec) / 1000000.0;

        float h_epoch_loss = 0.0f;
        int h_train_correct = 0;
        CUDA_CHECK(cudaMemcpy(&h_epoch_loss, d_epoch_loss, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_train_correct, d_correct, sizeof(int), cudaMemcpyDeviceToHost));
        float avg_train_loss = h_epoch_loss / (float)TRAIN_SAMPLES;
        float train_acc = ((float)h_train_correct / (float)TRAIN_SAMPLES) * 100.0f;

        EvalMetrics val_metrics = evaluate_metrics(d_val_x, d_val_y, VAL_SAMPLES,
                                                   d_w1, d_b1, d_w2, d_b2,
                                                   d_z1, d_a1, d_z2, d_p,
                                                   d_epoch_loss, d_correct);

        int is_best = 0;
        if (val_metrics.acc > best_val_acc ||
            (val_metrics.acc == best_val_acc && val_metrics.loss < best_val_loss)) {
            best_val_acc  = val_metrics.acc;
            best_val_loss = val_metrics.loss;
            best_epoch    = epoch;
            is_best       = 1;
            save_model_weights(weights_path, d_w1, d_b1, d_w2, d_b2);
        }

        printf("  [%2d/%2d] |    %8.4f  |   %6.2f%%   |    %8.4f  |    %6.2f%%   |   %7.1f ms | %s\n",
               epoch, NUM_EPOCHS, avg_train_loss, train_acc,
               val_metrics.loss, val_metrics.acc, epoch_ms,
               is_best ? " * saved" : "");

        if (csv_fp) {
            fprintf(csv_fp, "%d,%.6f,%.4f,%.6f,%.4f,%.3f,%d\n",
                    epoch, avg_train_loss, train_acc,
                    val_metrics.loss, val_metrics.acc, epoch_ms, is_best);
        }
    }

    printf("------------------------------------------------------------------------------------------------\n");
    if (csv_fp) {
        fclose(csv_fp);
        printf("[OK] Saved training log: %s\n", csv_path);
    }
    printf("[OK] Best checkpoint (Epoch %d | Val Loss: %.4f | Val Acc: %.2f%%) -> %s\n",
           best_epoch, best_val_loss, best_val_acc, weights_path);

    load_model_weights(weights_path, d_w1, d_b1, d_w2, d_b2);

    EvalMetrics final_test = evaluate_metrics(d_test_x, d_test_y, TEST_SAMPLES,
                                              d_w1, d_b1, d_w2, d_b2,
                                              d_z1, d_a1, d_z2, d_p,
                                              d_epoch_loss, d_correct);

    printf("\n===================================================================================\n");
    printf("   FINAL TEST EVALUATION (10,000 Unseen Images @ Best Val Epoch %d)\n", best_epoch);
    printf("   Test Loss: %.4f | Test Accuracy: %.2f%%\n", final_test.loss, final_test.acc);
    printf("===================================================================================\n\n");

    free(h_indices);
    CUDA_CHECK(cudaFree(d_train_x));
    CUDA_CHECK(cudaFree(d_train_y));
    CUDA_CHECK(cudaFree(d_val_x));
    CUDA_CHECK(cudaFree(d_val_y));
    CUDA_CHECK(cudaFree(d_test_x));
    CUDA_CHECK(cudaFree(d_test_y));
    CUDA_CHECK(cudaFree(d_w1));
    CUDA_CHECK(cudaFree(d_b1));
    CUDA_CHECK(cudaFree(d_w2));
    CUDA_CHECK(cudaFree(d_b2));
    CUDA_CHECK(cudaFree(d_batch_x));
    CUDA_CHECK(cudaFree(d_batch_y));
    CUDA_CHECK(cudaFree(d_z1));
    CUDA_CHECK(cudaFree(d_a1));
    CUDA_CHECK(cudaFree(d_z2));
    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_dz2));
    CUDA_CHECK(cudaFree(d_dw2));
    CUDA_CHECK(cudaFree(d_db2));
    CUDA_CHECK(cudaFree(d_da1));
    CUDA_CHECK(cudaFree(d_dz1));
    CUDA_CHECK(cudaFree(d_dw1));
    CUDA_CHECK(cudaFree(d_db1));
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_correct));
    CUDA_CHECK(cudaFree(d_epoch_loss));

    return 0;
}
