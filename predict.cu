#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include "layers.cuh"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define N_INPUT   784
#define N_HIDDEN  128
#define N_CLASSES 10

static void print_ascii_digit(const float* img) {
    const char* ramp = " .:-=+*#%@";
    int ramp_len = (int)strlen(ramp);

    printf("\n  +----------------------------------------------------------+\n");
    for (int r = 0; r < 28; r++) {
        printf("  | ");
        for (int c = 0; c < 28; c++) {
            float val = img[r * 28 + c];
            int ramp_idx = (int)(val * (ramp_len - 1));
            if (ramp_idx < 0) ramp_idx = 0;
            if (ramp_idx >= ramp_len) ramp_idx = ramp_len - 1;
            char ch = ramp[ramp_idx];
            printf("%c%c", ch, ch);
        }
        printf(" |\n");
    }
    printf("  +----------------------------------------------------------+\n\n");
}

static void print_prediction_bars(const float* p, int true_label) {
    int pred_digit = 0;
    float max_prob = p[0];
    for (int c = 1; c < N_CLASSES; c++) {
        if (p[c] > max_prob) {
            max_prob = p[c];
            pred_digit = c;
        }
    }

    printf("  Predicted Class Probabilities:\n");
    printf("  ---------------------------------------------------------\n");
    for (int c = 0; c < N_CLASSES; c++) {
        int bar_width = (int)(p[c] * 36.0f);
        char bar[40];
        for (int i = 0; i < bar_width; i++) bar[i] = '#';
        bar[bar_width] = '\0';

        const char* marker = (c == pred_digit) ? " <== PREDICTION" : "";
        printf("  Digit %d: [%-36s] %6.2f%%%s\n", c, bar, p[c] * 100.0f, marker);
    }
    printf("  ---------------------------------------------------------\n");

    const char* match_str = (pred_digit == true_label)
                            ? "\033[32m[CORRECT]\033[0m"
                            : "\033[31m[INCORRECT]\033[0m";
    printf("  Result:       %s\n", match_str);
    printf("  Ground Truth: Digit %d\n", true_label);
    printf("  Prediction:   Digit %d (Confidence: %.2f%%)\n", pred_digit, max_prob * 100.0f);
    printf("  =========================================================\n\n");
}

int main(int argc, char* argv[]) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    srand((unsigned int)(ts.tv_nsec ^ ts.tv_sec));

    int test_index = rand() % 10000;
    if (argc > 1 && strcmp(argv[1], "random") != 0 && strcmp(argv[1], "-r") != 0) {
        test_index = atoi(argv[1]);
        if (test_index < 0 || test_index >= 10000) {
            fprintf(stderr, "Error: Test index must be between 0 and 9999 (received %d)\n", test_index);
            return 1;
        }
    }

    printf("=========================================================\n");
    printf("   MNIST CUDA Inference Engine - Interactive Predictor   \n");
    printf("=========================================================\n");
    printf("  Test Image Index: #%d\n", test_index);

    FILE* f_img = fopen("data/test_images.bin", "rb");
    FILE* f_lbl = fopen("data/test_labels.bin", "rb");
    if (!f_img || !f_lbl) {
        fprintf(stderr, "Error: Test dataset files missing. Run python3 data/download_mnist.py first.\n");
        return 1;
    }

    fseek(f_img, (long)test_index * N_INPUT, SEEK_SET);
    fseek(f_lbl, (long)test_index, SEEK_SET);

    unsigned char raw_pixels[N_INPUT];
    unsigned char true_label = 0;
    if (fread(raw_pixels, 1, N_INPUT, f_img) != N_INPUT || fread(&true_label, 1, 1, f_lbl) != 1) {
        fprintf(stderr, "Error reading test image or label at index %d\n", test_index);
        fclose(f_img); fclose(f_lbl);
        return 1;
    }
    fclose(f_img);
    fclose(f_lbl);

    float h_x[N_INPUT];
    for (int i = 0; i < N_INPUT; i++) {
        h_x[i] = (float)raw_pixels[i] / 255.0f;
    }

    print_ascii_digit(h_x);

    const char* weights_path = "models/mnist_mlp_weights.bin";
    FILE* f_w = fopen(weights_path, "rb");
    if (!f_w) {
        fprintf(stderr, "Error: '%s' not found. Run './bin/main' first.\n", weights_path);
        return 1;
    }

    float *h_w1 = (float*)malloc(N_INPUT * N_HIDDEN * sizeof(float));
    float *h_b1 = (float*)malloc(N_HIDDEN * sizeof(float));
    float *h_w2 = (float*)malloc(N_HIDDEN * N_CLASSES * sizeof(float));
    float *h_b2 = (float*)malloc(N_CLASSES * sizeof(float));

    if (fread(h_w1, sizeof(float), N_INPUT * N_HIDDEN, f_w) != N_INPUT * N_HIDDEN ||
        fread(h_b1, sizeof(float), N_HIDDEN, f_w) != N_HIDDEN ||
        fread(h_w2, sizeof(float), N_HIDDEN * N_CLASSES, f_w) != N_HIDDEN * N_CLASSES ||
        fread(h_b2, sizeof(float), N_CLASSES, f_w) != N_CLASSES) {
        fprintf(stderr, "Error: Corrupted weights file '%s'\n", weights_path);
        fclose(f_w);
        return 1;
    }
    fclose(f_w);

    float *d_x, *d_w1, *d_b1, *d_z1, *d_a1, *d_w2, *d_b2, *d_z2, *d_p;
    CUDA_CHECK(cudaMalloc(&d_x,  N_INPUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w1, N_INPUT * N_HIDDEN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b1, N_HIDDEN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_z1, N_HIDDEN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_a1, N_HIDDEN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w2, N_HIDDEN * N_CLASSES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b2, N_CLASSES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_z2, N_CLASSES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p,  N_CLASSES * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_x,  h_x,  N_INPUT * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w1, h_w1, N_INPUT * N_HIDDEN * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1, h_b1, N_HIDDEN * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w2, h_w2, N_HIDDEN * N_CLASSES * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2, h_b2, N_CLASSES * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block2d(16, 16);
    dim3 grid_l1((N_HIDDEN + 15) / 16, 1);
    dim3 grid_l2((N_CLASSES + 15) / 16, 1);
    dim3 grid_out_2d((N_CLASSES + 15) / 16, 1);
    dim3 block1d(256);
    dim3 grid_relu((N_HIDDEN + 255) / 256);

    k_forward_linear<<<grid_l1, block2d>>>(d_x, d_w1, d_b1, d_z1, 1, N_INPUT, N_HIDDEN);
    k_relu<<<grid_relu, block1d>>>(d_z1, d_a1, N_HIDDEN);
    k_forward_linear<<<grid_l2, block2d>>>(d_a1, d_w2, d_b2, d_z2, 1, N_HIDDEN, N_CLASSES);
    k_softmax<<<grid_out_2d, block2d>>>(d_z2, d_p, 1, N_CLASSES);
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_p[N_CLASSES];
    CUDA_CHECK(cudaMemcpy(h_p, d_p, N_CLASSES * sizeof(float), cudaMemcpyDeviceToHost));
    print_prediction_bars(h_p, (int)true_label);

    free(h_w1); free(h_b1); free(h_w2); free(h_b2);
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_w1));
    CUDA_CHECK(cudaFree(d_b1));
    CUDA_CHECK(cudaFree(d_z1));
    CUDA_CHECK(cudaFree(d_a1));
    CUDA_CHECK(cudaFree(d_w2));
    CUDA_CHECK(cudaFree(d_b2));
    CUDA_CHECK(cudaFree(d_z2));
    CUDA_CHECK(cudaFree(d_p));

    return 0;
}
