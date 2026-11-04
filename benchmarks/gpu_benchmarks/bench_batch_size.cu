#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

#define N_INPUT       784
#define N_HIDDEN      128
#define N_CLASSES     10
#define MAX_BATCH     16384
#define LEARNING_RATE 0.01f
#define WARMUP_ITERS  50
#define BENCH_ITERS   200

static const double FLOPS_PER_SAMPLE = 409680.0;
static const double FLOPS_FIXED_STEP = 305310.0;

int main(int argc, char* argv[]) {
    char csv_path[512] = "benchmarks/gpu_benchmarks/batch_scaling.csv";
    if (argc > 1 && argv[1][0] != '\0') {
        if (strstr(argv[1], "/") != NULL) {
            snprintf(csv_path, sizeof(csv_path), "%s", argv[1]);
        } else {
            snprintf(csv_path, sizeof(csv_path), "benchmarks/gpu_benchmarks/%s", argv[1]);
        }
        if (strstr(csv_path, ".csv") == NULL) {
            strncat(csv_path, ".csv", sizeof(csv_path) - strlen(csv_path) - 1);
        }
    }

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("========================================================================================\n");
    printf("   GPU Batch Size Scaling Benchmark - %s (%d SMs, %.1f GB VRAM)\n",
           prop.name, prop.multiProcessorCount, (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("   Architecture: 784 -> 128 (ReLU) -> 10 (Softmax) | FP32\n");
    printf("   Timing: Hardware Events (%d warmup, %d measured iterations)\n", WARMUP_ITERS, BENCH_ITERS);
    printf("========================================================================================\n");
    printf(" %-10s | %-9s | %-9s | %-10s | %-18s | %-8s | %-7s\n",
           "Batch Size", "Fwd (ms)", "Bwd (ms)", "Total (ms)", "Throughput (img/s)", "Speedup", "TFLOP/s");
    printf("------------+-----------+-----------+------------+--------------------+----------+---------\n");

    float *d_x, *d_w1, *d_b1, *d_z1, *d_a1;
    float *d_w2, *d_b2, *d_z2, *d_p, *d_y;
    float *d_dz2, *d_dw2, *d_db2, *d_da1, *d_dz1, *d_dw1, *d_db1;

    CUDA_CHECK(cudaMalloc(&d_x,   MAX_BATCH * N_INPUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w1,  N_INPUT * N_HIDDEN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b1,  N_HIDDEN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_z1,  MAX_BATCH * N_HIDDEN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_a1,  MAX_BATCH * N_HIDDEN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w2,  N_HIDDEN * N_CLASSES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b2,  N_CLASSES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_z2,  MAX_BATCH * N_CLASSES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p,   MAX_BATCH * N_CLASSES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y,   MAX_BATCH * N_CLASSES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dz2, MAX_BATCH * N_CLASSES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dw2, N_HIDDEN * N_CLASSES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db2, N_CLASSES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_da1, MAX_BATCH * N_HIDDEN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dz1, MAX_BATCH * N_HIDDEN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dw1, N_INPUT * N_HIDDEN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db1, N_HIDDEN * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dz2, 0, MAX_BATCH * N_CLASSES * sizeof(float)));

    float* h_dummy_x = (float*)malloc(MAX_BATCH * N_INPUT * sizeof(float));
    float* h_dummy_y = (float*)calloc(MAX_BATCH * N_CLASSES, sizeof(float));
    for (int i = 0; i < MAX_BATCH * N_INPUT; i++) h_dummy_x[i] = 0.5f;
    for (int i = 0; i < MAX_BATCH; i++) h_dummy_y[i * N_CLASSES + (i % N_CLASSES)] = 1.0f;

    CUDA_CHECK(cudaMemcpy(d_x, h_dummy_x, MAX_BATCH * N_INPUT * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_dummy_y, MAX_BATCH * N_CLASSES * sizeof(float), cudaMemcpyHostToDevice));
    free(h_dummy_x);
    free(h_dummy_y);

    cudaEvent_t ev_start_fwd, ev_stop_fwd, ev_start_bwd, ev_stop_bwd;
    CUDA_CHECK(cudaEventCreate(&ev_start_fwd));
    CUDA_CHECK(cudaEventCreate(&ev_stop_fwd));
    CUDA_CHECK(cudaEventCreate(&ev_start_bwd));
    CUDA_CHECK(cudaEventCreate(&ev_stop_bwd));

    int batch_sizes[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};
    int num_batches = sizeof(batch_sizes) / sizeof(batch_sizes[0]);

    FILE* fp = fopen(csv_path, "w");
    if (!fp) {
        fprintf(stderr, "Error: Could not open %s for writing\n", csv_path);
        exit(EXIT_FAILURE);
    }
    fprintf(fp, "batch_size,fwd_ms,bwd_ms,total_ms,throughput_img_s,throughput_speedup,tflops\n");

    dim3 b_init(16, 16);
    dim3 g_init1((N_HIDDEN + 15) / 16, (N_INPUT + 15) / 16);
    dim3 g_init2((N_CLASSES + 15) / 16, (N_HIDDEN + 15) / 16);

    double baseline_throughput = 0.0;

    for (int b_idx = 0; b_idx < num_batches; b_idx++) {
        int B = batch_sizes[b_idx];

        k_init_weights_he<<<g_init1, b_init>>>(d_w1, N_HIDDEN, N_INPUT, 42ULL);
        k_init_weights_he<<<g_init2, b_init>>>(d_w2, N_CLASSES, N_HIDDEN, 84ULL);
        CUDA_CHECK(cudaMemset(d_b1, 0, N_HIDDEN * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_b2, 0, N_CLASSES * sizeof(float)));
        CUDA_CHECK_LAST();

        dim3 block2d(16, 16);
        dim3 grid_l1((N_HIDDEN + 15) / 16, (B + 15) / 16);
        dim3 grid_l2((N_CLASSES + 15) / 16, (B + 15) / 16);
        dim3 grid_out_2d((N_CLASSES + 15) / 16, (B + 15) / 16);

        dim3 block1d(256);
        dim3 grid_relu((B * N_HIDDEN + 255) / 256);
        dim3 grid_dw2((N_CLASSES + 15) / 16, (N_HIDDEN + 15) / 16);
        dim3 grid_db2((N_CLASSES + 255) / 256);
        dim3 grid_da1((N_HIDDEN + 15) / 16, (B + 15) / 16);
        dim3 grid_dw1((N_HIDDEN + 15) / 16, (N_INPUT + 15) / 16);
        dim3 grid_db1((N_HIDDEN + 255) / 256);

        dim3 grid_up_w1((N_INPUT * N_HIDDEN + 255) / 256);
        dim3 grid_up_b1((N_HIDDEN + 255) / 256);
        dim3 grid_up_w2((N_HIDDEN * N_CLASSES + 255) / 256);
        dim3 grid_up_b2((N_CLASSES + 255) / 256);

        for (int i = 0; i < WARMUP_ITERS; i++) {
            k_forward_linear<<<grid_l1, block2d>>>(d_x, d_w1, d_b1, d_z1, B, N_INPUT, N_HIDDEN);
            k_relu<<<grid_relu, block1d>>>(d_z1, d_a1, B * N_HIDDEN);
            k_forward_linear<<<grid_l2, block2d>>>(d_a1, d_w2, d_b2, d_z2, B, N_HIDDEN, N_CLASSES);
            k_softmax<<<grid_out_2d, block2d>>>(d_z2, d_p, B, N_CLASSES);

            k_cross_entropy_backward<<<grid_out_2d, block2d>>>(d_p, d_y, d_dz2, B, N_CLASSES);
            k_compute_dw<<<grid_dw2, block2d>>>(d_a1, d_dz2, d_dw2, B, N_HIDDEN, N_CLASSES);
            k_compute_db<<<grid_db2, block1d>>>(d_dz2, d_db2, B, N_CLASSES);
            k_compute_da<<<grid_da1, block2d>>>(d_dz2, d_w2, d_da1, B, N_HIDDEN, N_CLASSES);
            k_relu_backward<<<grid_relu, block1d>>>(d_z1, d_da1, d_dz1, B * N_HIDDEN);
            k_compute_dw<<<grid_dw1, block2d>>>(d_x, d_dz1, d_dw1, B, N_INPUT, N_HIDDEN);
            k_compute_db<<<grid_db1, block1d>>>(d_dz1, d_db1, B, N_HIDDEN);

            k_sgd_update<<<grid_up_w1, block1d>>>(d_w1, d_dw1, LEARNING_RATE, N_INPUT * N_HIDDEN);
            k_sgd_update<<<grid_up_b1, block1d>>>(d_b1, d_db1, LEARNING_RATE, N_HIDDEN);
            k_sgd_update<<<grid_up_w2, block1d>>>(d_w2, d_dw2, LEARNING_RATE, N_HIDDEN * N_CLASSES);
            k_sgd_update<<<grid_up_b2, block1d>>>(d_b2, d_db2, LEARNING_RATE, N_CLASSES);
        }
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaDeviceSynchronize());

        float total_fwd_ms = 0.0f;
        float total_bwd_ms = 0.0f;

        for (int i = 0; i < BENCH_ITERS; i++) {
            CUDA_CHECK(cudaEventRecord(ev_start_fwd));
            k_forward_linear<<<grid_l1, block2d>>>(d_x, d_w1, d_b1, d_z1, B, N_INPUT, N_HIDDEN);
            k_relu<<<grid_relu, block1d>>>(d_z1, d_a1, B * N_HIDDEN);
            k_forward_linear<<<grid_l2, block2d>>>(d_a1, d_w2, d_b2, d_z2, B, N_HIDDEN, N_CLASSES);
            k_softmax<<<grid_out_2d, block2d>>>(d_z2, d_p, B, N_CLASSES);
            CUDA_CHECK(cudaEventRecord(ev_stop_fwd));

            CUDA_CHECK(cudaEventRecord(ev_start_bwd));
            k_cross_entropy_backward<<<grid_out_2d, block2d>>>(d_p, d_y, d_dz2, B, N_CLASSES);
            k_compute_dw<<<grid_dw2, block2d>>>(d_a1, d_dz2, d_dw2, B, N_HIDDEN, N_CLASSES);
            k_compute_db<<<grid_db2, block1d>>>(d_dz2, d_db2, B, N_CLASSES);
            k_compute_da<<<grid_da1, block2d>>>(d_dz2, d_w2, d_da1, B, N_HIDDEN, N_CLASSES);
            k_relu_backward<<<grid_relu, block1d>>>(d_z1, d_da1, d_dz1, B * N_HIDDEN);
            k_compute_dw<<<grid_dw1, block2d>>>(d_x, d_dz1, d_dw1, B, N_INPUT, N_HIDDEN);
            k_compute_db<<<grid_db1, block1d>>>(d_dz1, d_db1, B, N_HIDDEN);

            k_sgd_update<<<grid_up_w1, block1d>>>(d_w1, d_dw1, LEARNING_RATE, N_INPUT * N_HIDDEN);
            k_sgd_update<<<grid_up_b1, block1d>>>(d_b1, d_db1, LEARNING_RATE, N_HIDDEN);
            k_sgd_update<<<grid_up_w2, block1d>>>(d_w2, d_dw2, LEARNING_RATE, N_HIDDEN * N_CLASSES);
            k_sgd_update<<<grid_up_b2, block1d>>>(d_b2, d_db2, LEARNING_RATE, N_CLASSES);
            CUDA_CHECK(cudaEventRecord(ev_stop_bwd));

            CUDA_CHECK(cudaEventSynchronize(ev_stop_bwd));

            float iter_fwd_ms = 0.0f;
            float iter_bwd_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&iter_fwd_ms, ev_start_fwd, ev_stop_fwd));
            CUDA_CHECK(cudaEventElapsedTime(&iter_bwd_ms, ev_start_bwd, ev_stop_bwd));

            total_fwd_ms += iter_fwd_ms;
            total_bwd_ms += iter_bwd_ms;
        }
        CUDA_CHECK_LAST();

        float avg_fwd_ms = total_fwd_ms / BENCH_ITERS;
        float avg_bwd_ms = total_bwd_ms / BENCH_ITERS;
        float avg_total_ms = avg_fwd_ms + avg_bwd_ms;

        double throughput = (double)B / ((double)avg_total_ms / 1000.0);
        if (b_idx == 0) baseline_throughput = throughput;
        double speedup = throughput / baseline_throughput;

        double total_flops = FLOPS_PER_SAMPLE * (double)B + FLOPS_FIXED_STEP;
        double tflops = (total_flops / ((double)avg_total_ms / 1000.0)) / 1e12;

        printf(" %10d | %9.4f | %9.4f | %10.4f | %18.1f | %7.2fx | %7.3f\n",
               B, avg_fwd_ms, avg_bwd_ms, avg_total_ms, throughput, speedup, tflops);

        fprintf(fp, "%d,%.5f,%.5f,%.5f,%.2f,%.2f,%.4f\n",
                B, avg_fwd_ms, avg_bwd_ms, avg_total_ms, throughput, speedup, tflops);
    }

    printf("========================================================================================\n");
    printf("[OK] Benchmark data written to: %s\n\n", csv_path);
    fclose(fp);

    CUDA_CHECK(cudaEventDestroy(ev_start_fwd));
    CUDA_CHECK(cudaEventDestroy(ev_stop_fwd));
    CUDA_CHECK(cudaEventDestroy(ev_start_bwd));
    CUDA_CHECK(cudaEventDestroy(ev_stop_bwd));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_w1));
    CUDA_CHECK(cudaFree(d_b1));
    CUDA_CHECK(cudaFree(d_z1));
    CUDA_CHECK(cudaFree(d_a1));
    CUDA_CHECK(cudaFree(d_w2));
    CUDA_CHECK(cudaFree(d_b2));
    CUDA_CHECK(cudaFree(d_z2));
    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_dz2));
    CUDA_CHECK(cudaFree(d_dw2));
    CUDA_CHECK(cudaFree(d_db2));
    CUDA_CHECK(cudaFree(d_da1));
    CUDA_CHECK(cudaFree(d_dz1));
    CUDA_CHECK(cudaFree(d_dw1));
    CUDA_CHECK(cudaFree(d_db1));

    return 0;
}
