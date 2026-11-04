#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda_runtime.h>
#include "layers.cuh"

#define EPSILON 1e-4f

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

static void assert_allclose(const float* a, const float* b, int n, const char* name) {
    for (int i = 0; i < n; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > EPSILON) {
            fprintf(stderr, "FAIL: %s at index %d: CPU=%f, GPU=%f (diff=%f > %f)\n",
                    name, i, a[i], b[i], diff, EPSILON);
            exit(EXIT_FAILURE);
        }
    }
    printf("  [PASSED] %s\n", name);
}

int main() {
    printf("=== Unit Testing Backward CUDA Kernels & Optimizer ===\n\n");

    const int B = 4;
    const int N_IN = 8;
    const int N_OUT = 4;

    {
        float h_p[B * N_OUT], h_y[B * N_OUT] = {0};
        float h_dz_cpu[B * N_OUT], h_dz_gpu[B * N_OUT];

        for (int i = 0; i < B * N_OUT; i++) h_p[i] = (float)rand() / RAND_MAX;
        for (int r = 0; r < B; r++) {
            h_y[r * N_OUT + (r % N_OUT)] = 1.0f;
            for (int c = 0; c < N_OUT; c++) {
                int idx = r * N_OUT + c;
                h_dz_cpu[idx] = h_p[idx] - h_y[idx];
            }
        }

        float *d_p, *d_y, *d_dz;
        CUDA_CHECK(cudaMalloc(&d_p, sizeof(h_p)));
        CUDA_CHECK(cudaMalloc(&d_y, sizeof(h_y)));
        CUDA_CHECK(cudaMalloc(&d_dz, sizeof(h_dz_gpu)));

        CUDA_CHECK(cudaMemcpy(d_p, h_p, sizeof(h_p), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y, h_y, sizeof(h_y), cudaMemcpyHostToDevice));

        dim3 block(16, 16);
        dim3 grid((N_OUT + block.x - 1) / block.x, (B + block.y - 1) / block.y);
        k_cross_entropy_backward<<<grid, block>>>(d_p, d_y, d_dz, B, N_OUT);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_dz_gpu, d_dz, sizeof(h_dz_gpu), cudaMemcpyDeviceToHost));
        assert_allclose(h_dz_cpu, h_dz_gpu, B * N_OUT, "k_cross_entropy_backward");

        cudaFree(d_p); cudaFree(d_y); cudaFree(d_dz);
    }

    {
        const int N = 16;
        float h_z[N] = {-2.0f, -0.5f, 0.0f, 0.5f, 1.5f, -3.0f, 4.0f, 0.0f,
                         1.0f, -1.0f, 2.0f, -2.0f, 3.0f, -3.0f, 0.5f, -0.5f};
        float h_da[N], h_dz_cpu[N], h_dz_gpu[N];
        for (int i = 0; i < N; i++) h_da[i] = ((float)rand() / RAND_MAX) - 0.5f;
        for (int i = 0; i < N; i++) h_dz_cpu[i] = h_z[i] > 0.0f ? h_da[i] : 0.0f;

        float *d_z, *d_da, *d_dz;
        CUDA_CHECK(cudaMalloc(&d_z, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_da, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dz, N * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_z, h_z, N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_da, h_da, N * sizeof(float), cudaMemcpyHostToDevice));

        k_relu_backward<<<(N + 255) / 256, 256>>>(d_z, d_da, d_dz, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_dz_gpu, d_dz, N * sizeof(float), cudaMemcpyDeviceToHost));
        assert_allclose(h_dz_cpu, h_dz_gpu, N, "k_relu_backward");

        cudaFree(d_z); cudaFree(d_da); cudaFree(d_dz);
    }

    {
        float h_a[B * N_IN], h_dz[B * N_OUT];
        float h_dw_cpu[N_IN * N_OUT], h_dw_gpu[N_IN * N_OUT];

        for (int i = 0; i < B * N_IN; i++) h_a[i] = ((float)rand() / RAND_MAX) - 0.5f;
        for (int i = 0; i < B * N_OUT; i++) h_dz[i] = ((float)rand() / RAND_MAX) - 0.5f;

        for (int r = 0; r < N_IN; r++) {
            for (int c = 0; c < N_OUT; c++) {
                float sum = 0.0f;
                for (int k = 0; k < B; k++) {
                    sum += h_a[k * N_IN + r] * h_dz[k * N_OUT + c];
                }
                h_dw_cpu[r * N_OUT + c] = sum / (float)B;
            }
        }

        float *d_a, *d_dz, *d_dw;
        CUDA_CHECK(cudaMalloc(&d_a, sizeof(h_a)));
        CUDA_CHECK(cudaMalloc(&d_dz, sizeof(h_dz)));
        CUDA_CHECK(cudaMalloc(&d_dw, sizeof(h_dw_gpu)));

        CUDA_CHECK(cudaMemcpy(d_a, h_a, sizeof(h_a), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_dz, h_dz, sizeof(h_dz), cudaMemcpyHostToDevice));

        dim3 block(16, 16);
        dim3 grid((N_OUT + block.x - 1) / block.x, (N_IN + block.y - 1) / block.y);
        k_compute_dw<<<grid, block>>>(d_a, d_dz, d_dw, B, N_IN, N_OUT);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_dw_gpu, d_dw, sizeof(h_dw_gpu), cudaMemcpyDeviceToHost));
        assert_allclose(h_dw_cpu, h_dw_gpu, N_IN * N_OUT, "k_compute_dw");

        cudaFree(d_a); cudaFree(d_dz); cudaFree(d_dw);
    }

    {
        float h_dz[B * N_OUT];
        float h_db_cpu[N_OUT], h_db_gpu[N_OUT];

        for (int i = 0; i < B * N_OUT; i++) h_dz[i] = ((float)rand() / RAND_MAX) - 0.5f;
        for (int c = 0; c < N_OUT; c++) {
            float sum = 0.0f;
            for (int k = 0; k < B; k++) sum += h_dz[k * N_OUT + c];
            h_db_cpu[c] = sum / (float)B;
        }

        float *d_dz, *d_db;
        CUDA_CHECK(cudaMalloc(&d_dz, sizeof(h_dz)));
        CUDA_CHECK(cudaMalloc(&d_db, sizeof(h_db_gpu)));
        CUDA_CHECK(cudaMemcpy(d_dz, h_dz, sizeof(h_dz), cudaMemcpyHostToDevice));

        k_compute_db<<<(N_OUT + 255) / 256, 256>>>(d_dz, d_db, B, N_OUT);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_db_gpu, d_db, sizeof(h_db_gpu), cudaMemcpyDeviceToHost));
        assert_allclose(h_db_cpu, h_db_gpu, N_OUT, "k_compute_db");

        cudaFree(d_dz); cudaFree(d_db);
    }

    {
        float h_dz[B * N_OUT], h_w[N_IN * N_OUT];
        float h_da_cpu[B * N_IN], h_da_gpu[B * N_IN];

        for (int i = 0; i < B * N_OUT; i++) h_dz[i] = ((float)rand() / RAND_MAX) - 0.5f;
        for (int i = 0; i < N_IN * N_OUT; i++) h_w[i] = ((float)rand() / RAND_MAX) - 0.5f;

        for (int r = 0; r < B; r++) {
            for (int c = 0; c < N_IN; c++) {
                float sum = 0.0f;
                for (int k = 0; k < N_OUT; k++) {
                    sum += h_dz[r * N_OUT + k] * h_w[c * N_OUT + k];
                }
                h_da_cpu[r * N_IN + c] = sum;
            }
        }

        float *d_dz, *d_w, *d_da;
        CUDA_CHECK(cudaMalloc(&d_dz, sizeof(h_dz)));
        CUDA_CHECK(cudaMalloc(&d_w, sizeof(h_w)));
        CUDA_CHECK(cudaMalloc(&d_da, sizeof(h_da_gpu)));

        CUDA_CHECK(cudaMemcpy(d_dz, h_dz, sizeof(h_dz), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_w, h_w, sizeof(h_w), cudaMemcpyHostToDevice));

        dim3 block(16, 16);
        dim3 grid((N_IN + block.x - 1) / block.x, (B + block.y - 1) / block.y);
        k_compute_da<<<grid, block>>>(d_dz, d_w, d_da, B, N_IN, N_OUT);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_da_gpu, d_da, sizeof(h_da_gpu), cudaMemcpyDeviceToHost));
        assert_allclose(h_da_cpu, h_da_gpu, B * N_IN, "k_compute_da");

        cudaFree(d_dz); cudaFree(d_w); cudaFree(d_da);
    }

    {
        const int N = 8;
        float h_w[N]  = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        float h_dw[N] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
        float lr = 0.05f;
        float h_w_cpu[N], h_w_gpu[N];

        for (int i = 0; i < N; i++) h_w_cpu[i] = h_w[i] - lr * h_dw[i];

        float *d_w, *d_dw;
        CUDA_CHECK(cudaMalloc(&d_w, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dw, N * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_w, h_w, N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_dw, h_dw, N * sizeof(float), cudaMemcpyHostToDevice));

        k_sgd_update<<<(N + 255) / 256, 256>>>(d_w, d_dw, lr, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_w_gpu, d_w, N * sizeof(float), cudaMemcpyDeviceToHost));
        assert_allclose(h_w_cpu, h_w_gpu, N, "k_sgd_update");

        cudaFree(d_w); cudaFree(d_dw);
    }

    printf("\nAll backward kernel unit tests passed.\n");
    return 0;
}
