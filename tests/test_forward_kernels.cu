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
    printf("=== Unit Testing Forward CUDA Kernels & Init ===\n\n");

    const int B = 4;
    const int N_IN = 8;
    const int N_OUT = 4;

    {
        const int width = 512;
        const int height = 512;
        const int total = width * height;
        float *h_w = (float*)malloc(total * sizeof(float));
        float *d_w;
        CUDA_CHECK(cudaMalloc(&d_w, total * sizeof(float)));

        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        k_init_weights_he<<<grid, block>>>(d_w, width, height, 1337ULL);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_w, d_w, total * sizeof(float), cudaMemcpyDeviceToHost));

        double sum = 0.0;
        int nan_count = 0;
        for (int i = 0; i < total; i++) {
            if (isnan(h_w[i]) || isinf(h_w[i])) nan_count++;
            sum += h_w[i];
        }
        assert(nan_count == 0 && "He init produced NaN/Inf values!");

        double mean = sum / total;
        double sq_sum = 0.0;
        for (int i = 0; i < total; i++) {
            sq_sum += (h_w[i] - mean) * (h_w[i] - mean);
        }
        double std = sqrt(sq_sum / total);
        double expected_std = sqrt(2.0 / (double)height);

        assert(fabs(mean) < 0.01 && "He init mean deviated from 0!");
        assert(fabs(std - expected_std) < 0.005 && "He init standard deviation deviated!");

        printf("  [PASSED] k_init_weights_he (Mean=%.4f, Std=%.4f, Target=%.4f)\n",
               mean, std, expected_std);

        free(h_w);
        cudaFree(d_w);
    }

    {
        float h_x[B * N_IN], h_w[N_IN * N_OUT], h_b[N_OUT];
        float h_out_cpu[B * N_OUT], h_out_gpu[B * N_OUT];

        for (int i = 0; i < B * N_IN; i++) h_x[i] = ((float)rand() / RAND_MAX) - 0.5f;
        for (int i = 0; i < N_IN * N_OUT; i++) h_w[i] = ((float)rand() / RAND_MAX) - 0.5f;
        for (int i = 0; i < N_OUT; i++) h_b[i] = 0.1f * i;

        for (int r = 0; r < B; r++) {
            for (int c = 0; c < N_OUT; c++) {
                float sum = 0.0f;
                for (int k = 0; k < N_IN; k++) {
                    sum += h_x[r * N_IN + k] * h_w[k * N_OUT + c];
                }
                h_out_cpu[r * N_OUT + c] = sum + h_b[c];
            }
        }

        float *d_x, *d_w, *d_b, *d_out;
        CUDA_CHECK(cudaMalloc(&d_x, sizeof(h_x)));
        CUDA_CHECK(cudaMalloc(&d_w, sizeof(h_w)));
        CUDA_CHECK(cudaMalloc(&d_b, sizeof(h_b)));
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(h_out_gpu)));

        CUDA_CHECK(cudaMemcpy(d_x, h_x, sizeof(h_x), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_w, h_w, sizeof(h_w), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b, sizeof(h_b), cudaMemcpyHostToDevice));

        dim3 block(16, 16);
        dim3 grid((N_OUT + block.x - 1) / block.x, (B + block.y - 1) / block.y);
        k_forward_linear<<<grid, block>>>(d_x, d_w, d_b, d_out, B, N_IN, N_OUT);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_out_gpu, d_out, sizeof(h_out_gpu), cudaMemcpyDeviceToHost));
        assert_allclose(h_out_cpu, h_out_gpu, B * N_OUT, "k_forward_linear");

        cudaFree(d_x); cudaFree(d_w); cudaFree(d_b); cudaFree(d_out);
    }

    {
        const int N = 16;
        float h_z[N] = {-2.0f, -0.5f, 0.0f, 0.5f, 1.5f, -3.0f, 4.0f, 0.0f,
                         1.0f, -1.0f, 2.0f, -2.0f, 3.0f, -3.0f, 0.5f, -0.5f};
        float h_a_cpu[N], h_a_gpu[N];
        for (int i = 0; i < N; i++) h_a_cpu[i] = h_z[i] > 0.0f ? h_z[i] : 0.0f;

        float *d_z, *d_a;
        CUDA_CHECK(cudaMalloc(&d_z, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_z, h_z, N * sizeof(float), cudaMemcpyHostToDevice));

        k_relu<<<(N + 255) / 256, 256>>>(d_z, d_a, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_a_gpu, d_a, N * sizeof(float), cudaMemcpyDeviceToHost));
        assert_allclose(h_a_cpu, h_a_gpu, N, "k_relu");

        cudaFree(d_z); cudaFree(d_a);
    }

    {
        float h_z[B * N_OUT], h_p_cpu[B * N_OUT], h_p_gpu[B * N_OUT];
        for (int i = 0; i < B * N_OUT; i++) h_z[i] = ((float)rand() / RAND_MAX) * 2.0f;

        for (int r = 0; r < B; r++) {
            float max_val = h_z[r * N_OUT];
            for (int c = 1; c < N_OUT; c++) {
                if (h_z[r * N_OUT + c] > max_val) max_val = h_z[r * N_OUT + c];
            }
            float sum = 0.0f;
            for (int c = 0; c < N_OUT; c++) sum += expf(h_z[r * N_OUT + c] - max_val);
            for (int c = 0; c < N_OUT; c++) {
                h_p_cpu[r * N_OUT + c] = expf(h_z[r * N_OUT + c] - max_val) / sum;
            }
        }

        float *d_z, *d_p;
        CUDA_CHECK(cudaMalloc(&d_z, sizeof(h_z)));
        CUDA_CHECK(cudaMalloc(&d_p, sizeof(h_p_gpu)));
        CUDA_CHECK(cudaMemcpy(d_z, h_z, sizeof(h_z), cudaMemcpyHostToDevice));

        dim3 block(16, 16);
        dim3 grid((N_OUT + block.x - 1) / block.x, (B + block.y - 1) / block.y);
        k_softmax<<<grid, block>>>(d_z, d_p, B, N_OUT);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_p_gpu, d_p, sizeof(h_p_gpu), cudaMemcpyDeviceToHost));
        assert_allclose(h_p_cpu, h_p_gpu, B * N_OUT, "k_softmax");

        cudaFree(d_z); cudaFree(d_p);
    }

    {
        float h_p[B * N_OUT], h_y[B * N_OUT] = {0};
        float h_loss_cpu[B], h_loss_gpu[B];

        for (int r = 0; r < B; r++) {
            h_y[r * N_OUT + (r % N_OUT)] = 1.0f;
            float sum = 0.0f;
            for (int c = 0; c < N_OUT; c++) {
                h_p[r * N_OUT + c] = (float)rand() / RAND_MAX + 0.1f;
                sum += h_p[r * N_OUT + c];
            }
            for (int c = 0; c < N_OUT; c++) h_p[r * N_OUT + c] /= sum;
            h_loss_cpu[r] = -logf(fmaxf(h_p[r * N_OUT + (r % N_OUT)], 1e-7f));
        }

        float *d_p, *d_y, *d_loss;
        CUDA_CHECK(cudaMalloc(&d_p, sizeof(h_p)));
        CUDA_CHECK(cudaMalloc(&d_y, sizeof(h_y)));
        CUDA_CHECK(cudaMalloc(&d_loss, B * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_p, h_p, sizeof(h_p), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y, h_y, sizeof(h_y), cudaMemcpyHostToDevice));

        k_cross_entropy_loss<<<(B + 255) / 256, 256>>>(d_p, d_y, d_loss, B, N_OUT);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_loss_gpu, d_loss, B * sizeof(float), cudaMemcpyDeviceToHost));
        assert_allclose(h_loss_cpu, h_loss_gpu, B, "k_cross_entropy_loss");

        cudaFree(d_p); cudaFree(d_y); cudaFree(d_loss);
    }

    printf("\nAll forward kernel unit tests passed.\n");
    return 0;
}
