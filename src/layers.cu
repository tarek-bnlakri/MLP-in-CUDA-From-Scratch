#include <math.h>
#include <curand_kernel.h>
#include "layers.cuh"

__global__ void k_init_weights_he(float* mat, int width, int height, unsigned long long seed) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width) {
        curandState state;
        int unique_id = row * width + col;
        curand_init(seed, unique_id, 0, &state);
        float scale = sqrtf(2.0f / (float)height);
        mat[unique_id] = scale * curand_normal(&state);
    }
}

__global__ void k_forward_linear(const float* input, const float* weight, const float* bias,
                                 float* output, int batch_size, int n_in, int n_out) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < batch_size && col < n_out) {
        float psum = 0.0f;
        for (int k = 0; k < n_in; k++) {
            psum += input[row * n_in + k] * weight[k * n_out + col];
        }
        output[row * n_out + col] = psum + bias[col];
    }
}

__global__ void k_relu(const float* z, float* a, int total_elements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < total_elements) {
        a[i] = fmaxf(0.0f, z[i]);
    }
}

__global__ void k_softmax(const float* z, float* p, int batch_size, int num_classes) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < batch_size && col < num_classes) {
        float max_val = z[row * num_classes];
        for (int k = 1; k < num_classes; k++) {
            max_val = fmaxf(max_val, z[row * num_classes + k]);
        }

        float sum_exp = 0.0f;
        for (int k = 0; k < num_classes; k++) {
            sum_exp += expf(z[row * num_classes + k] - max_val);
        }

        p[row * num_classes + col] = expf(z[row * num_classes + col] - max_val) / sum_exp;
    }
}

__global__ void k_cross_entropy_loss(const float* p, const float* y, float* loss, int batch_size, int num_classes) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < batch_size) {
        float sum = 0.0f;
        for (int k = 0; k < num_classes; k++) {
            if (y[row * num_classes + k] > 0.5f) {
                sum -= logf(fmaxf(p[row * num_classes + k], 1e-7f));
            }
        }
        loss[row] = sum;
    }
}

__global__ void k_cross_entropy_backward(const float* p, const float* y, float* dz, int batch_size, int num_classes) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < batch_size && col < num_classes) {
        int idx = row * num_classes + col;
        dz[idx] = p[idx] - y[idx];
    }
}

__global__ void k_compute_dw(const float* a, const float* dz, float* dw, int batch_size, int n_in, int n_out) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n_in && col < n_out) {
        float sum = 0.0f;
        for (int k = 0; k < batch_size; k++) {
            sum += a[k * n_in + row] * dz[k * n_out + col];
        }
        dw[row * n_out + col] = sum / (float)batch_size;
    }
}

__global__ void k_compute_db(const float* dz, float* db, int batch_size, int num_classes) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < num_classes) {
        float sum = 0.0f;
        for (int k = 0; k < batch_size; k++) {
            sum += dz[k * num_classes + col];
        }
        db[col] = sum / (float)batch_size;
    }
}

__global__ void k_compute_da(const float* dz, const float* w, float* da, int batch_size, int n_in, int n_out) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < n_in) {
        float sum = 0.0f;
        for (int k = 0; k < n_out; k++) {
            sum += dz[row * n_out + k] * w[col * n_out + k];
        }
        da[row * n_in + col] = sum;
    }
}

__global__ void k_relu_backward(const float* z, const float* da, float* dz, int total_elements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_elements) {
        dz[i] = (z[i] > 0.0f) ? da[i] : 0.0f;
    }
}

__global__ void k_sgd_update(float* param, const float* grad, float lr, int total_elements) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < total_elements) {
        param[idx] -= lr * grad[idx];
    }
}

__global__ void k_gather_batch(const float* src, float* dst, const int* indices, int batch_size, int feature_dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < feature_dim) {
        dst[row * feature_dim + col] = src[indices[row] * feature_dim + col];
    }
}

__global__ void k_count_correct(const float* p, const float* y, int* d_correct_count, int batch_size, int num_classes) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < batch_size) {
        int pred_class = 0;
        float max_p = p[i * num_classes];
        for (int c = 1; c < num_classes; c++) {
            float val = p[i * num_classes + c];
            if (val > max_p) {
                max_p = val;
                pred_class = c;
            }
        }

        int true_class = 0;
        for (int c = 0; c < num_classes; c++) {
            if (y[i * num_classes + c] > 0.5f) {
                true_class = c;
                break;
            }
        }

        if (pred_class == true_class) {
            atomicAdd(d_correct_count, 1);
        }
    }
}

__global__ void k_accumulate_loss(const float* p, const float* y, float* d_total_loss, int batch_size, int num_classes) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < batch_size) {
        float sample_loss = 0.0f;
        for (int c = 0; c < num_classes; c++) {
            if (y[i * num_classes + c] > 0.5f) {
                sample_loss = -logf(fmaxf(p[i * num_classes + c], 1e-7f));
                break;
            }
        }
        atomicAdd(d_total_loss, sample_loss);
    }
}
