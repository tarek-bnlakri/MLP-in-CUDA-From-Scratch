#ifndef LAYERS_CUH
#define LAYERS_CUH

#include <cuda_runtime.h>

__global__ void k_init_weights_he(float* mat, int width, int height, unsigned long long seed);

__global__ void k_forward_linear(const float* input, const float* weight, const float* bias,
                                 float* output, int batch_size, int n_in, int n_out);
__global__ void k_relu(const float* z, float* a, int total_elements);
__global__ void k_softmax(const float* z, float* p, int batch_size, int num_classes);
__global__ void k_cross_entropy_loss(const float* p, const float* y, float* loss, int batch_size, int num_classes);

__global__ void k_cross_entropy_backward(const float* p, const float* y, float* dz, int batch_size, int num_classes);
__global__ void k_compute_dw(const float* a, const float* dz, float* dw, int batch_size, int n_in, int n_out);
__global__ void k_compute_db(const float* dz, float* db, int batch_size, int num_classes);
__global__ void k_compute_da(const float* dz, const float* w, float* da, int batch_size, int n_in, int n_out);
__global__ void k_relu_backward(const float* z, const float* da, float* dz, int total_elements);

__global__ void k_sgd_update(float* param, const float* grad, float lr, int total_elements);

__global__ void k_gather_batch(const float* src, float* dst, const int* indices, int batch_size, int feature_dim);
__global__ void k_count_correct(const float* p, const float* y, int* d_correct_count, int batch_size, int num_classes);
__global__ void k_accumulate_loss(const float* p, const float* y, float* d_total_loss, int batch_size, int num_classes);

#endif
