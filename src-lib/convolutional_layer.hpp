#pragma once

#include "darknet_internal.hpp"

#ifdef DARKNET_GPU
void forward_convolutional_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
void backward_convolutional_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
void update_convolutional_layer_gpu(Darknet::Layer & l, int batch, float learning_rate, float momentum, float decay, float loss_scale);

void push_convolutional_layer(Darknet::Layer & l);
void pull_convolutional_layer(Darknet::Layer & l);

void add_bias_gpu(float *output, float *biases, int batch, int n, int size);
void backward_bias_gpu(float *bias_updates, float *delta, int batch, int n, int size);
#ifdef CUDNN
void cudnn_convolutional_setup(Darknet::Layer *l, int cudnn_preference, size_t workspace_size_specify);
void create_convolutional_cudnn_tensors(Darknet::Layer *l);
void cuda_convert_f32_to_f16(float* input_f32, size_t size, float *output_f16);
void cuda_convert_f16_to_f32(float* input_f16, size_t size, float *output_f32);
#if defined(DARKNET_GPU_CUDA) && (CUDNN_MAJOR >= 8) && (CUDART_VERSION >= 11000)
__nv_bfloat16 *cuda_make_bf16_from_f32_array(float *src, size_t n);
void cuda_convert_f32_to_bf16(float* input_f32, size_t size, float *output_bf16);
void cuda_convert_bf16_to_f32(float* input_bf16, size_t size, float *output_f32);
void cuda_round_f32_to_bf16_inplace(float * data, size_t n);
void cuda_bf16_kahan_sgd(int n, float lr, float decay_factor, float momentum_factor, float * weight_updates, float * weights_bf16, float * compensation_bf16, float * weights_f32_mirror);
void cuda_bf16_kahan_commit(int n, const float * weights_f32, float * weights_bf16, float * compensation_bf16);
#endif
void* cuda_make_lowp_from_f32_array(float* src, size_t n, bool is_bf16);
void cuda_convert_f32_to_lowp(float* src, size_t n, void* dst, bool is_bf16);
void cuda_convert_lowp_to_f32(void* src, size_t n, float* dst, bool is_bf16);
#if defined(DARKNET_GPU_CUDA) && (CUDNN_MAJOR >= 9) && (CUDART_VERSION >= 12000)
void cuda_convert_f32_to_fp8(float* input_f32, size_t size, float *output_fp8);
void cuda_convert_fp8_to_f32(float* input_fp8, size_t size, float *output_f32);
void cuda_convert_f32_to_bf16_via_fp8(float* input_f32, size_t size, float *output_bf16);
void cuda_quantize_f32_to_fp8_and_dequantize_bf16(float *input_f32, size_t size, uint8_t *output_fp8, float *scale_gpu, float *amax_ema_gpu, int fp8_format, int update_scale, float *output_bf16);
void cuda_quantize_bf16_to_fp8_and_dequantize_bf16(float *input_bf16, size_t size, uint8_t *output_fp8, float *scale_gpu, float *amax_ema_gpu, int fp8_format, int update_scale, float *output_bf16);
void cuda_fp8_current_scale_quantize_bf16(float *input_f32, size_t size, uint8_t *output_fp8, float *amax_gpu, float *amax_ema_gpu, int fp8_format, float *output_bf16);
void cuda_quantize_f32_to_fp8_bf16_by_policy(float *input_f32, size_t size, uint8_t *output_fp8, float *scale_gpu, float *amax_ema_gpu, int fp8_format, int use_current_scaling, int update_scale, float *output_bf16);
void cuda_quantize_bf16_to_fp8_bf16_by_policy(float *input_bf16, size_t size, uint8_t *output_fp8, float *scale_gpu, float *amax_ema_gpu, int fp8_format, int use_current_scaling, int update_scale, float *output_bf16);
#endif
#endif
#endif
void free_convolutional_batchnorm(Darknet::Layer *l);

size_t get_convolutional_workspace_size(const Darknet::Layer & l);
Darknet::Layer make_convolutional_layer(int batch, int steps, int h, int w, int c, int n, int groups, int size, int stride_x, int stride_y, int dilation, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam, int use_bin_output, int index, int antialiasing, Darknet::Layer * share_layer, int assisted_excitation, int deform, int train);
void denormalize_convolutional_layer(Darknet::Layer & l);
void set_specified_workspace_limit(Darknet::Layer *l, size_t workspace_size_limit);
void resize_convolutional_layer(Darknet::Layer * l, int w, int h);
void forward_convolutional_layer(Darknet::Layer & l, Darknet::NetworkState state);
void update_convolutional_layer(Darknet::Layer & l, int batch, float learning_rate, float momentum, float decay);
Darknet::Image *visualize_convolutional_layer(const Darknet::Layer & l, const char * window, Darknet::Image * prev_weights);
void binarize_weights(float *weights, int n, int size, float *binary);
void swap_binary(Darknet::Layer *l);
void binarize_weights2(float *weights, int n, int size, char *binary, float *scales);

void binary_align_weights(Darknet::Layer *l);

void backward_convolutional_layer(Darknet::Layer & l, Darknet::NetworkState state);

void add_bias(float *output, float *biases, int batch, int n, int size);
void backward_bias(float *bias_updates, float *delta, int batch, int n, int size);

Darknet::Image get_convolutional_image(const Darknet::Layer & l);
Darknet::Image get_convolutional_delta(const Darknet::Layer & l);
Darknet::Image get_convolutional_weight(const Darknet::Layer & l, int i);

int convolutional_out_height(const Darknet::Layer & l);
int convolutional_out_width(const Darknet::Layer & l);
void rescale_weights(Darknet::Layer & l, float scale, float trans);
void rgbgr_weights(const Darknet::Layer & l);
void assisted_excitation_forward(Darknet::Layer & l, Darknet::NetworkState state);
void assisted_excitation_forward_gpu(Darknet::Layer & l, Darknet::NetworkState state);
