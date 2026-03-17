#include "darknet_internal.hpp"
#include "gemm.hpp"
#include "col2im.hpp"
#include "im2col.hpp"
#ifdef DARKNET_GPU_CUDA
#include <cuda_bf16.h>
#if CUDART_VERSION >= 12000
#include <cuda_fp8.h>
#endif
#endif


namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();

#if defined(DARKNET_GPU_CUDA) && (CUDART_VERSION >= 12000)
	constexpr float k_fp8_e4m3_max = 448.0f;
	constexpr float k_fp8_e5m2_max = 57344.0f;
	constexpr float k_fp8_ema_decay = 0.95f;
	constexpr float k_fp8_min_scale = 1.0e-8f;
#endif
}


__global__ void binarize_kernel(float *x, int n, float *binary)
{
	int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (i >= n) return;
	binary[i] = (x[i] >= 0) ? 1 : -1;
}

void binarize_gpu(float *x, int n, float *binary)
{
	TAT(TATPARMS);

	binarize_kernel<<<cuda_gridsize(n), BLOCK, 0, get_cuda_stream() >>>(x, n, binary);
	CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void binarize_input_kernel(float *input, int n, int size, float *binary)
{
	int s = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (s >= size) return;
	int i = 0;
	float mean = 0;
	for(i = 0; i < n; ++i){
		mean += fabs(input[i*size + s]);
	}
	mean = mean / n;
	for(i = 0; i < n; ++i){
		binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
	}
}

void binarize_input_gpu(float *input, int n, int size, float *binary)
{
	TAT(TATPARMS);

	binarize_input_kernel<<<cuda_gridsize(size), BLOCK, 0, get_cuda_stream() >>>(input, n, size, binary);
	CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void binarize_weights_kernel(float *weights, int n, int size, float *binary)
{
	int f = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (f >= n) return;
	int i = 0;
	float mean = 0;
	for (i = 0; i < size; ++i)
	{
		mean += fabs(weights[f*size + i]);
	}
	mean = mean / size;
	for (i = 0; i < size; ++i)
	{
		binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
	}
}

void binarize_weights_gpu(float *weights, int n, int size, float *binary)
{
	TAT(TATPARMS);

	binarize_weights_kernel <<<cuda_gridsize(n), BLOCK, 0, get_cuda_stream() >>>(weights, n, size, binary);
	CHECK_CUDA(cudaPeekAtLastError());
}


__global__ void set_zero_kernel(float *src, int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) src[i] = 0;
}

__inline__ __device__ float warpAllReduceSum(float val)
{
	for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2)
#if CUDART_VERSION >= 9000
		val += __shfl_xor_sync(0xffffffff, val, mask);
#else
		val += __shfl_xor(val, mask);
#endif
	return val;
}

// only if (size % 32 == 0)
__global__ void reduce_kernel(float *weights, int n, int size, float *mean_arr_gpu)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int f = i / size;
	if (f >= n) return;
	float warp_mean = warpAllReduceSum(fabs(weights[i]));
	if (i % 32 == 0)
	{
		atomicAdd(&mean_arr_gpu[f], warp_mean / size);
	}
}

__global__ void binarize_weights_mean_kernel(float *weights, int n, int size, float *binary, float *mean_arr_gpu)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int f = i / size;
	if (f >= n) return;
	float mean = mean_arr_gpu[f];
	binary[i] = (weights[i] > 0) ? mean : -mean;
}

void fast_binarize_weights_gpu(float *weights, int n, int size, float *binary, float *mean_arr_gpu)
{
	TAT(TATPARMS);

	if (size % 32 == 0) {
		size_t gridsize = n * size;
		const int num_blocks = get_number_of_blocks(gridsize, BLOCK);// gridsize / BLOCK + 1;

		set_zero_kernel <<<(n/BLOCK + 1), BLOCK, 0, get_cuda_stream() >>> (mean_arr_gpu, n);
		reduce_kernel <<<num_blocks, BLOCK, 0, get_cuda_stream() >>> (weights, n, size, mean_arr_gpu);
		binarize_weights_mean_kernel <<<num_blocks, BLOCK, 0, get_cuda_stream() >>> (weights, n, size, binary, mean_arr_gpu);
		CHECK_CUDA(cudaPeekAtLastError());
	}
	else {
		binarize_weights_gpu(weights, n, size, binary);
	}
}


__global__ void cuda_f32_to_f16(float* input_f32, size_t size, half *output_f16)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) output_f16[idx] = __float2half(input_f32[idx]);
}

void cuda_convert_f32_to_f16(float* input_f32, size_t size, float *output_f16)
{
	TAT(TATPARMS);

	cuda_f32_to_f16 <<< get_number_of_blocks(size, BLOCK), BLOCK, 0, get_cuda_stream() >>> (input_f32, size, (half *)output_f16);
	CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void cuda_f16_to_f32(half* input_f16, size_t size, float *output_f32)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) output_f32[idx] = __half2float(input_f16[idx]);
}

void cuda_convert_f16_to_f32(float* input_f16, size_t size, float *output_f32)
{
	TAT(TATPARMS);

	cuda_f16_to_f32 <<< get_number_of_blocks(size, BLOCK), BLOCK, 0, get_cuda_stream() >>> ((half *)input_f16, size, output_f32);
	CHECK_CUDA(cudaPeekAtLastError());
}

half *cuda_make_f16_from_f32_array(float *src, size_t n)
{
	TAT(TATPARMS);

	half *dst16;
	size_t size = sizeof(half)*n;
	CHECK_CUDA(cudaMalloc((void **)&dst16, size));
	if (src) {
		assert(n > 0);
		cuda_convert_f32_to_f16(src, n, (float *)dst16);
	}
	if (!dst16)
	{
		darknet_fatal_error(DARKNET_LOC, "CUDA malloc failed (n=%d)", n);
	}
	return dst16;
}

#if defined(DARKNET_GPU_CUDA) && (CUDART_VERSION >= 11000)
__global__ void cuda_f32_to_bf16(float* input_f32, size_t size, __nv_bfloat16 *output_bf16)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) output_bf16[idx] = __float2bfloat16(input_f32[idx]);
}

void cuda_convert_f32_to_bf16(float* input_f32, size_t size, float *output_bf16)
{
	TAT(TATPARMS);

	cuda_f32_to_bf16 <<< get_number_of_blocks(size, BLOCK), BLOCK, 0, get_cuda_stream() >>> (input_f32, size, (__nv_bfloat16 *)output_bf16);
	CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void cuda_bf16_to_f32(__nv_bfloat16* input_bf16, size_t size, float *output_f32)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) output_f32[idx] = __bfloat162float(input_bf16[idx]);
}

void cuda_convert_bf16_to_f32(float* input_bf16, size_t size, float *output_f32)
{
	TAT(TATPARMS);

	cuda_bf16_to_f32 <<< get_number_of_blocks(size, BLOCK), BLOCK, 0, get_cuda_stream() >>> ((__nv_bfloat16 *)input_bf16, size, output_f32);
	CHECK_CUDA(cudaPeekAtLastError());
}

/// Rounds each FP32 value to BF16 precision in-place (stores result back as FP32).
/// After this call every element has at most 7 mantissa bits — identical semantics to
/// storing in BF16 but the buffer stays float* so all existing kernels continue to work.
__global__ void cuda_round_to_bf16_kernel(float * __restrict__ data, size_t n)
{
	const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
	{
		data[i] = __bfloat162float(__float2bfloat16(data[i]));
	}
}

void cuda_round_f32_to_bf16_inplace(float * data, size_t n)
{
	TAT(TATPARMS);

	cuda_round_to_bf16_kernel <<< get_number_of_blocks(n, BLOCK), BLOCK, 0, get_cuda_stream() >>> (data, n);
	CHECK_CUDA(cudaPeekAtLastError());
}

__nv_bfloat16 *cuda_make_bf16_from_f32_array(float *src, size_t n)
{
	TAT(TATPARMS);

	__nv_bfloat16 *dst16;
	size_t size = sizeof(__nv_bfloat16) * n;
	CHECK_CUDA(cudaMalloc((void **)&dst16, size));
	if (src)
	{
		assert(n > 0);
		cuda_convert_f32_to_bf16(src, n, (float *)dst16);
	}
	if (!dst16)
	{
		darknet_fatal_error(DARKNET_LOC, "CUDA malloc failed (n=%d)", n);
	}
	return dst16;
}

void* cuda_make_lowp_from_f32_array(float* src, size_t n, bool is_bf16)
{
#if defined(DARKNET_GPU_CUDA) && (CUDART_VERSION >= 11000)
	if (is_bf16) return cuda_make_bf16_from_f32_array(src, n);
#endif
	return cuda_make_f16_from_f32_array(src, n);
}

void cuda_convert_f32_to_lowp(float* src, size_t n, void* dst, bool is_bf16)
{
#if defined(DARKNET_GPU_CUDA) && (CUDART_VERSION >= 11000)
	if (is_bf16)
	{
		cuda_convert_f32_to_bf16(src, n, (float*)dst);
		return;
	}
#endif
	cuda_convert_f32_to_f16(src, n, (float*)dst);
}

void cuda_convert_lowp_to_f32(void* src, size_t n, float* dst, bool is_bf16)
{
#if defined(DARKNET_GPU_CUDA) && (CUDART_VERSION >= 11000)
	if (is_bf16)
	{
		cuda_convert_bf16_to_f32((float*)src, n, dst);
		return;
	}
#endif
	cuda_convert_f16_to_f32((float*)src, n, dst);
}

bool use_bf16_master_weight_storage(const Darknet::Layer & l)
{
	return (cfg_and_state.use_bf16_master_weights && l.weights_gpu16);
}

static inline bool layer_can_use_lowp_conv(const Darknet::Network & net, const Darknet::Layer & l, const bool train)
{
	if (l.type != Darknet::ELayerType::CONVOLUTIONAL || l.xnor)
	{
		return false;
	}

	const int iteration_num = get_current_iteration(net);
	const int tensor_cores_min_iteration = std::max(0, net.tensor_cores_min_iteration);
	const bool bf16_compute_mode = (
		net.precision_mode == Darknet::PrecisionMode::BF16 ||
		net.precision_mode == Darknet::PrecisionMode::BF16_MASTER_KAHAN ||
		net.precision_mode == Darknet::PrecisionMode::FP8_BF16 ||
		net.cudnn_bfloat16 || net.cudnn_fp8);
	const bool allow_bf16_tensor_core_now = (!train || net.loss_scale > 1.0f || iteration_num >= tensor_cores_min_iteration);
	const bool tc_aligned = ((l.c / std::max(1, l.groups)) % 8 == 0 && l.n % 8 == 0 && l.groups <= 1);

	return (bf16_compute_mode && allow_bf16_tensor_core_now && tc_aligned);
}

static inline bool next_layer_can_consume_cached_bf16_output(const Darknet::Network & net, const Darknet::Layer & l, const int state_index, const bool train)
{
	if (l.output_gpu16 == nullptr || state_index < 0 || state_index + 1 >= net.n)
	{
		return false;
	}

	if (&net.layers[state_index] != &l)
	{
		return false;
	}

	if (!layer_can_use_lowp_conv(net, l, train))
	{
		return false;
	}

	const Darknet::Layer & next = net.layers[state_index + 1];
	return (layer_can_use_lowp_conv(net, next, train) && !(net.cudnn_fp8 && next.fp8_enabled > 0));
}

static inline bool can_materialize_post_activation_bf16_cache(const Darknet::Network & net, const Darknet::Layer & l, const bool cache_bf16_output)
{
#if defined(DARKNET_GPU_CUDA) && (CUDART_VERSION >= 11000)
	return (
		cache_bf16_output &&
		l.output_gpu16 != nullptr &&
		!net.try_fix_nan &&
		!l.assisted_excitation &&
		!l.antialiasing &&
		!l.coordconv &&
		can_activate_array_bf16_ongpu(l.activation));
#else
	(void)net;
	(void)l;
	(void)cache_bf16_output;
	return false;
#endif
}

static inline float *get_lowp_convolution_weights_gpu16(Darknet::Layer & l, const bool fp8_layer_enabled)
{
	if (fp8_layer_enabled && use_bf16_master_weight_storage(l) && l.weights_conv_gpu16)
	{
		return l.weights_conv_gpu16;
	}

	return l.weights_gpu16;
}

class ScopedConvolutionalWeightsF32Scratch
{
public:
	ScopedConvolutionalWeightsF32Scratch(Darknet::Layer & layer, bool needs_scratch)
		: l(layer), original_weights_gpu(layer.weights_gpu), scratch_weights_gpu(nullptr)
	{
		if (needs_scratch && use_bf16_master_weight_storage(layer))
		{
			scratch_weights_gpu = cuda_make_array(NULL, layer.nweights);
			cuda_convert_bf16_to_f32(layer.weights_gpu16, layer.nweights, scratch_weights_gpu);
			l.weights_gpu = scratch_weights_gpu;
		}
	}

	~ScopedConvolutionalWeightsF32Scratch()
	{
		if (scratch_weights_gpu)
		{
			cuda_free(scratch_weights_gpu);
			l.weights_gpu = original_weights_gpu;
		}
	}

	bool active() const
	{
		return (scratch_weights_gpu != nullptr);
	}

private:
	Darknet::Layer & l;
	float * original_weights_gpu;
	float * scratch_weights_gpu;
};

__global__ void bf16_kahan_sgd_kernel(
	int n,
	float lr,
	float decay_factor,
	float momentum_factor,
	float * weight_updates,
	__nv_bfloat16 * weights_bf16,
	__nv_bfloat16 * compensation,
	float * weights_f32_mirror)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;

	float w = __bfloat162float(weights_bf16[i]);
	float c = __bfloat162float(compensation[i]);

	if (!isfinite(w)) w = 0.0f;
	if (!isfinite(c)) c = 0.0f;

	float grad = weight_updates[i];
	grad += decay_factor * w;

	float update = lr * grad + c;
	float new_w = w + update;
	float new_c = (w - new_w) + update;

	weights_bf16[i] = __float2bfloat16(new_w);
	compensation[i] = __float2bfloat16(new_c);
	if (weights_f32_mirror) weights_f32_mirror[i] = new_w;
	weight_updates[i] = grad * momentum_factor;
}

void cuda_bf16_kahan_sgd(int n, float lr, float decay_factor, float momentum_factor,
	float * weight_updates, float * weights_bf16, float * compensation_bf16, float * weights_f32_mirror)
{
	TAT(TATPARMS);
	bf16_kahan_sgd_kernel <<< get_number_of_blocks(n, BLOCK), BLOCK, 0, get_cuda_stream() >>> (
		n, lr, decay_factor, momentum_factor,
		weight_updates,
		reinterpret_cast<__nv_bfloat16 *>(weights_bf16),
		reinterpret_cast<__nv_bfloat16 *>(compensation_bf16),
		weights_f32_mirror);
	CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void bf16_kahan_commit_kernel(
	int n,
	const float * weights_f32,
	__nv_bfloat16 * weights_bf16,
	__nv_bfloat16 * compensation)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;

	float w = weights_f32[i];
	float c = __bfloat162float(compensation[i]);
	float compensated = w + c;
	__nv_bfloat16 bf16_val = __float2bfloat16(compensated);
	float new_c = compensated - __bfloat162float(bf16_val);

	weights_bf16[i] = bf16_val;
	compensation[i] = __float2bfloat16(new_c);
}

void cuda_bf16_kahan_commit(int n, const float * weights_f32, float * weights_bf16, float * compensation_bf16)
{
	TAT(TATPARMS);
	bf16_kahan_commit_kernel <<< get_number_of_blocks(n, BLOCK), BLOCK, 0, get_cuda_stream() >>> (
		n, weights_f32,
		reinterpret_cast<__nv_bfloat16 *>(weights_bf16),
		reinterpret_cast<__nv_bfloat16 *>(compensation_bf16));
	CHECK_CUDA(cudaPeekAtLastError());
}
#endif

#if defined(DARKNET_GPU_CUDA) && (CUDART_VERSION >= 12000)
__global__ void cuda_f32_to_fp8(float* input_f32, size_t size, __nv_fp8_e4m3 *output_fp8)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{
		// Use __NV_SATFINITE to clamp overflows to ±448 instead of NaN
		__nv_fp8_storage_t s = __nv_cvt_float_to_fp8(input_f32[idx], __NV_SATFINITE, __NV_E4M3);
		output_fp8[idx] = *reinterpret_cast<__nv_fp8_e4m3*>(&s);
	}
}

void cuda_convert_f32_to_fp8(float* input_f32, size_t size, float *output_fp8)
{
	TAT(TATPARMS);

	cuda_f32_to_fp8 <<< get_number_of_blocks(size, BLOCK), BLOCK, 0, get_cuda_stream() >>> (input_f32, size, (__nv_fp8_e4m3 *)output_fp8);
	CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void cuda_fp8_to_f32(__nv_fp8_e4m3* input_fp8, size_t size, float *output_f32)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) output_f32[idx] = static_cast<float>(input_fp8[idx]);
}

void cuda_convert_fp8_to_f32(float* input_fp8, size_t size, float *output_f32)
{
	TAT(TATPARMS);

	cuda_fp8_to_f32 <<< get_number_of_blocks(size, BLOCK), BLOCK, 0, get_cuda_stream() >>> ((__nv_fp8_e4m3 *)input_fp8, size, output_f32);
	CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void cuda_f32_to_bf16_via_fp8_kernel(float* input_f32, size_t size, __nv_bfloat16 *output_bf16)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{
		// Use __NV_SATFINITE to clamp overflows to ±448 instead of NaN
		const __nv_fp8_storage_t s = __nv_cvt_float_to_fp8(input_f32[idx], __NV_SATFINITE, __NV_E4M3);
		const __half_raw hr = __nv_cvt_fp8_to_halfraw(s, __NV_E4M3);
		output_bf16[idx] = __float2bfloat16(__half2float(*reinterpret_cast<const __half *>(&hr)));
	}
}

void cuda_convert_f32_to_bf16_via_fp8(float* input_f32, size_t size, float *output_bf16)
{
	TAT(TATPARMS);

	cuda_f32_to_bf16_via_fp8_kernel <<< get_number_of_blocks(size, BLOCK), BLOCK, 0, get_cuda_stream() >>> (input_f32, size, (__nv_bfloat16 *)output_bf16);
	CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void fp8_reduce_absmax_kernel(const float *input_f32, size_t size, float *amax_out)
{
	float local_max = 0.0f;
	for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += blockDim.x * gridDim.x)
	{
		const float v = input_f32[idx];
		if (isfinite(v))
		{
			local_max = fmaxf(local_max, fabsf(v));
		}
	}

	__shared__ float smem[BLOCK];
	smem[threadIdx.x] = local_max;
	__syncthreads();

	for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if (threadIdx.x < stride)
		{
			smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
		}
		__syncthreads();
	}

	if (threadIdx.x == 0)
	{
		float result = smem[0];
		if (!isfinite(result)) result = 0.0f;  // NaN guard: prevent poisoning the amax
		atomicMax((unsigned int *)amax_out, __float_as_uint(result));
	}
}

__global__ void fp8_reduce_absmax_bf16_kernel(const __nv_bfloat16 *input_bf16, size_t size, float *amax_out)
{
	float local_max = 0.0f;
	for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += blockDim.x * gridDim.x)
	{
		const float v = __bfloat162float(input_bf16[idx]);
		if (isfinite(v))
		{
			local_max = fmaxf(local_max, fabsf(v));
		}
	}

	__shared__ float smem[BLOCK];
	smem[threadIdx.x] = local_max;
	__syncthreads();

	for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if (threadIdx.x < stride)
		{
			smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
		}
		__syncthreads();
	}

	if (threadIdx.x == 0)
	{
		float result = smem[0];
		if (!isfinite(result)) result = 0.0f;
		atomicMax((unsigned int *)amax_out, __float_as_uint(result));
	}
}

__global__ void fp8_update_scale_from_amax_kernel(float *scale_gpu, float *amax_ema_gpu, float decay, float fp8_max, float min_scale)
{
	float amax = *scale_gpu;
	float amax_ema = *amax_ema_gpu;

	if (!isfinite(amax) || amax < 0.0f) amax = 0.0f;
	if (!isfinite(amax_ema) || amax_ema < 0.0f) amax_ema = 0.0f;

	amax_ema = fmaxf(amax_ema * decay, amax);
	float scale = fmaxf(amax_ema / fp8_max, min_scale);
	if (!isfinite(scale))
	{
		scale = 1.0f;
		amax_ema = fp8_max;
	}

	*amax_ema_gpu = amax_ema;
	*scale_gpu = scale;
}

template<typename TFP8>
__device__ inline TFP8 safe_cast_to_fp8(float val);

template<>
__device__ inline __nv_fp8_e4m3 safe_cast_to_fp8<__nv_fp8_e4m3>(float val) {
	__nv_fp8_storage_t s = __nv_cvt_float_to_fp8(val, __NV_SATFINITE, __NV_E4M3);
	return *reinterpret_cast<__nv_fp8_e4m3*>(&s);
}

template<>
__device__ inline __nv_fp8_e5m2 safe_cast_to_fp8<__nv_fp8_e5m2>(float val) {
	__nv_fp8_storage_t s = __nv_cvt_float_to_fp8(val, __NV_SATFINITE, __NV_E5M2);
	return *reinterpret_cast<__nv_fp8_e5m2*>(&s);
}

template<typename TFP8>
__global__ void fp8_quantize_scaled_kernel(const float *input_f32, size_t size, const float *scale_gpu, float fp8_max, TFP8 *output_fp8)
{
	const float scale = fmaxf(*scale_gpu, k_fp8_min_scale);
	const float inv_scale = 1.0f / scale;

	for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += blockDim.x * gridDim.x)
	{
		float val = input_f32[idx];
		if (!isfinite(val))
		{
			val = 0.0f;
		}

		float q = val * inv_scale;
		q = fminf(fmaxf(q, -fp8_max), fp8_max);
		output_fp8[idx] = safe_cast_to_fp8<TFP8>(q);
	}
}

template<typename TFP8>
__global__ void fp8_dequantize_to_bf16_kernel(const TFP8 *input_fp8, size_t size, const float *scale_gpu, __nv_bfloat16 *output_bf16)
{
	const float scale = fmaxf(*scale_gpu, k_fp8_min_scale);

	for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += blockDim.x * gridDim.x)
	{
		const float val = static_cast<float>(input_fp8[idx]) * scale;
		output_bf16[idx] = __float2bfloat16(val);
	}
}

template<typename TFP8>
__global__ void fp8_f32_to_bf16_scaled_kernel(const float *input_f32, size_t size, const float *scale_gpu, float fp8_max, __nv_bfloat16 *output_bf16)
{
	const float scale = fmaxf(*scale_gpu, k_fp8_min_scale);
	const float inv_scale = 1.0f / scale;

	for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += blockDim.x * gridDim.x)
	{
		float val = input_f32[idx];
		if (!isfinite(val))
		{
			val = 0.0f;
		}

		float q = val * inv_scale;
		q = fminf(fmaxf(q, -fp8_max), fp8_max);
		const TFP8 q8 = safe_cast_to_fp8<TFP8>(q);
		output_bf16[idx] = __float2bfloat16(static_cast<float>(q8) * scale);
	}
}

template<typename TFP8>
__global__ void fp8_quantize_bf16_scaled_kernel(const __nv_bfloat16 *input_bf16, size_t size, const float *scale_gpu, float fp8_max, TFP8 *output_fp8)
{
	const float scale = fmaxf(*scale_gpu, k_fp8_min_scale);
	const float inv_scale = 1.0f / scale;

	for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += blockDim.x * gridDim.x)
	{
		float val = __bfloat162float(input_bf16[idx]);
		if (!isfinite(val))
		{
			val = 0.0f;
		}

		float q = val * inv_scale;
		q = fminf(fmaxf(q, -fp8_max), fp8_max);
		output_fp8[idx] = safe_cast_to_fp8<TFP8>(q);
	}
}

template<typename TFP8>
__global__ void fp8_bf16_to_bf16_scaled_kernel(const __nv_bfloat16 *input_bf16, size_t size, const float *scale_gpu, float fp8_max, __nv_bfloat16 *output_bf16)
{
	const float scale = fmaxf(*scale_gpu, k_fp8_min_scale);
	const float inv_scale = 1.0f / scale;

	for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += blockDim.x * gridDim.x)
	{
		float val = __bfloat162float(input_bf16[idx]);
		if (!isfinite(val))
		{
			val = 0.0f;
		}

		float q = val * inv_scale;
		q = fminf(fmaxf(q, -fp8_max), fp8_max);
		const TFP8 q8 = safe_cast_to_fp8<TFP8>(q);
		output_bf16[idx] = __float2bfloat16(static_cast<float>(q8) * scale);
	}
}

void cuda_quantize_f32_to_fp8_and_dequantize_bf16(float *input_f32, size_t size, uint8_t *output_fp8, float *scale_gpu, float *amax_ema_gpu, int fp8_format, int update_scale, float *output_bf16)
{
	TAT(TATPARMS);

	if (!input_f32 || !output_bf16 || size == 0)
	{
		return;
	}

	if (!scale_gpu || !amax_ema_gpu)
	{
		cuda_convert_f32_to_bf16(input_f32, size, output_bf16);
		return;
	}

	const float fp8_max = (fp8_format == 1 ? k_fp8_e5m2_max : k_fp8_e4m3_max);
	const int blocks = get_number_of_blocks(size, BLOCK);
	if (update_scale)
	{
		CHECK_CUDA(cudaMemsetAsync(scale_gpu, 0, sizeof(float), get_cuda_stream()));
		fp8_reduce_absmax_kernel<<<blocks, BLOCK, 0, get_cuda_stream()>>>(input_f32, size, scale_gpu);
		fp8_update_scale_from_amax_kernel<<<1, 1, 0, get_cuda_stream()>>>(scale_gpu, amax_ema_gpu, k_fp8_ema_decay, fp8_max, k_fp8_min_scale);
	}

	if (fp8_format == 1)
	{
		if (output_fp8)
		{
			fp8_quantize_scaled_kernel<__nv_fp8_e5m2><<<blocks, BLOCK, 0, get_cuda_stream()>>>(input_f32, size, scale_gpu, fp8_max, (__nv_fp8_e5m2 *)output_fp8);
			fp8_dequantize_to_bf16_kernel<__nv_fp8_e5m2><<<blocks, BLOCK, 0, get_cuda_stream()>>>((const __nv_fp8_e5m2 *)output_fp8, size, scale_gpu, (__nv_bfloat16 *)output_bf16);
		}
		else
		{
			fp8_f32_to_bf16_scaled_kernel<__nv_fp8_e5m2><<<blocks, BLOCK, 0, get_cuda_stream()>>>(input_f32, size, scale_gpu, fp8_max, (__nv_bfloat16 *)output_bf16);
		}
	}
	else
	{
		if (output_fp8)
		{
			fp8_quantize_scaled_kernel<__nv_fp8_e4m3><<<blocks, BLOCK, 0, get_cuda_stream()>>>(input_f32, size, scale_gpu, fp8_max, (__nv_fp8_e4m3 *)output_fp8);
			fp8_dequantize_to_bf16_kernel<__nv_fp8_e4m3><<<blocks, BLOCK, 0, get_cuda_stream()>>>((const __nv_fp8_e4m3 *)output_fp8, size, scale_gpu, (__nv_bfloat16 *)output_bf16);
		}
		else
		{
			fp8_f32_to_bf16_scaled_kernel<__nv_fp8_e4m3><<<blocks, BLOCK, 0, get_cuda_stream()>>>(input_f32, size, scale_gpu, fp8_max, (__nv_bfloat16 *)output_bf16);
		}
	}

	CHECK_CUDA(cudaPeekAtLastError());
}

void cuda_quantize_bf16_to_fp8_and_dequantize_bf16(float *input_bf16, size_t size, uint8_t *output_fp8, float *scale_gpu, float *amax_ema_gpu, int fp8_format, int update_scale, float *output_bf16)
{
	TAT(TATPARMS);

	if (!input_bf16 || !output_bf16 || size == 0)
	{
		return;
	}

	if (!scale_gpu || !amax_ema_gpu)
	{
		CHECK_CUDA(cudaMemcpyAsync(output_bf16, input_bf16, sizeof(__nv_bfloat16) * size, cudaMemcpyDeviceToDevice, get_cuda_stream()));
		return;
	}

	const float fp8_max = (fp8_format == 1 ? k_fp8_e5m2_max : k_fp8_e4m3_max);
	const int blocks = get_number_of_blocks(size, BLOCK);
	if (update_scale)
	{
		CHECK_CUDA(cudaMemsetAsync(scale_gpu, 0, sizeof(float), get_cuda_stream()));
		fp8_reduce_absmax_bf16_kernel<<<blocks, BLOCK, 0, get_cuda_stream()>>>((const __nv_bfloat16 *)input_bf16, size, scale_gpu);
		fp8_update_scale_from_amax_kernel<<<1, 1, 0, get_cuda_stream()>>>(scale_gpu, amax_ema_gpu, k_fp8_ema_decay, fp8_max, k_fp8_min_scale);
	}

	if (fp8_format == 1)
	{
		if (output_fp8)
		{
			fp8_quantize_bf16_scaled_kernel<__nv_fp8_e5m2><<<blocks, BLOCK, 0, get_cuda_stream()>>>((const __nv_bfloat16 *)input_bf16, size, scale_gpu, fp8_max, (__nv_fp8_e5m2 *)output_fp8);
			fp8_dequantize_to_bf16_kernel<__nv_fp8_e5m2><<<blocks, BLOCK, 0, get_cuda_stream()>>>((const __nv_fp8_e5m2 *)output_fp8, size, scale_gpu, (__nv_bfloat16 *)output_bf16);
		}
		else
		{
			fp8_bf16_to_bf16_scaled_kernel<__nv_fp8_e5m2><<<blocks, BLOCK, 0, get_cuda_stream()>>>((const __nv_bfloat16 *)input_bf16, size, scale_gpu, fp8_max, (__nv_bfloat16 *)output_bf16);
		}
	}
	else
	{
		if (output_fp8)
		{
			fp8_quantize_bf16_scaled_kernel<__nv_fp8_e4m3><<<blocks, BLOCK, 0, get_cuda_stream()>>>((const __nv_bfloat16 *)input_bf16, size, scale_gpu, fp8_max, (__nv_fp8_e4m3 *)output_fp8);
			fp8_dequantize_to_bf16_kernel<__nv_fp8_e4m3><<<blocks, BLOCK, 0, get_cuda_stream()>>>((const __nv_fp8_e4m3 *)output_fp8, size, scale_gpu, (__nv_bfloat16 *)output_bf16);
		}
		else
		{
			fp8_bf16_to_bf16_scaled_kernel<__nv_fp8_e4m3><<<blocks, BLOCK, 0, get_cuda_stream()>>>((const __nv_bfloat16 *)input_bf16, size, scale_gpu, fp8_max, (__nv_bfloat16 *)output_bf16);
		}
	}

	CHECK_CUDA(cudaPeekAtLastError());
}

/// Current-scaling quantize kernel: reads the just-computed amax from amax_gpu,
/// derives the scale inline, then quantizes to FP8 and dequantizes back to BF16.
/// Eliminates the EMA lag of delayed scaling entirely.
template<typename TFP8>
__global__ void fp8_quantize_current_scale_kernel(
	const float *input_f32, size_t size, const float *amax_gpu,
	float fp8_max, uint8_t *output_fp8, __nv_bfloat16 *output_bf16)
{
	const float scale = fmaxf(*amax_gpu, k_fp8_min_scale);
	const float inv_scale = 1.0f / scale;

	for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += blockDim.x * gridDim.x)
	{
		float val = input_f32[idx];
		if (!isfinite(val)) val = 0.0f;

		float q = val * inv_scale;
		q = fminf(fmaxf(q, -fp8_max), fp8_max);
		const TFP8 q8 = safe_cast_to_fp8<TFP8>(q);
		const float dq = static_cast<float>(q8) * scale;
		output_bf16[idx] = __float2bfloat16(dq);

		if (output_fp8)
		{
			output_fp8[idx] = *reinterpret_cast<const uint8_t*>(&q8);
		}
	}
}

template<typename TFP8>
__global__ void fp8_quantize_current_scale_bf16_kernel(
	const __nv_bfloat16 *input_bf16, size_t size, const float *amax_gpu,
	float fp8_max, uint8_t *output_fp8, __nv_bfloat16 *output_bf16)
{
	const float scale = fmaxf(*amax_gpu, k_fp8_min_scale);
	const float inv_scale = 1.0f / scale;

	for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += blockDim.x * gridDim.x)
	{
		float val = __bfloat162float(input_bf16[idx]);
		if (!isfinite(val)) val = 0.0f;

		float q = val * inv_scale;
		q = fminf(fmaxf(q, -fp8_max), fp8_max);
		const TFP8 q8 = safe_cast_to_fp8<TFP8>(q);
		const float dq = static_cast<float>(q8) * scale;
		output_bf16[idx] = __float2bfloat16(dq);

		if (output_fp8)
		{
			output_fp8[idx] = *reinterpret_cast<const uint8_t*>(&q8);
		}
	}
}

/// Two-pass current scaling: compute amax NOW, then quantize with that amax.
/// No EMA, no delay — the scale is always computed from the exact tensor being quantized.
void cuda_fp8_current_scale_quantize_bf16(
	float *input_f32, size_t size,
	uint8_t *output_fp8,
	float *amax_gpu,
	float *amax_ema_gpu,
	int fp8_format,
	float *output_bf16)
{
	TAT(TATPARMS);

	if (!input_f32 || !output_bf16 || size == 0)
	{
		return;
	}

	if (!amax_gpu)
	{
		cuda_convert_f32_to_bf16(input_f32, size, output_bf16);
		return;
	}

	const float fp8_max = (fp8_format == 1 ? k_fp8_e5m2_max : k_fp8_e4m3_max);
	const int blocks = get_number_of_blocks(size, BLOCK);

	// Pass 1: compute current amax of the tensor
	CHECK_CUDA(cudaMemsetAsync(amax_gpu, 0, sizeof(float), get_cuda_stream()));
	fp8_reduce_absmax_kernel<<<blocks, BLOCK, 0, get_cuda_stream()>>>(input_f32, size, amax_gpu);
	fp8_update_scale_from_amax_kernel<<<1, 1, 0, get_cuda_stream()>>>(amax_gpu, amax_ema_gpu, 0.0f, fp8_max, k_fp8_min_scale);

	// Pass 2: quantize using the just-computed amax (no EMA, no delay)
	if (fp8_format == 1)
	{
		fp8_quantize_current_scale_kernel<__nv_fp8_e5m2><<<blocks, BLOCK, 0, get_cuda_stream()>>>(
			input_f32, size, amax_gpu, fp8_max, output_fp8, (__nv_bfloat16 *)output_bf16);
	}
	else
	{
		fp8_quantize_current_scale_kernel<__nv_fp8_e4m3><<<blocks, BLOCK, 0, get_cuda_stream()>>>(
			input_f32, size, amax_gpu, fp8_max, output_fp8, (__nv_bfloat16 *)output_bf16);
	}

	CHECK_CUDA(cudaPeekAtLastError());
}

void cuda_fp8_current_scale_quantize_bf16_from_bf16(
	float *input_bf16, size_t size,
	uint8_t *output_fp8,
	float *amax_gpu,
	float *amax_ema_gpu,
	int fp8_format,
	float *output_bf16)
{
	TAT(TATPARMS);

	if (!input_bf16 || !output_bf16 || size == 0)
	{
		return;
	}

	if (!amax_gpu)
	{
		CHECK_CUDA(cudaMemcpyAsync(output_bf16, input_bf16, sizeof(__nv_bfloat16) * size, cudaMemcpyDeviceToDevice, get_cuda_stream()));
		return;
	}

	const float fp8_max = (fp8_format == 1 ? k_fp8_e5m2_max : k_fp8_e4m3_max);
	const int blocks = get_number_of_blocks(size, BLOCK);

	CHECK_CUDA(cudaMemsetAsync(amax_gpu, 0, sizeof(float), get_cuda_stream()));
	fp8_reduce_absmax_bf16_kernel<<<blocks, BLOCK, 0, get_cuda_stream()>>>((const __nv_bfloat16 *)input_bf16, size, amax_gpu);
	fp8_update_scale_from_amax_kernel<<<1, 1, 0, get_cuda_stream()>>>(amax_gpu, amax_ema_gpu, 0.0f, fp8_max, k_fp8_min_scale);

	if (fp8_format == 1)
	{
		fp8_quantize_current_scale_bf16_kernel<__nv_fp8_e5m2><<<blocks, BLOCK, 0, get_cuda_stream()>>>(
			(const __nv_bfloat16 *)input_bf16, size, amax_gpu, fp8_max, output_fp8, (__nv_bfloat16 *)output_bf16);
	}
	else
	{
		fp8_quantize_current_scale_bf16_kernel<__nv_fp8_e4m3><<<blocks, BLOCK, 0, get_cuda_stream()>>>(
			(const __nv_bfloat16 *)input_bf16, size, amax_gpu, fp8_max, output_fp8, (__nv_bfloat16 *)output_bf16);
	}

	CHECK_CUDA(cudaPeekAtLastError());
}

void cuda_quantize_f32_to_fp8_bf16_by_policy(
	float *input_f32, size_t size,
	uint8_t *output_fp8,
	float *scale_gpu,
	float *amax_ema_gpu,
	int fp8_format,
	int use_current_scaling,
	int update_scale,
	float *output_bf16)
{
	TAT(TATPARMS);

	if (use_current_scaling)
	{
		cuda_fp8_current_scale_quantize_bf16(input_f32, size, output_fp8, scale_gpu, amax_ema_gpu, fp8_format, output_bf16);
	}
	else
	{
		cuda_quantize_f32_to_fp8_and_dequantize_bf16(input_f32, size, output_fp8, scale_gpu, amax_ema_gpu, fp8_format, update_scale, output_bf16);
	}
}

void cuda_quantize_bf16_to_fp8_bf16_by_policy(
	float *input_bf16, size_t size,
	uint8_t *output_fp8,
	float *scale_gpu,
	float *amax_ema_gpu,
	int fp8_format,
	int use_current_scaling,
	int update_scale,
	float *output_bf16)
{
	TAT(TATPARMS);

	if (use_current_scaling)
	{
		cuda_fp8_current_scale_quantize_bf16_from_bf16(input_bf16, size, output_fp8, scale_gpu, amax_ema_gpu, fp8_format, output_bf16);
	}
	else
	{
		cuda_quantize_bf16_to_fp8_and_dequantize_bf16(input_bf16, size, output_fp8, scale_gpu, amax_ema_gpu, fp8_format, update_scale, output_bf16);
	}
}

__nv_fp8_e4m3 *cuda_make_fp8_from_f32_array(float *src, size_t n)
{
	TAT(TATPARMS);

	__nv_fp8_e4m3 *dst8;
	size_t size = sizeof(__nv_fp8_e4m3) * n;
	CHECK_CUDA(cudaMalloc((void **)&dst8, size));
	if (src)
	{
		assert(n > 0);
		cuda_convert_f32_to_fp8(src, n, (float *)dst8);
	}
	if (!dst8)
	{
		darknet_fatal_error(DARKNET_LOC, "CUDA malloc failed (n=%d)", n);
	}
	return dst8;
}
#endif

static void reallocate_lowp_tensor(float** ptr, size_t* max_size, size_t required_size, bool is_bf16)
{
	if (*max_size < required_size)
	{
		*max_size = required_size;
		if (*ptr) cuda_free(*ptr);
		assert(*max_size > 0);
		*ptr = (float*)cuda_make_lowp_from_f32_array(NULL, *max_size, is_bf16);
	}
}

void forward_convolutional_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	if (l.train == 0) state.train = 0;

	if (l.stream >= 0)
	{
		switch_stream(l.stream);
	}

	if (l.wait_stream_id >= 0)
	{
		wait_stream(l.wait_stream_id);
	}

#ifdef CUDNN
	int iteration_num = get_current_iteration(state.net); // (*state.net.seen) / (state.net.batch*state.net.subdivisions);
	const int tensor_cores_min_iteration = std::max(0, state.net.tensor_cores_min_iteration);
	const bool bf16_compute_mode = (
		state.net.precision_mode == Darknet::PrecisionMode::BF16 ||
		state.net.precision_mode == Darknet::PrecisionMode::BF16_MASTER_KAHAN ||
		state.net.precision_mode == Darknet::PrecisionMode::FP8_BF16 ||
		state.net.cudnn_bfloat16 || state.net.cudnn_fp8);
	const bool allow_bf16_tensor_core_now = (!state.train || state.net.loss_scale > 1.0f || iteration_num >= tensor_cores_min_iteration);
	// cuDNN BF16 convolution is only safe on Tensor Core aligned shapes.
	const bool tc_aligned = ((l.c / std::max(1, l.groups)) % 8 == 0 && l.n % 8 == 0 && l.groups <= 1);
	const bool use_lowp_conv = (bf16_compute_mode && allow_bf16_tensor_core_now && tc_aligned);
#else
	const bool use_lowp_conv = false;
#endif
	ScopedConvolutionalWeightsF32Scratch weights_f32_scratch(l, (l.binary || l.xnor || !use_lowp_conv));

	//fill_ongpu(l.outputs*l.batch, 0, l.output_gpu, 1);
	if (l.binary)
	{
		binarize_weights_gpu(l.weights_gpu, l.n, (l.c / l.groups)*l.size*l.size, l.binary_weights_gpu);
		swap_binary(&l);
	}

	if (l.xnor)
	{
		if (!l.align_bit_weights_gpu || state.train)
		{
			fast_binarize_weights_gpu(l.weights_gpu, l.n, (l.c / l.groups)*l.size*l.size, l.binary_weights_gpu, l.mean_arr_gpu);
		}

		if (l.align_bit_weights_gpu && !state.train && l.c >= 32 && l.stride_x == l.stride_y)
		{
			int m = l.n / l.groups;
			int k = l.size*l.size*l.c / l.groups;
			int n = l.out_w*l.out_h;

			const int ldb_align = l.lda_align;
			const size_t new_ldb = k + (ldb_align - k%ldb_align); // (k / 8 + 1) * 8;

			if (l.c % 32 == 0)
			{
				const int new_c = l.c / 32;

				repack_input_gpu_bin(state.input, (uint32_t *)l.align_workspace_gpu, l.w, l.h, l.c);

				im2col_ongpu(l.align_workspace_gpu, new_c, l.h, l.w, l.size, l.stride, l.pad, state.workspace);

				int new_k = l.size*l.size*l.c / 32;

				transpose_uint32_gpu((uint32_t *)state.workspace, (uint32_t *)l.transposed_align_workspace_gpu, new_k, n, n, new_ldb);
				gemm_nn_custom_bin_mean_transposed_gpu(m, n, k,
					(unsigned char *)l.align_bit_weights_gpu, new_ldb, (unsigned char *)l.transposed_align_workspace_gpu,
					new_ldb, l.output_gpu, n, l.mean_arr_gpu, l.biases_gpu, l.activation == LEAKY,
					l.bin_conv_shortcut_in_gpu, l.bin_conv_shortcut_out_gpu);
			}
			else
			{
				int i = 0;
				{
					im2col_align_ongpu(state.input + i*l.c*l.h*l.w, l.c, l.h, l.w, l.size, l.stride, l.pad, l.align_workspace_gpu, l.bit_align);

					// should be optimized
					float_to_bit_gpu(l.align_workspace_gpu, (unsigned char *)state.workspace, l.align_workspace_size);
				}
				transpose_bin_gpu((unsigned char *)state.workspace, (unsigned char *)l.transposed_align_workspace_gpu, k, n, l.bit_align, new_ldb, 8);

				gemm_nn_custom_bin_mean_transposed_gpu(m, n, k,
						(unsigned char *)l.align_bit_weights_gpu, new_ldb, (unsigned char *)l.transposed_align_workspace_gpu,
						new_ldb, l.output_gpu, n, l.mean_arr_gpu, l.biases_gpu, l.activation == LEAKY,
						l.bin_conv_shortcut_in_gpu, l.bin_conv_shortcut_out_gpu);
			}

			if (l.activation == SWISH) activate_array_swish_ongpu(l.output_gpu, l.outputs*l.batch, l.activation_input_gpu, l.output_gpu);
			else if (l.activation == MISH) activate_array_mish_ongpu(l.output_gpu, l.outputs*l.batch, l.activation_input_gpu, l.output_gpu);
			else if (l.activation == HARD_MISH) activate_array_hard_mish_ongpu(l.output_gpu, l.outputs*l.batch, l.activation_input_gpu, l.output_gpu);
			else if (l.activation == NORM_CHAN) activate_array_normalize_channels_ongpu(l.output_gpu, l.outputs*l.batch, l.batch, l.out_c, l.out_w*l.out_h, l.output_gpu);
			else if (l.activation == NORM_CHAN_SOFTMAX) activate_array_normalize_channels_softmax_ongpu(l.output_gpu, l.outputs*l.batch, l.batch, l.out_c, l.out_w*l.out_h, l.output_gpu, 0);
			else if (l.activation == NORM_CHAN_SOFTMAX_MAXVAL) activate_array_normalize_channels_softmax_ongpu(l.output_gpu, l.outputs*l.batch, l.batch, l.out_c, l.out_w*l.out_h, l.output_gpu, 1);
			else if (l.activation != LINEAR && l.activation != LEAKY) activate_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation);
			return;
		}
	}

	if (l.xnor)
	{
		swap_binary(&l);
		binarize_gpu(state.input, l.c*l.h*l.w*l.batch, l.binary_input_gpu);
		state.input = l.binary_input_gpu;
	}

	bool activation_applied = false;
	bool cached_bf16_output = false;

	//fill_ongpu(l.outputs*l.batch, 0, l.output_gpu, 1);

#ifdef CUDNN
	//float one = 1;    // alpha[0], beta[0] is float for HALF and FLOAT
	float alpha = 1, beta = 0;
	const bool fp8_layer_enabled = (state.net.cudnn_fp8 && l.fp8_enabled > 0);
	if (use_lowp_conv && !l.xnor)
	{
		const bool cache_bf16_output = next_layer_can_consume_cached_bf16_output(state.net, l, state.index, state.train);

		// Note: For improved performance it is advised to use beta[0] = 0.0.
		// For Tensor Core: cudnnSetConvolutionMathType() where cudnnMathType_t mathType = CUDNN_TENSOR_OP_MATH;
		// 1. or CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM and use CUDNN_DATA_HALF
		// 2. or CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED
		// More: http://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#tensor_ops

		const size_t input16_size = l.batch*l.c*l.w*l.h;
		const size_t output16_size = l.batch*l.out_c*l.out_h*l.out_w;
		assert(input16_size > 0);
		// fp8_update_scale_now is needed by the FP8 input-quantization path inside the else block below.
		const int fp8_scale_update_interval = std::max(1, state.net.fp8_scale_update_interval);
		const int64_t lowp_step = static_cast<int64_t>(get_current_iteration(state.net)) * std::max(1, state.net.subdivisions) + std::max(0, state.net.current_subdivision);
		const int fp8_update_scale_now = ((lowp_step % fp8_scale_update_interval) == 0 ? 1 : 0);

		// When state.input is already BF16 (from the previous layer's output_gpu16),
		// use it directly — no allocation or conversion needed.
		// FP8 path still needs FP32 input for quantization, so it always converts.
		float *input16;
		if (state.input_is_bf16 && !fp8_layer_enabled)
		{
			input16 = state.input; // zero-copy: already BF16 from prev layer's output_gpu16
		}
		else
		{
			reallocate_lowp_tensor(state.net.input16_gpu, state.net.max_input16_size, input16_size, bf16_compute_mode);
			input16 = *state.net.input16_gpu;

			assert(input16_size > 0);
#if defined(DARKNET_GPU_CUDA) && (CUDART_VERSION >= 12000)
			if (fp8_layer_enabled)
			{
				if (state.input_is_bf16)
				{
					cuda_quantize_bf16_to_fp8_bf16_by_policy(
						state.input, input16_size,
						(uint8_t *)l.act_fp8_gpu,
						l.x_scale_gpu,
						l.x_amax_ema_gpu,
						l.fp8_format,
						state.net.fp8_current_scaling,
						fp8_update_scale_now,
						input16);
				}
				else
				{
					cuda_quantize_f32_to_fp8_bf16_by_policy(
						state.input, input16_size,
						(uint8_t *)l.act_fp8_gpu,
						l.x_scale_gpu,
						l.x_amax_ema_gpu,
						l.fp8_format,
						state.net.fp8_current_scaling,
						fp8_update_scale_now,
						input16);
				}

				if (state.net.fp8_debug && (get_current_iteration(state.net) % 100 == 0))
				{
					float x_metric = 0.0f, w_metric = 0.0f;
					float x_aux = 0.0f, w_aux = 0.0f;
					CHECK_CUDA(cudaMemcpyAsync(&x_metric, l.x_scale_gpu, sizeof(float), cudaMemcpyDeviceToHost, get_cuda_stream()));
					CHECK_CUDA(cudaMemcpyAsync(&w_metric, l.w_scale_gpu, sizeof(float), cudaMemcpyDeviceToHost, get_cuda_stream()));
					if (!state.net.fp8_current_scaling)
					{
						CHECK_CUDA(cudaMemcpyAsync(&x_aux, l.x_amax_ema_gpu, sizeof(float), cudaMemcpyDeviceToHost, get_cuda_stream()));
						CHECK_CUDA(cudaMemcpyAsync(&w_aux, l.w_amax_ema_gpu, sizeof(float), cudaMemcpyDeviceToHost, get_cuda_stream()));
					}
					CHECK_CUDA(cudaStreamSynchronize(get_cuda_stream()));
					const float fp8_max = (l.fp8_format == 1 ? k_fp8_e5m2_max : k_fp8_e4m3_max);
					if (state.net.fp8_current_scaling)
					{
						printf("[FP8 Debug] Layer %d: x_amax=%.4f w_amax=%.4f x_scale=%.6f w_scale=%.6f (current)\n",
							l.index, x_metric, w_metric,
							fmaxf(x_metric / fp8_max, k_fp8_min_scale),
							fmaxf(w_metric / fp8_max, k_fp8_min_scale));
					}
					else
					{
						printf("[FP8 Debug] Layer %d: x_amax_ema=%.4f w_amax_ema=%.4f x_scale=%.6f w_scale=%.6f (delayed)\n",
							l.index, x_aux, w_aux, x_metric, w_metric);
					}
				}
			}
			else
#endif
			{
				cuda_convert_f32_to_lowp(state.input, input16_size, input16, bf16_compute_mode);
			}
		}

		reallocate_lowp_tensor(state.net.output16_gpu, state.net.max_output16_size, output16_size, bf16_compute_mode);
		float *output16 = *state.net.output16_gpu;
		float *conv_weights16 = get_lowp_convolution_weights_gpu16(l, fp8_layer_enabled);

			const auto cudnn_status = cudnnConvolutionForward(cudnn_handle(),
				&alpha,
				l.srcTensorDesc16,
				input16,
				l.weightDesc16,
				conv_weights16,
				l.convDesc,
				l.fw_algo16,
				state.workspace,
				l.workspace_size,
				&beta,
				l.dstTensorDesc16,
				output16);
			if (cudnn_status != CUDNN_STATUS_SUCCESS)
			{
				*cfg_and_state.output
					<< "BF16 forward failure: layer=" << l.index
					<< ", c=" << l.c
					<< ", n=" << l.n
					<< ", groups=" << l.groups
					<< ", size=" << l.size
					<< ", w=" << l.w
					<< ", h=" << l.h
					<< ", out_w=" << l.out_w
					<< ", out_h=" << l.out_h
					<< ", tc_aligned=" << tc_aligned
					<< ", workspace=" << l.workspace_size
					<< std::endl;
			}
			CHECK_CUDNN(cudnn_status);

			if (l.batch_normalize)
			{
				if (state.train && !state.net.adversarial) // Training
				{
					simple_copy_ongpu(l.outputs*l.batch / 2, output16, l.x_gpu);
					float one = 1.0f;
					float zero = 0.0f;
					// Batch-normalization can still take FP16/BF16 inputs and outputs, saving bandwidth,
					// while statistics and scaling remain in FP32.
					CHECK_CUDNN(cudnnBatchNormalizationForwardTraining(cudnn_handle(),
						CUDNN_BATCHNORM_SPATIAL,
						&one,
						&zero,
						l.normDstTensorDescF16,
						l.x_gpu,            // input
						l.normDstTensorDescF16,
						output16,            // output
						l.normTensorDesc,
						l.scales_gpu,       // input
						l.biases_gpu,       // input
						.01,
						l.rolling_mean_gpu,        // input/output (should be FP32)
						l.rolling_variance_gpu,    // input/output (should be FP32)
						.00001,
						l.mean_gpu,            // output (should be FP32) - optional cache to speedup cudnnBatchNormalizationBackward()
						l.variance_gpu));    // output (should be FP32) - optional cache to speedup cudnnBatchNormalizationBackward()

#if defined(DARKNET_GPU_CUDA) && (CUDART_VERSION >= 11000)
					if (can_materialize_post_activation_bf16_cache(state.net, l, cache_bf16_output))
					{
						activate_array_bf16_ongpu(output16, output16_size, l.activation);
						CHECK_CUDA(cudaMemcpyAsync(l.output_gpu16, output16, sizeof(__nv_bfloat16) * output16_size, cudaMemcpyDeviceToDevice, get_cuda_stream()));
						activation_applied = true;
						cached_bf16_output = true;
					}
#endif

					cuda_convert_lowp_to_f32(output16, output16_size, l.output_gpu, bf16_compute_mode);
			}
			else // Detection
			{
				cuda_convert_lowp_to_f32(output16, output16_size, l.output_gpu, bf16_compute_mode);
				normalize_gpu(l.output_gpu, l.rolling_mean_gpu, l.rolling_variance_gpu, l.batch, l.out_c, l.out_h*l.out_w);
				scale_bias_gpu(l.output_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);
				add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.out_c, l.out_w*l.out_h);
			}
		}
		else // BIAS only
		{
#if defined(DARKNET_GPU_CUDA) && (CUDART_VERSION >= 11000)
			if ((state.net.cudnn_bfloat16 || state.net.cudnn_fp8) && can_materialize_post_activation_bf16_cache(state.net, l, cache_bf16_output))
			{
				add_bias_activate_bf16_ongpu(output16, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h, l.activation);
				CHECK_CUDA(cudaMemcpyAsync(l.output_gpu16, output16, sizeof(__nv_bfloat16) * output16_size, cudaMemcpyDeviceToDevice, get_cuda_stream()));
				activation_applied = true;
				cached_bf16_output = true;
				cuda_convert_lowp_to_f32(output16, output16_size, l.output_gpu, true);
			}
			else
#endif
			{
				cuda_convert_lowp_to_f32(output16, output16_size, l.output_gpu, bf16_compute_mode);
				add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
			}
		}

	}
	else
	{
		CHECK_CUDNN(cudnnConvolutionForward(cudnn_handle(),
			&alpha, //&one,
			l.srcTensorDesc,
			state.input,
			l.weightDesc,
			l.weights_gpu,
			l.convDesc,
			l.fw_algo,
			state.workspace,
			l.workspace_size,
			&beta,  //&one,
			l.dstTensorDesc,
			l.output_gpu));

		//cudaDeviceSynchronize();
		if (l.batch_normalize) {
			forward_batchnorm_layer_gpu(l, state);
		}
		else {
			add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
		}
	//#endif    // CUDNN_HALF
	}


#else
	fill_ongpu(l.outputs*l.batch, 0, l.output_gpu, 1);

	int i, j;
	int m = l.n / l.groups;
	int k = l.size*l.size*l.c / l.groups;
	int n = l.out_w*l.out_h;

	for(i = 0; i < l.batch; ++i)
	{
		for (j = 0; j < l.groups; ++j)
		{
			//float *im = state.input + i*l.c*l.h*l.w;
			float *im = state.input + (i*l.groups + j)*l.c / l.groups*l.h*l.w;
			float *a = l.weights_gpu + j*l.nweights / l.groups;
			float *b = state.workspace;
			float *c = l.output_gpu + (i*l.groups + j)*n*m;
			if (l.size == 1 && l.stride == 1 && l.dilation == 1)
			{
				b = im;
			}
			else
			{
				//im2col_ongpu(im, l.c / l.groups, l.h, l.w, l.size, l.stride, l.pad, state.workspace);

				im2col_gpu_ext(im,          // input
					l.c / l.groups,         // input channels
					l.h, l.w,               // input size (h, w)
					l.size, l.size,         // kernel size (h, w)
					l.pad * l.dilation, l.pad * l.dilation,   // padding (h, w)
					l.stride_y, l.stride_x,     // stride (h, w)
					l.dilation, l.dilation, // dilation (h, w)
					state.workspace);       // output

			}
			//gemm_ongpu(0, 0, m, n, k, 1., a, k, b, n, 1., c + i*m*n, n);
			gemm_ongpu(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
		}
	}

	if (l.batch_normalize)
	{
		forward_batchnorm_layer_gpu(l, state);
	}
	else
	{
		add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
	}
#endif

//#ifndef CUDNN_HALF
//#endif // no CUDNN_HALF

	if (!activation_applied)
	{
		if (l.activation == SWISH) activate_array_swish_ongpu(l.output_gpu, l.outputs*l.batch, l.activation_input_gpu, l.output_gpu);
		else if (l.activation == MISH) activate_array_mish_ongpu(l.output_gpu, l.outputs*l.batch, l.activation_input_gpu, l.output_gpu);
		else if (l.activation == HARD_MISH) activate_array_hard_mish_ongpu(l.output_gpu, l.outputs*l.batch, l.activation_input_gpu, l.output_gpu);
		else if (l.activation == NORM_CHAN) activate_array_normalize_channels_ongpu(l.output_gpu, l.outputs*l.batch, l.batch, l.out_c, l.out_w*l.out_h, l.output_gpu);
		else if (l.activation == NORM_CHAN_SOFTMAX) activate_array_normalize_channels_softmax_ongpu(l.output_gpu, l.outputs*l.batch, l.batch, l.out_c, l.out_w*l.out_h, l.output_gpu, 0);
		else if (l.activation == NORM_CHAN_SOFTMAX_MAXVAL) activate_array_normalize_channels_softmax_ongpu(l.output_gpu, l.outputs*l.batch, l.batch, l.out_c, l.out_w*l.out_h, l.output_gpu, 1);
		else if (l.activation != LINEAR) activate_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation);
	}
	//if(l.dot > 0) dot_error_gpu(l);
	if(l.binary || l.xnor) swap_binary(&l);
	//cudaDeviceSynchronize();    // for correct profiling of performance

	if (state.net.try_fix_nan)
	{
		fix_nan_and_inf(l.output_gpu, l.outputs*l.batch);
	}

	if (l.assisted_excitation && state.train)
	{
		assisted_excitation_forward_gpu(l, state);
	}

	if (l.antialiasing)
	{
		Darknet::NetworkState s = { 0 };
		s.train = state.train;
		s.workspace = state.workspace;
		s.net = state.net;
		if (!state.train) s.index = state.index;  // don't use TC for training (especially without cuda_convert_f32_to_f16() )
		s.input = l.output_gpu;
		forward_convolutional_layer_gpu(*(l.input_layer), s);
		simple_copy_ongpu(l.outputs*l.batch, l.output_gpu, l.input_antialiasing_gpu);
		simple_copy_ongpu(l.input_layer->outputs*l.input_layer->batch, l.input_layer->output_gpu, l.output_gpu);
	}

	if (l.coordconv)
	{
		coord_conv_gpu(l.output_gpu, l.outputs*l.batch, l.out_w, l.out_h, l.out_c, l.batch, 0);
	}

	if (!cached_bf16_output && next_layer_can_consume_cached_bf16_output(state.net, l, state.index, state.train))
	{
		cuda_convert_f32_to_lowp(l.output_gpu, l.outputs*l.batch, l.output_gpu16, true);
	}

}


void backward_convolutional_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	if (l.coordconv) {
		coord_conv_gpu(l.delta_gpu, l.outputs*l.batch, l.out_w, l.out_h, l.out_c, l.batch, 1);
	}

	if (l.antialiasing) {
		Darknet::NetworkState s = { 0 };
		s.train = state.train;
		s.workspace = state.workspace;
		s.net = state.net;
		s.delta = l.delta_gpu;  // s.delta will be returned to l.delta_gpu
		s.input = l.input_antialiasing_gpu;
		//if (!state.train) s.index = state.index;  // don't use TC for training (especially without cuda_convert_f32_to_f16() )
		simple_copy_ongpu(l.input_layer->outputs*l.input_layer->batch, l.delta_gpu, l.input_layer->delta_gpu);
		backward_convolutional_layer_gpu(*(l.input_layer), s);

		simple_copy_ongpu(l.outputs*l.batch, l.input_antialiasing_gpu, l.output_gpu);
	}

	if(state.net.try_fix_nan) constrain_ongpu(l.outputs*l.batch, 1, l.delta_gpu, 1);

	if (l.activation == SWISH) gradient_array_swish_ongpu(l.output_gpu, l.outputs*l.batch, l.activation_input_gpu, l.delta_gpu);
	else if (l.activation == MISH) gradient_array_mish_ongpu(l.outputs*l.batch, l.activation_input_gpu, l.delta_gpu);
	else if (l.activation == HARD_MISH) gradient_array_hard_mish_ongpu(l.outputs*l.batch, l.activation_input_gpu, l.delta_gpu);
	else if (l.activation == NORM_CHAN_SOFTMAX || l.activation == NORM_CHAN_SOFTMAX_MAXVAL) gradient_array_normalize_channels_softmax_ongpu(l.output_gpu, l.outputs*l.batch, l.batch, l.out_c, l.out_w*l.out_h, l.delta_gpu);
	else if (l.activation == NORM_CHAN) gradient_array_normalize_channels_ongpu(l.output_gpu, l.outputs*l.batch, l.batch, l.out_c, l.out_w*l.out_h, l.delta_gpu);
	else gradient_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);

	if (!l.batch_normalize)
		backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h);

//#ifndef CUDNN_HALF
	//if(l.batch_normalize){
	//    backward_batchnorm_layer_gpu(l, state);
	//} else {
	//    //backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h);
	//}
//#endif // no CUDNN_HALF
	float *original_input = state.input;

#ifdef CUDNN
	int iteration_num = get_current_iteration(state.net); //(*state.net.seen) / (state.net.batch*state.net.subdivisions);
	const int tensor_cores_min_iteration = std::max(0, state.net.tensor_cores_min_iteration);
	const bool bf16_compute_mode = (
		state.net.precision_mode == Darknet::PrecisionMode::BF16 ||
		state.net.precision_mode == Darknet::PrecisionMode::BF16_MASTER_KAHAN ||
		state.net.precision_mode == Darknet::PrecisionMode::FP8_BF16 ||
		state.net.cudnn_bfloat16 || state.net.cudnn_fp8);
	const bool allow_bf16_tensor_core_now = (!state.train || state.net.loss_scale > 1.0f || iteration_num >= tensor_cores_min_iteration);
	const bool tc_aligned = ((l.c / std::max(1, l.groups)) % 8 == 0 && l.n % 8 == 0 && l.groups <= 1);
	const bool use_lowp_conv = (bf16_compute_mode && allow_bf16_tensor_core_now && tc_aligned);
#else
	const bool use_lowp_conv = false;
#endif
	ScopedConvolutionalWeightsF32Scratch weights_f32_scratch(l, (l.binary || l.xnor || !use_lowp_conv));

	if(l.xnor) state.input = l.binary_input_gpu;
#ifdef CUDNN
	float alpha = 1.0f;
	float beta = 0.0f;
	const bool fp8_layer_enabled = (state.net.cudnn_fp8 && l.fp8_enabled > 0);

	if (use_lowp_conv && !l.xnor)
	{
		const size_t input16_size = l.batch*l.c*l.w*l.h;
		const size_t delta16_size = l.batch*l.n*l.out_w*l.out_h;

		float *input16;
		if (state.input_is_bf16 && !fp8_layer_enabled)
		{
			input16 = state.input; // zero-copy: already BF16 from prev layer's output_gpu16
		}
		else
		{
			reallocate_lowp_tensor(state.net.input16_gpu, state.net.max_input16_size, input16_size, bf16_compute_mode);
			input16 = *state.net.input16_gpu;
		}

		reallocate_lowp_tensor(state.net.output16_gpu, state.net.max_output16_size, delta16_size, bf16_compute_mode);
		float *delta16 = *state.net.output16_gpu;

		assert(input16_size > 0);
		assert(delta16_size > 0);
		const int fp8_scale_update_interval = std::max(1, state.net.fp8_scale_update_interval);
		const int64_t lowp_step = static_cast<int64_t>(get_current_iteration(state.net)) * std::max(1, state.net.subdivisions) + std::max(0, state.net.current_subdivision);
		const int fp8_update_scale_now = ((lowp_step % fp8_scale_update_interval) == 0 ? 1 : 0);
#if defined(DARKNET_GPU_CUDA) && (CUDART_VERSION >= 12000)
		if (fp8_layer_enabled)
		{
			if (state.input_is_bf16)
			{
				cuda_quantize_bf16_to_fp8_bf16_by_policy(
					state.input, input16_size,
					(uint8_t *)l.act_fp8_gpu,
					l.x_scale_gpu,
					l.x_amax_ema_gpu,
					0 /* E4M3 */,
					state.net.fp8_current_scaling,
					fp8_update_scale_now,
					input16);
			}
			else
			{
				cuda_quantize_f32_to_fp8_bf16_by_policy(
					state.input, input16_size,
					(uint8_t *)l.act_fp8_gpu,
					l.x_scale_gpu,
					l.x_amax_ema_gpu,
					0 /* E4M3 */,
					state.net.fp8_current_scaling,
					fp8_update_scale_now,
					input16);
			}
			cuda_quantize_f32_to_fp8_bf16_by_policy(
				l.delta_gpu, delta16_size,
				NULL,
				l.grad_scale_gpu,
				l.grad_amax_ema_gpu,
				1 /* E5M2 */,
				state.net.fp8_current_scaling,
				fp8_update_scale_now,
				delta16);
		}
		else
#endif
		{
			if (!state.input_is_bf16 || fp8_layer_enabled)
				cuda_convert_f32_to_lowp(state.input, input16_size, input16, bf16_compute_mode);
			cuda_convert_f32_to_lowp(l.delta_gpu, delta16_size, delta16, bf16_compute_mode);
		}

		if (l.batch_normalize)
		{
			float one = 1.0f;
			float zero = 0.0f;
			CHECK_CUDNN(cudnnBatchNormalizationBackward(cudnn_handle(),
				CUDNN_BATCHNORM_SPATIAL,
				&one,
				&zero,
				&one,
				&one,
				l.normDstTensorDescF16,
				l.x_gpu,                // input (input in BN-forward-inference)
				l.normDstTensorDescF16,
				delta16,                // input
				l.normDstTensorDescF16,
				l.output_gpu, //l.x_norm_gpu,            // output (new delta)
				l.normTensorDesc,
				l.scales_gpu,            // input (should be FP32)
				l.scale_updates_gpu,    // output (should be FP32)
				l.bias_updates_gpu,        // output (should be FP32)
				.00001,
				l.mean_gpu,                // input (should be FP32)
				l.variance_gpu));        // input (should be FP32)

			simple_copy_ongpu(l.outputs*l.batch / 2, l.output_gpu, delta16);
		}

		// convert input: state.input (x), l.delta_gpu (y) from fp32 to fp16
		// get output: l.weight_updates_gpu (dw) and convert it to fp32 (ONLY if it is fp16)

		// calculate conv weight updates
		// Already: l.weight_updates_gpu = (l.weight_updates_gpu - l.weight*decay*batch*subdivision)*momentum
		//   so we should copy f32 to f16, or compute: f16=(w_up - w*d*b*s)*m
		assert((l.nweights) > 0);
		cuda_convert_f32_to_lowp(l.weight_updates_gpu, l.nweights, l.weight_updates_gpu16, bf16_compute_mode);

		float one = 1.0f;
		if (!state.net.adversarial && !l.train_only_bn)
		{
			CHECK_CUDNN(cudnnConvolutionBackwardFilter(cudnn_handle(),
				&one,
				l.srcTensorDesc16,
				input16, //state.input,
				l.ddstTensorDesc16,
				delta16, //l.delta_gpu,
				l.convDesc,
				l.bf_algo16,
				state.workspace,
				l.workspace_size,
				&one,
				l.dweightDesc16,
				l.weight_updates_gpu16));    // l.weight_updates_gpu);

			cuda_convert_lowp_to_f32(l.weight_updates_gpu16, l.nweights, l.weight_updates_gpu, bf16_compute_mode);
		}

		if (state.delta)
		{
			if (l.binary || l.xnor) swap_binary(&l);

			// http://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionBackwardData
			// calculate delta for the next layer
			// convert input: l.weights_gpu (w), l.delta_gpu (dy) from fp32 to fp16
			// get output: state.delta (dx) and convert it to fp32 (ONLY if it is fp16)
			CHECK_CUDNN(cudnnConvolutionBackwardData(cudnn_handle(),
				&alpha,
				l.weightDesc16,
				get_lowp_convolution_weights_gpu16(l, fp8_layer_enabled), //l.weights_gpu,
				l.ddstTensorDesc16,
				delta16, //l.delta_gpu,
				l.convDesc,
				l.bd_algo16,
				state.workspace,
				l.workspace_size,
				&beta,
				l.dsrcTensorDesc16,
				input16));    // state.delta);

			cuda_convert_lowp_to_f32(input16, input16_size, state.delta, bf16_compute_mode);

			if (l.binary || l.xnor) swap_binary(&l);
			if (l.xnor) gradient_array_ongpu(original_input, l.batch*l.c*l.h*l.w, HARDTAN, state.delta);
		}
	}
	else
	{
		//#else    // CUDNN_HALF

		if(l.batch_normalize){
			backward_batchnorm_layer_gpu(l, state);
		}

		if (!state.net.adversarial && !l.train_only_bn)
		{
			float *old_input = state.input;

			// calculate conv weight updates
			// if used: beta=1 then loss decreases faster
			float one = 1.0f;
			CHECK_CUDNN(cudnnConvolutionBackwardFilter(cudnn_handle(),
				&one,
				l.srcTensorDesc,
				state.input,
				l.ddstTensorDesc,
				l.delta_gpu,
				l.convDesc,
				l.bf_algo,
				state.workspace,
				l.workspace_size,
				&one,
				l.dweightDesc,
				l.weight_updates_gpu));

			state.input = old_input;
		}

		if (state.delta)
		{
			if (l.binary || l.xnor) swap_binary(&l);

			float *old_weights = l.weights_gpu;

			// http://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionBackwardData
			// calculate delta for the next layer
			float one = 1.0f;
			CHECK_CUDNN(cudnnConvolutionBackwardData(cudnn_handle(),
				&one,
				l.weightDesc,
				l.weights_gpu,
				l.ddstTensorDesc,
				l.delta_gpu,
				l.convDesc,
				l.bd_algo,
				state.workspace,
				l.workspace_size,
				&one,
				l.dsrcTensorDesc,
				state.delta));

			l.weights_gpu = old_weights;

			if (l.binary || l.xnor) swap_binary(&l);
			if (l.xnor) gradient_array_ongpu(original_input, l.batch*l.c*l.h*l.w, HARDTAN, state.delta);
		}
	}

//#endif    // CUDNN_HALF

#else    // CUDNN
	if (l.batch_normalize)
	{
		backward_batchnorm_layer_gpu(l, state);
	}

	int m = l.n / l.groups;
	int n = l.size*l.size*l.c / l.groups;
	int k = l.out_w*l.out_h;

	int i, j;
	for(i = 0; i < l.batch; ++i)
	{
		for (j = 0; j < l.groups; ++j)
		{
			float * a = l.delta_gpu + (i*l.groups + j)*m*k;
			float * b = state.workspace;
			float * c = l.weight_updates_gpu + j*l.nweights / l.groups;

			float *im = state.input + (i*l.groups + j)*l.c / l.groups*l.h*l.w;

			if (!state.net.adversarial && !l.train_only_bn)
			{
				//im2col_ongpu(im, l.c / l.groups, l.h, l.w, l.size, l.stride, l.pad, state.workspace);
				im2col_gpu_ext(im,          // input
					l.c / l.groups,         // input channels
					l.h, l.w,               // input size (h, w)
					l.size, l.size,         // kernel size (h, w)
					l.pad * l.dilation, l.pad * l.dilation,   // padding (h, w)
					l.stride_y, l.stride_x,     // stride (h, w)
					l.dilation, l.dilation, // dilation (h, w)
					state.workspace);       // output
				//gemm_ongpu(0, 1, m, n, k, 1, a + i*m*k, k, b, k, 1, c, n);
				gemm_ongpu(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);
			}

			if (state.delta)
			{
				if (l.binary || l.xnor) swap_binary(&l);
				float * a = l.weights_gpu + j*l.nweights / l.groups;
				float * b = l.delta_gpu + (i*l.groups + j)*m*k;
				float * c = state.workspace;

				//gemm_ongpu(1, 0, n, k, m, 1, a, n, b + i*k*m, k, 0, c, k);
				gemm_ongpu(1, 0, n, k, m, 1, a, n, b, k, 0, c, k);

				float *delta = state.delta + (i*l.groups + j)*l.c / l.groups*l.h*l.w;

				//col2im_ongpu(state.workspace, l.c / l.groups, l.h, l.w, l.size, l.stride, l.pad, delta);
				col2im_gpu_ext(
					state.workspace,        // input
					l.c / l.groups,         // input channels
					l.h, l.w,               // input size (h, w)
					l.size, l.size,         // kernel size (h, w)
					l.pad * l.dilation, l.pad * l.dilation,   // padding size (h, w)
					l.stride_y, l.stride_x,     // stride size (h, w)
					l.dilation, l.dilation, // dilation size (h, w)
					delta);                 // output (delta)

				if (l.binary || l.xnor)
				{
					swap_binary(&l);
				}
				if (l.xnor)
				{
					gradient_array_ongpu(original_input + i*l.c*l.h*l.w, l.c*l.h*l.w, HARDTAN, state.delta + i*l.c*l.h*l.w);
				}
			}
		}
	}
#endif
	if (state.net.try_fix_nan)
	{
		if (state.delta)
		{
			reset_nan_and_inf(state.delta, l.inputs * l.batch);
		}
		int size = l.nweights;
		reset_nan_and_inf(l.weight_updates_gpu, size);
		if (!use_bf16_master_weight_storage(l) && l.weights_gpu) fix_nan_and_inf(l.weights_gpu, size);
	}


}

__global__ void calc_avg_activation_kernel(float *src, float *dst, int size, int channels, int batches)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int xy = i % size;
	int b = i / size;

	if (i < size*batches) {
		dst[i] = 0;
		for (int c = 0; c < channels; ++c) {
			dst[i] += src[xy + size*(c + channels*b)];
		}
		dst[i] = dst[i] / channels;
	}
}

void calc_avg_activation_gpu(float *src, float *dst, int size, int channels, int batches)
{
	TAT(TATPARMS);

	const int num_blocks = get_number_of_blocks(size*batches, BLOCK);

	calc_avg_activation_kernel <<<num_blocks, BLOCK, 0, get_cuda_stream() >>> (src, dst, size, channels, batches);
}


__global__ void assisted_activation_kernel(float alpha, float *output, float *gt_gpu, float *a_avg_gpu, int size, int channels, int batches)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int xy = i % size;
	int b = i / size;

	if (b < batches)
	{
		for (int c = 0; c < channels; ++c)
		{
			output[xy + size*(c + channels*b)] += alpha * gt_gpu[i] * a_avg_gpu[i];
		}
	}
}

void assisted_activation_gpu(float alpha, float *output, float *gt_gpu, float *a_avg_gpu, int size, int channels, int batches)
{
	TAT(TATPARMS);

	const int num_blocks = get_number_of_blocks(size*batches, BLOCK);

	assisted_activation_kernel <<<num_blocks, BLOCK, 0, get_cuda_stream() >>> (alpha, output, gt_gpu, a_avg_gpu, size, channels, batches);
}


__global__ void assisted_activation2_kernel(float alpha, float *output, float *gt_gpu, float *a_avg_gpu, int size, int channels, int batches)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int xy = i % size;
	int b = i / size;
	float beta = 1 - alpha;

	if (b < batches) {
		for (int c = 0; c < channels; ++c) {
			if(gt_gpu[i] == 0)
				output[xy + size*(c + channels*b)] *= beta;

		}
	}
}

void assisted_activation2_gpu(float alpha, float *output, float *gt_gpu, float *a_avg_gpu, int size, int channels, int batches)
{
	TAT(TATPARMS);

	const int num_blocks = get_number_of_blocks(size*batches, BLOCK);

	assisted_activation2_kernel <<<num_blocks, BLOCK, 0, get_cuda_stream() >>> (alpha, output, gt_gpu, a_avg_gpu, size, channels, batches);
}

void assisted_excitation_forward_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	const int iteration_num = get_current_iteration(state.net); //(*state.net.seen) / (state.net.batch*state.net.subdivisions);

	float alpha = (1 + cos(3.141592 * iteration_num / state.net.max_batches)) / 2;

	if (l.assisted_excitation == 1)
	{
		if (iteration_num > state.net.max_batches / 2) return;
	}
	else
	{
		if (iteration_num < state.net.burn_in) return;
		else
			if (iteration_num > l.assisted_excitation) return;
		else
			alpha = (1 + cos(3.141592 * iteration_num / (state.net.burn_in + l.assisted_excitation))) / 2; // from 1 to 0
	}

	float *a_avg = (float *)calloc(l.out_w * l.out_h * l.batch, sizeof(float));
	float *gt = (float *)calloc(l.out_w * l.out_h * l.batch, sizeof(float));

	int b;
	int w, h;

	l.max_boxes = state.net.num_boxes;
	l.truths = l.max_boxes*(4 + 1);

	int num_truth = l.batch*l.truths;
	float *truth_cpu = (float *)calloc(num_truth, sizeof(float));
	cuda_pull_array(state.truth, truth_cpu, num_truth);

	for (b = 0; b < l.batch; ++b)
	{
		// calculate G
		int t;
		for (t = 0; t < state.net.num_boxes; ++t) {
			Darknet::Box truth = float_to_box_stride(truth_cpu + t*(4 + 1) + b*l.truths, 1);
			if (!truth.x) break;  // continue;
			float beta = 0;
			//float beta = 1 - alpha; // from 0 to 1
			float dw = (1 - truth.w) * beta;
			float dh = (1 - truth.h) * beta;

			int left = floorf((truth.x - (dw + truth.w) / 2) * l.out_w);
			int right = ceilf((truth.x + (dw + truth.w) / 2) * l.out_w);
			int top = floorf((truth.y - (dh + truth.h) / 2) * l.out_h);
			int bottom = ceilf((truth.y + (dh + truth.h) / 2) * l.out_h);
			if (left < 0) left = 0;
			if (top < 0) top = 0;
			if (right > l.out_w) right = l.out_w;
			if (bottom > l.out_h) bottom = l.out_h;

			for (w = left; w <= right; w++) {
				for (h = top; h < bottom; h++) {
					gt[w + l.out_w * h + l.out_w*l.out_h*b] = 1;
				}
			}
		}
	}

	cuda_push_array(l.gt_gpu, gt, l.out_w * l.out_h * l.batch);

	// calc avg_output on GPU - for whole batch
	calc_avg_activation_gpu(l.output_gpu, l.a_avg_gpu, l.out_w * l.out_h, l.out_c, l.batch);

	// calc new output
	assisted_activation_gpu(alpha, l.output_gpu, l.gt_gpu, l.a_avg_gpu, l.out_w * l.out_h, l.out_c, l.batch);

	if (0)   // visualize ground truth
	{
		cuda_pull_array(l.output_gpu, l.output, l.outputs * l.batch);
		CHECK_CUDA(cudaStreamSynchronize(get_cuda_stream()));

		for (b = 0; b < l.batch; ++b)
		{
			*cfg_and_state.output << "Assisted Excitation alpha = " << alpha << std::endl;
			Darknet::Image img = Darknet::float_to_image(l.out_w, l.out_h, 1, &gt[l.out_w*l.out_h*b]);
			char buff[100];
			sprintf(buff, "a_excitation_gt_%d", b);
			show_image_cv(img, buff);

			//image img2 = float_to_image(l.out_w, l.out_h, 1, &l.output[l.out_w*l.out_h*l.out_c*b]);
			Darknet::Image img2 = Darknet::float_to_image_scaled(l.out_w, l.out_h, 1, &l.output[l.out_w*l.out_h*l.out_c*b]);
			char buff2[100];
			sprintf(buff2, "a_excitation_output_%d", b);
			show_image_cv(img2, buff2);

			cv::waitKey(5);
		}
		cv::waitKey(0);
	}

	free(truth_cpu);
	free(gt);
	free(a_avg);
}

void pull_convolutional_layer(Darknet::Layer & l)
{
	TAT(TATPARMS);

	ScopedConvolutionalWeightsF32Scratch weights_f32_scratch(l, use_bf16_master_weight_storage(l));
	cuda_pull_array_async(l.weights_gpu, l.weights, l.nweights);
	cuda_pull_array_async(l.biases_gpu, l.biases, l.n);
	if (l.weight_updates_gpu) cuda_pull_array_async(l.weight_updates_gpu, l.weight_updates, l.nweights);
	if (l.bias_updates_gpu) cuda_pull_array_async(l.bias_updates_gpu, l.bias_updates, l.n);
	if (l.batch_normalize){
		cuda_pull_array_async(l.scales_gpu, l.scales, l.n);
		cuda_pull_array_async(l.rolling_mean_gpu, l.rolling_mean, l.n);
		cuda_pull_array_async(l.rolling_variance_gpu, l.rolling_variance, l.n);
	}
	if (l.adam){
		cuda_pull_array_async(l.m_gpu, l.m, l.nweights);
		cuda_pull_array_async(l.v_gpu, l.v, l.nweights);
	}
	CHECK_CUDA(cudaPeekAtLastError());
	CHECK_CUDA(cudaStreamSynchronize(get_cuda_stream()));
}

void push_convolutional_layer(Darknet::Layer & l)
{
	TAT(TATPARMS);

	if (use_bf16_master_weight_storage(l))
	{
		float *weights_gpu_tmp = cuda_make_array(l.weights, l.nweights);
		cuda_convert_f32_to_lowp(weights_gpu_tmp, l.nweights, l.weights_gpu16, true);
#if defined(DARKNET_GPU_CUDA) && (CUDART_VERSION >= 12000)
		if (cfg_and_state.use_cudnn_fp8 && l.fp8_enabled > 0 && l.weights_conv_gpu16)
		{
			cuda_quantize_bf16_to_fp8_bf16_by_policy(
				l.weights_gpu16, l.nweights,
				l.weights_fp8_gpu,
				l.w_scale_gpu,
				l.w_amax_ema_gpu,
				l.fp8_format,
				1,
				1,
				l.weights_conv_gpu16);
		}
#endif
		cuda_free(weights_gpu_tmp);
	}
	else
	{
		cuda_push_array(l.weights_gpu, l.weights, l.nweights);
#ifdef CUDNN_HALF
		assert(l.nweights > 0);
#if defined(DARKNET_GPU_CUDA) && (CUDART_VERSION >= 12000)
		if (cfg_and_state.use_cudnn_fp8 && l.fp8_enabled > 0)
		{
			float *dst16 = (l.weights_conv_gpu16 ? l.weights_conv_gpu16 : l.weights_gpu16);
			cuda_quantize_f32_to_fp8_bf16_by_policy(l.weights_gpu, l.nweights, l.weights_fp8_gpu, l.w_scale_gpu, l.w_amax_ema_gpu, l.fp8_format, 1, 1, dst16);
		}
		else
#endif
		{
			const bool is_bf16 = (cfg_and_state.use_cudnn_bf16 || cfg_and_state.use_cudnn_fp8);
			cuda_convert_f32_to_lowp(l.weights_gpu, l.nweights, l.weights_gpu16, is_bf16);
		}
#endif
	}
#if defined(DARKNET_GPU_CUDA) && (CUDNN_MAJOR >= 8) && (CUDART_VERSION >= 11000)
	if (l.weight_compensation_gpu)
	{
		CHECK_CUDA(cudaMemsetAsync(l.weight_compensation_gpu, 0, sizeof(float) * (l.nweights / 2 + 1), get_cuda_stream()));
	}
#endif
	cuda_push_array(l.biases_gpu, l.biases, l.n);
	if (l.train) {
		cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
		cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.n);
	}
	if (l.batch_normalize){
		cuda_push_array(l.scales_gpu, l.scales, l.n);
		cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
		cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
	}
	if (l.adam){
		cuda_push_array(l.m_gpu, l.m, l.nweights);
		cuda_push_array(l.v_gpu, l.v, l.nweights);
	}
	CHECK_CUDA(cudaPeekAtLastError());
}

void update_convolutional_layer_gpu(Darknet::Layer & l, int batch, float learning_rate_init, float momentum, float decay, float loss_scale)
{
	TAT(TATPARMS);

	if (l.deform)
	{
		if (l.rotate) rotate_weights_gpu(l.weight_updates_gpu, l.weight_deform_gpu, l.nweights, l.n, l.size, 1);
		else if (l.sway) sway_and_flip_weights_gpu(l.weight_updates_gpu, l.weight_deform_gpu, l.nweights, l.n, l.size, l.angle, 1);
		else if (l.stretch) stretch_weights_gpu(l.weight_updates_gpu, l.weight_deform_gpu, l.nweights, l.n, l.size, 0, 1);
		else if (l.stretch_sway) stretch_sway_flip_weights_gpu(l.weight_updates_gpu, l.weight_deform_gpu, l.nweights, l.n, l.size, l.angle, 1);

		reduce_and_expand_array_gpu(l.weight_deform_gpu, l.weight_updates_gpu, l.nweights, 4);
	}

	// Loss scale for Mixed-Precision on Tensor-Cores
	float learning_rate = learning_rate_init*l.learning_rate_scale / loss_scale;
	const bool bf16_master_weights = use_bf16_master_weight_storage(l);
	const bool needs_f32_weight_ops = (l.adam || l.deform || l.clip || l.reverse || !l.weight_compensation_gpu);
	ScopedConvolutionalWeightsF32Scratch weights_f32_scratch(l, needs_f32_weight_ops);

	reset_nan_and_inf(l.weight_updates_gpu, l.nweights);
	if (l.weights_gpu) fix_nan_and_inf(l.weights_gpu, l.nweights);

	// Gradient Centralization
	if (l.grad_centr && l.batch_normalize)
	{
		gradient_centralization_gpu(l.size, l.size, l.c / l.groups, l.n, l.weight_updates_gpu);
	}

	if (l.adam)
	{
		adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, l.B1, l.B2, l.eps, decay, learning_rate, l.nweights, batch, l.t);

		adam_update_gpu(l.biases_gpu, l.bias_updates_gpu, l.bias_m_gpu, l.bias_v_gpu, l.B1, l.B2, l.eps, decay, learning_rate, l.n, batch, l.t);
		if (l.scales_gpu)
		{
			adam_update_gpu(l.scales_gpu, l.scale_updates_gpu, l.scale_m_gpu, l.scale_v_gpu, l.B1, l.B2, l.eps, decay, learning_rate, l.n, batch, l.t);
		}
	}
	else
	{
		float *old_weight_updates_gpu = l.weight_updates_gpu;

		if (l.reverse)
		{
			float clip = 0.0;
			float divider = 1.0;
			float abs_add = 1.0;
			mult_inverse_array_gpu(l.weight_updates_gpu, l.output_gpu, l.inputs*l.batch, l.reverse, divider, clip, abs_add);
			l.weight_updates_gpu = l.output_gpu;
		}

#if defined(DARKNET_GPU_CUDA) && (CUDNN_MAJOR >= 8) && (CUDART_VERSION >= 11000)
		if (bf16_master_weights && l.weight_compensation_gpu && !l.reverse && !weights_f32_scratch.active())
		{
			cuda_bf16_kahan_sgd(l.nweights, learning_rate / batch, -decay * batch * loss_scale, momentum,
				l.weight_updates_gpu, l.weights_gpu16, l.weight_compensation_gpu, nullptr);
		}
		else
#endif
		{
			axpy_ongpu(l.nweights, -decay*batch*loss_scale, l.weights_gpu, 1, l.weight_updates_gpu, 1);
			axpy_ongpu(l.nweights, learning_rate / batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);

			l.weight_updates_gpu = old_weight_updates_gpu;

			scal_ongpu(l.nweights, momentum, l.weight_updates_gpu, 1);
		}

		axpy_ongpu(l.n, learning_rate / batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
		scal_ongpu(l.n, momentum, l.bias_updates_gpu, 1);

		if (l.scales_gpu) {
			axpy_ongpu(l.n, learning_rate / batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
			scal_ongpu(l.n, momentum, l.scale_updates_gpu, 1);
		}
	}

	if (l.deform)
	{
		expand_array_gpu(l.weights_gpu, l.weight_deform_gpu, l.nweights, 4);

		if (l.rotate) rotate_weights_gpu(l.weight_deform_gpu, l.weights_gpu, l.nweights, l.n, l.size, 0);
		else if (l.sway) sway_and_flip_weights_gpu(l.weight_deform_gpu, l.weights_gpu, l.nweights, l.n, l.size, l.angle, 0);
		else if (l.stretch) stretch_weights_gpu(l.weight_deform_gpu, l.weights_gpu, l.nweights, l.n, l.size, 0, 0);
		else if (l.stretch_sway) stretch_sway_flip_weights_gpu(l.weight_deform_gpu, l.weights_gpu, l.nweights, l.n, l.size, l.angle, 0);
	}

	if (l.clip)
	{
		constrain_ongpu(l.nweights, l.clip, l.weights_gpu, 1);
	}

	// BF16 master weights: commit the FP32 mirror only when the fused Kahan path
	// did not already write the authoritative BF16 weights.
#if defined(DARKNET_GPU_CUDA) && (CUDNN_MAJOR >= 8) && (CUDART_VERSION >= 11000)
	if (bf16_master_weights && l.weights_gpu16 && weights_f32_scratch.active())
	{
		if (l.weight_compensation_gpu)	cuda_bf16_kahan_commit(l.nweights, l.weights_gpu, l.weights_gpu16, l.weight_compensation_gpu);
		else							cuda_convert_f32_to_lowp(l.weights_gpu, l.nweights, l.weights_gpu16, true);
	}
#endif

#if defined(DARKNET_GPU_CUDA) && (CUDART_VERSION >= 12000)
	if (l.fp8_enabled > 0 && (l.weights_conv_gpu16 || l.weights_gpu16))
	{
		float *dst16 = (l.weights_conv_gpu16 ? l.weights_conv_gpu16 : l.weights_gpu16);
		cuda_quantize_f32_to_fp8_bf16_by_policy(l.weights_gpu, l.nweights, l.weights_fp8_gpu, l.w_scale_gpu, l.w_amax_ema_gpu, l.fp8_format, 1, 1, dst16);
	}
#endif
}
