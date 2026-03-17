#include "darknet_internal.hpp"


namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();

	static bool layer_can_use_lowp_conv(const Darknet::Network & net, const Darknet::Layer & l, const bool train)
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

	static bool layer_can_consume_cached_bf16_input(const Darknet::Network & net, const Darknet::Layer & l, const bool train)
	{
		return (layer_can_use_lowp_conv(net, l, train) && !(net.cudnn_fp8 && l.fp8_enabled > 0));
	}

	static void set_network_state_input_for_next_layer(const Darknet::Network & net, const int index, const bool train, float * & input, bool & input_is_bf16)
	{
		const Darknet::Layer & current = net.layers[index];
		if (index + 1 < net.n)
		{
			const Darknet::Layer & next = net.layers[index + 1];
			if (current.output_gpu16 && layer_can_use_lowp_conv(net, current, train) && layer_can_consume_cached_bf16_input(net, next, train))
			{
				input = current.output_gpu16;
				input_is_bf16 = true;
				return;
			}
		}

		input = current.output_gpu;
		input_is_bf16 = false;
	}
}


typedef struct time_benchmark_layers
{
	float time;
	int layer_id;
	Darknet::ELayerType layer_type;
} time_benchmark_layers;


int time_comparator(const void *pa, const void *pb)
{
	TAT(TATPARMS);

	time_benchmark_layers a = *(time_benchmark_layers *)pa;
	time_benchmark_layers b = *(time_benchmark_layers *)pb;
	float diff = a.time - b.time;
	if (diff < 0) return 1;
	else if (diff > 0) return -1;
	return 0;
}

void forward_network_gpu(Darknet::Network & net, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	static time_benchmark_layers *avg_time_per_layer = NULL;
	static time_benchmark_layers *sorted_avg_time_per_layer = NULL;
	if (net.benchmark_layers)
	{
		if (!avg_time_per_layer)
		{
			avg_time_per_layer = (time_benchmark_layers *)calloc(net.n, sizeof(time_benchmark_layers));
			sorted_avg_time_per_layer = (time_benchmark_layers *)calloc(net.n, sizeof(time_benchmark_layers));
		}
		/// @todo in previous versions we did not CHECK_CUDA here -- was that intentional?
		CHECK_CUDA(cudaDeviceSynchronize()); // was this removed in CUDA 11.6+?
	}

	state.workspace = net.workspace;
	for (int i = 0; i < net.n; ++i)
	{
		state.index = i;
		Darknet::Layer & l = net.layers[i];

		if (l.delta_gpu && state.train)
		{
			fill_ongpu(l.outputs * l.batch, 0, l.delta_gpu, 1);
		}

		l.forward_gpu(l, state);

		set_network_state_input_for_next_layer(net, i, state.train, state.input, state.input_is_bf16);
		if(net.wait_stream)
		{
			CHECK_CUDA(cudaStreamSynchronize(get_cuda_stream()));
		}
	}

	if (net.benchmark_layers)
	{
		*cfg_and_state.output << std::endl << std::endl << "Sorted by time (forward):" << std::endl;

		/// @todo replace qsort() low priority
		qsort(sorted_avg_time_per_layer, net.n, sizeof(time_benchmark_layers), time_comparator);

		for (int i = 0; i < net.n; ++i)
		{
			*cfg_and_state.output
				<< i
				<< " - fw-sort-layer " << sorted_avg_time_per_layer[i].layer_id
				<< " - type: " << static_cast<int>(sorted_avg_time_per_layer[i].layer_type)
				<< " - avg_time " << sorted_avg_time_per_layer[i].time << " ms"
				<< std::endl;
		}
	}
}

void backward_network_gpu(Darknet::Network & net, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	static time_benchmark_layers *avg_time_per_layer = NULL;
	static time_benchmark_layers *sorted_avg_time_per_layer = NULL;
	if (net.benchmark_layers)
	{
		if (!avg_time_per_layer)
		{
			avg_time_per_layer = (time_benchmark_layers *)calloc(net.n, sizeof(time_benchmark_layers));
			sorted_avg_time_per_layer = (time_benchmark_layers *)calloc(net.n, sizeof(time_benchmark_layers));
		}
		CHECK_CUDA(cudaDeviceSynchronize());
	}

	state.workspace = net.workspace;
	int i;
	float * original_input = state.input;
	float * original_delta = state.delta;
	for(i = net.n-1; i >= 0; --i)
	{
		state.index = i;
		Darknet::Layer & l = net.layers[i];
		if (l.stopbackward == 1)
		{
			break;
		}

		if (l.stopbackward > get_current_iteration(net))
		{
			break;
		}

		if (i == 0)
		{
			state.input = original_input;
			state.input_is_bf16 = false;
			state.delta = original_delta;
		}
		else
		{
			const Darknet::Layer & prev = net.layers[i-1];
			if (prev.output_gpu16 && layer_can_use_lowp_conv(net, prev, state.train) && layer_can_consume_cached_bf16_input(net, l, state.train))
			{
				state.input = prev.output_gpu16;
				state.input_is_bf16 = true;
			}
			else
			{
				state.input = prev.output_gpu;
				state.input_is_bf16 = false;
			}
			state.delta = prev.delta_gpu;
			if (net.optimized_memory && !prev.keep_delta_gpu)
			{
				state.delta = net.state_delta_gpu;
			}
		}

		if (l.onlyforward)
		{
			continue;
		}

		l.backward_gpu(l, state);

		if (i != 0)
		{
			Darknet::Layer & prev = net.layers[i - 1];
			if (net.optimized_memory && state.delta && !prev.keep_delta_gpu)
			{
				if (prev.delta_gpu != state.delta)
				{
					simple_copy_ongpu(prev.outputs*prev.batch, state.delta, prev.delta_gpu);
				}
				fill_ongpu(prev.outputs*prev.batch, 0, net.state_delta_gpu, 1);
			}
		}
	}

	if (net.adversarial && net.attention)
	{
		int img_size = net.w * net.h * net.c;
		float *original_input_cpu = (float *)xcalloc(img_size, sizeof(float));
		float *original_delta_cpu = (float *)xcalloc(img_size, sizeof(float));
		cuda_pull_array(original_input, original_input_cpu, img_size);
		cuda_pull_array(original_delta, original_delta_cpu, img_size);

		Darknet::Image attention_img = Darknet::make_attention_image(img_size, original_delta_cpu, original_input_cpu, net.w, net.h, net.c, 0.7);
		Darknet::show_image(attention_img, "attention_img");
		cv::resizeWindow("attention_img", 500, 500);

		Darknet::free_image(attention_img);

		Darknet::Image attention_mask_img = Darknet::make_attention_image(img_size, original_delta_cpu, original_delta_cpu, net.w, net.h, net.c, 1.0);
		Darknet::show_image(attention_mask_img, "attention_mask_img");
		cv::resizeWindow("attention_mask_img", 500, 500);

		Darknet::free_image(attention_mask_img);

		free(original_input_cpu);
		free(original_delta_cpu);
	}

	if (net.adversarial)
	{
		int x_size = get_network_input_size(net) * net.batch;
		*cfg_and_state.output
			<< "x_size=" << x_size
			<< ", original_delta=" << original_delta
			<< ", original_input=" << original_input
			<< ", net.learning_rate=" << net.learning_rate
			<< std::endl;
		axpy_ongpu(x_size, net.learning_rate, original_delta, 1, original_input, 1);
		constrain_min_max_ongpu(x_size, 0, 1, original_input, 1);
	}

	if (net.benchmark_layers)
	{
		*cfg_and_state.output << std::endl << std::endl << "Sorted by time (backward):" << std::endl;

		/// @todo replace qsort() unknown priority
		qsort(sorted_avg_time_per_layer, net.n, sizeof(time_benchmark_layers), time_comparator);

		for (i = 0; i < net.n; ++i)
		{
			*cfg_and_state.output
				<< i
				<< " - bw-sort-layer " << sorted_avg_time_per_layer[i].layer_id
				<< " - type: " << static_cast<int>(sorted_avg_time_per_layer[i].layer_type)
				<< " - avg_time " << sorted_avg_time_per_layer[i].time << " ms"
				<< std::endl;
		}
	}
}

void update_network_gpu(Darknet::Network & net)
{
	TAT(TATPARMS);

	cuda_set_device(net.gpu_index);
	const int iteration_num = (*net.seen) / (net.batch * net.subdivisions);

	int update_batch = net.batch*net.subdivisions * get_sequence_value(net);

	float rate = get_current_rate(net);
	for (int i = 0; i < net.n; ++i)
	{
		Darknet::Layer & l = net.layers[i];
		if (l.train == 0)
		{
			continue;
		}
		l.t = get_current_batch(net);
		if (iteration_num > (net.max_batches * 1 / 2))
		{
			l.deform = 0;
		}
		if (l.burnin_update && (l.burnin_update*net.burn_in > iteration_num))
		{
			continue;
		}
		if (l.train_only_bn)
		{
			continue;
		}

		if (l.update_gpu && l.dont_update < iteration_num)
		{
			l.update_gpu(l, update_batch, rate, net.momentum, net.decay, net.loss_scale);
		}
	}
}

void forward_backward_network_gpu(Darknet::Network & net, float *x, float *y)
{
	TAT(TATPARMS);

	Darknet::NetworkState state;
	state.index = 0;
	state.net = net;
	int x_size = get_network_input_size(net)*net.batch;
	int y_size = get_network_output_size(net)*net.batch;
	if (net.layers[net.n-1].truths)
	{
		y_size = net.layers[net.n-1].truths*net.batch;
	}
	if (!*net.input_gpu)
	{
		*net.input_gpu = cuda_make_array(x, x_size);
		*net.truth_gpu = cuda_make_array(y, y_size);
	}
	else
	{
		cuda_push_array(*net.input_gpu, x, x_size);
		cuda_push_array(*net.truth_gpu, y, y_size);
	}
	state.input = *net.input_gpu;
	state.delta = 0;
	if (net.adversarial)
	{
		state.delta = cuda_make_array(NULL, x_size);
	}
	state.truth = *net.truth_gpu;
	state.train = 1;
#if defined(CUDNN_HALF) && defined(CUDNN)
	const bool use_lowp_weights = (net.cudnn_bfloat16 || net.cudnn_fp8 ||
		net.precision_mode == Darknet::PrecisionMode::BF16 ||
		net.precision_mode == Darknet::PrecisionMode::BF16_MASTER_KAHAN ||
		net.precision_mode == Darknet::PrecisionMode::FP8_BF16);
	const int current_iteration = get_current_iteration(net);
	const int fp8_requant_interval = std::max(1, net.fp8_requant_interval);
	const int fp8_scale_update_interval = std::max(1, net.fp8_scale_update_interval);
	const int64_t lowp_step = static_cast<int64_t>(current_iteration) * std::max(1, net.subdivisions) + std::max(0, net.current_subdivision);
#if defined(DARKNET_GPU_CUDA) && (CUDNN_MAJOR >= 9) && (CUDART_VERSION >= 12000)
	const int fp8_update_scale_now = ((lowp_step % fp8_scale_update_interval) == 0 ? 1 : 0);
	auto refresh_fp8_if_needed = [&](Darknet::Layer & wl)
	{
		const bool tc_aligned = ((wl.c / std::max(1, wl.groups)) % 8 == 0 && wl.n % 8 == 0 && wl.groups <= 1);
		if (!(net.cudnn_fp8 && wl.fp8_enabled > 0 && tc_aligned))
		{
			return false;
		}

		float *weights_bf16_dst = (net.bf16_master_weights && wl.weights_conv_gpu16) ? wl.weights_conv_gpu16 : wl.weights_gpu16;
		if (!weights_bf16_dst)
		{
			return false;
		}

		// Weights are updated every training iteration. Keep cache only across subdivisions
		// inside the same iteration by default. fp8_requant_interval can trade accuracy
		// for lower quantization overhead by letting the staged weights lag a few steps.
		if (wl.fp8_weight_cache_iteration < 0 ||
			(current_iteration - wl.fp8_weight_cache_iteration) >= fp8_requant_interval)
		{
			if (net.bf16_master_weights)
			{
				cuda_quantize_bf16_to_fp8_bf16_by_policy(
					wl.weights_gpu16, wl.nweights,
					wl.weights_fp8_gpu,
					wl.w_scale_gpu,
					wl.w_amax_ema_gpu,
					wl.fp8_format,
					net.fp8_current_scaling,
					fp8_update_scale_now,
					weights_bf16_dst);
			}
			else
			{
				cuda_quantize_f32_to_fp8_bf16_by_policy(
					wl.weights_gpu, wl.nweights,
					wl.weights_fp8_gpu,
					wl.w_scale_gpu,
					wl.w_amax_ema_gpu,
					wl.fp8_format,
					net.fp8_current_scaling,
					fp8_update_scale_now,
					weights_bf16_dst);
			}
			wl.fp8_weight_cache_iteration = current_iteration;
		}

		return true;
	};
#endif
	int i;
	for (i = 0; i < net.n; ++i)
	{
		Darknet::Layer & l = net.layers[i];
		if (use_lowp_weights)
		{
			if (l.type == Darknet::ELayerType::CONVOLUTIONAL && l.weights_gpu16 && (net.bf16_master_weights || l.weights_gpu))
			{
				assert((l.nweights) > 0);
				// BF16 master: weights_gpu16 is already up-to-date (committed after each optimizer step); skip re-derivation.
#if defined(DARKNET_GPU_CUDA) && (CUDNN_MAJOR >= 9) && (CUDART_VERSION >= 12000)
				if (refresh_fp8_if_needed(l))
				{
					// FP8 path handled.
				}
				else if (net.bf16_master_weights)
				{
					// nothing to do — weights_gpu16 is the authoritative master
				}
				else
				{
					if (net.cudnn_bfloat16 || net.cudnn_fp8)	cuda_convert_f32_to_bf16(l.weights_gpu, l.nweights, l.weights_gpu16);
					else							cuda_convert_f32_to_f16(l.weights_gpu, l.nweights, l.weights_gpu16);
				}
#elif defined(DARKNET_GPU_CUDA) && (CUDNN_MAJOR >= 8) && (CUDART_VERSION >= 11000)
				if (net.bf16_master_weights)
				{
					// nothing to do — weights_gpu16 is the authoritative master
				}
				else
				{
				if (net.cudnn_bfloat16)		cuda_convert_f32_to_bf16(l.weights_gpu, l.nweights, l.weights_gpu16);
				else							cuda_convert_f32_to_f16(l.weights_gpu, l.nweights, l.weights_gpu16);
				}
#else
				if (!net.bf16_master_weights)
				{
					cuda_convert_f32_to_f16(l.weights_gpu, l.nweights, l.weights_gpu16);
				}
#endif
			}
			else if (l.type == Darknet::ELayerType::CRNN &&
				l.input_layer->weights_gpu16 &&
				l.self_layer->weights_gpu16 &&
				l.output_layer->weights_gpu16 &&
				(net.bf16_master_weights || (l.input_layer->weights_gpu && l.self_layer->weights_gpu && l.output_layer->weights_gpu)))
			{
				assert((l.input_layer->c*l.input_layer->n*l.input_layer->size*l.input_layer->size) > 0);
				// BF16 master: sub-layer weights_gpu16 are already authoritative; skip re-derivation.
		#if defined(DARKNET_GPU_CUDA) && (CUDNN_MAJOR >= 9) && (CUDART_VERSION >= 12000)
					if (!refresh_fp8_if_needed(*l.input_layer) && !net.bf16_master_weights)
					{
						if (net.cudnn_bfloat16 || net.cudnn_fp8)
						{
							cuda_convert_f32_to_bf16(l.input_layer->weights_gpu, l.input_layer->nweights, l.input_layer->weights_gpu16);
						}
						else
						{
							cuda_convert_f32_to_f16(l.input_layer->weights_gpu, l.input_layer->nweights, l.input_layer->weights_gpu16);
						}
					}

					if (!refresh_fp8_if_needed(*l.self_layer) && !net.bf16_master_weights)
					{
						if (net.cudnn_bfloat16 || net.cudnn_fp8)
						{
							cuda_convert_f32_to_bf16(l.self_layer->weights_gpu, l.self_layer->nweights, l.self_layer->weights_gpu16);
						}
						else
						{
							cuda_convert_f32_to_f16(l.self_layer->weights_gpu, l.self_layer->nweights, l.self_layer->weights_gpu16);
						}
					}

					if (!refresh_fp8_if_needed(*l.output_layer) && !net.bf16_master_weights)
					{
						if (net.cudnn_bfloat16 || net.cudnn_fp8)
						{
							cuda_convert_f32_to_bf16(l.output_layer->weights_gpu, l.output_layer->nweights, l.output_layer->weights_gpu16);
						}
						else
						{
							cuda_convert_f32_to_f16(l.output_layer->weights_gpu, l.output_layer->nweights, l.output_layer->weights_gpu16);
						}
					}
#elif defined(DARKNET_GPU_CUDA) && (CUDNN_MAJOR >= 8) && (CUDART_VERSION >= 11000)
				if (!net.bf16_master_weights && net.cudnn_bfloat16)
				{
					cuda_convert_f32_to_bf16(l.input_layer->weights_gpu, l.input_layer->nweights, l.input_layer->weights_gpu16);
					cuda_convert_f32_to_bf16(l.self_layer->weights_gpu, l.self_layer->nweights, l.self_layer->weights_gpu16);
					cuda_convert_f32_to_bf16(l.output_layer->weights_gpu, l.output_layer->nweights, l.output_layer->weights_gpu16);
				}
				else if (!net.bf16_master_weights)
				{
					cuda_convert_f32_to_f16(l.input_layer->weights_gpu, l.input_layer->nweights, l.input_layer->weights_gpu16);
					cuda_convert_f32_to_f16(l.self_layer->weights_gpu, l.self_layer->nweights, l.self_layer->weights_gpu16);
					cuda_convert_f32_to_f16(l.output_layer->weights_gpu, l.output_layer->nweights, l.output_layer->weights_gpu16);
				}
#else
				if (!net.bf16_master_weights)
				{
					cuda_convert_f32_to_f16(l.input_layer->weights_gpu, l.input_layer->nweights, l.input_layer->weights_gpu16);
					cuda_convert_f32_to_f16(l.self_layer->weights_gpu, l.self_layer->nweights, l.self_layer->weights_gpu16);
					cuda_convert_f32_to_f16(l.output_layer->weights_gpu, l.output_layer->nweights, l.output_layer->weights_gpu16);
				}
#endif
			}
		}
	}
#endif
	forward_network_gpu(net, state);
	//cudaStreamSynchronize(get_cuda_stream());
	backward_network_gpu(net, state);

	if (net.adversarial)
	{
		cuda_free(state.delta);
		cuda_pull_array(*net.input_gpu, x, x_size);
	}
}

float train_network_datum_gpu(Darknet::Network & net, float *x, float *y)
{
	TAT(TATPARMS);

	*net.seen += net.batch;
	if (net.adversarial_lr && rand_bool() && get_current_iteration(net) > net.burn_in)
	{
		net.adversarial = 1;
		float lr_old = net.learning_rate;
		float scale = (get_current_iteration(net) / ((float)net.max_batches));
		//scale = sin(scale * M_PI);
		net.learning_rate = net.adversarial_lr * scale;
		int y_size = get_network_output_size(net)*net.batch;
		if (net.layers[net.n - 1].truths)
		{
			y_size = net.layers[net.n - 1].truths*net.batch;
		}
		float *truth_cpu = (float *)xcalloc(y_size, sizeof(float));

		const int img_size = net.w*net.h*net.c;
		float *old_input = (float *)xcalloc(img_size*net.batch, sizeof(float));
		memcpy(old_input, x, img_size*net.batch * sizeof(float));

		*cfg_and_state.output << std::endl << "adversarial training, adversarial_lr=" << net.adversarial_lr * scale << std::endl;

		forward_backward_network_gpu(net, x, truth_cpu);

		int b;
		for (b = 0; b < net.batch; ++b)
		{
			if (b % 2 == 1 && net.contrastive)
			{
				memcpy(x + img_size*b, old_input + img_size*b, img_size * sizeof(float));
			}
		}

		Darknet::Image im;
		im.w = net.w;
		im.h = net.h;
		im.c = net.c;
		im.data = x;
		Darknet::show_image(im, "adversarial data augmentation");
		cv::resizeWindow("adversarial data augmentation", 500, 500);
		cv::waitKey(1);

		free(old_input);
		free(truth_cpu);
		net.learning_rate = lr_old;
		net.adversarial = 0;
	}
	forward_backward_network_gpu(net, x, y);
	float error = get_network_cost(net);

	return error;
}


void pull_updates(Darknet::Layer & l)
{
	TAT(TATPARMS);

	if (l.type == Darknet::ELayerType::CONVOLUTIONAL)
	{
		cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.n);
		cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
		if(l.scale_updates)
		{
			cuda_pull_array(l.scale_updates_gpu, l.scale_updates, l.n);
		}
	}
	else if (l.type == Darknet::ELayerType::CONNECTED)
	{
		cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
		cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.outputs*l.inputs);
	}
}

void push_updates(Darknet::Layer & l)
{
	TAT(TATPARMS);

	if (l.type == Darknet::ELayerType::CONVOLUTIONAL)
	{
		cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.n);
		cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
		if(l.scale_updates) cuda_push_array(l.scale_updates_gpu, l.scale_updates, l.n);
	}
	else if (l.type == Darknet::ELayerType::CONNECTED)
	{
		cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
		cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.outputs*l.inputs);
	}
}

void update_layer(Darknet::Layer & l, Darknet::Network net)
{
	TAT(TATPARMS);

	int update_batch = net.batch*net.subdivisions;
	float rate = get_current_rate(net);
	l.t = get_current_batch(net);
	if(l.update_gpu)
	{
		l.update_gpu(l, update_batch, rate, net.momentum, net.decay, net.loss_scale);
	}
}

void merge_weights(Darknet::Layer & l, Darknet::Layer & base)
{
	TAT(TATPARMS);

	if (l.type == Darknet::ELayerType::CONVOLUTIONAL)
	{
		axpy_cpu(l.n, 1, l.biases, 1, base.biases, 1);
		axpy_cpu(l.nweights, 1, l.weights, 1, base.weights, 1);
		if (l.scales)
		{
			axpy_cpu(l.n, 1, l.scales, 1, base.scales, 1);
		}
	}
	else if (l.type == Darknet::ELayerType::CONNECTED)
	{
		axpy_cpu(l.outputs, 1, l.biases, 1, base.biases, 1);
		axpy_cpu(l.outputs*l.inputs, 1, l.weights, 1, base.weights, 1);
	}
}

void scale_weights(Darknet::Layer & l, float s)
{
	TAT(TATPARMS);

	if (l.type == Darknet::ELayerType::CONVOLUTIONAL)
	{
		scal_cpu(l.n, s, l.biases, 1);
		scal_cpu(l.nweights, s, l.weights, 1);
		if (l.scales)
		{
			scal_cpu(l.n, s, l.scales, 1);
		}
	}
	else if (l.type == Darknet::ELayerType::CONNECTED)
	{
		scal_cpu(l.outputs, s, l.biases, 1);
		scal_cpu(l.outputs*l.inputs, s, l.weights, 1);
	}
}


void pull_weights(Darknet::Layer & l)
{
	TAT(TATPARMS);

	if (l.type == Darknet::ELayerType::CONVOLUTIONAL)
	{
		cuda_pull_array(l.biases_gpu, l.biases, l.n);
#if defined(DARKNET_GPU_CUDA) && (CUDART_VERSION >= 11000)
		if (cfg_and_state.use_bf16_master_weights && l.weights_gpu16)
		{
			float *weights_gpu_tmp = cuda_make_array(NULL, l.nweights);
			cuda_convert_bf16_to_f32(l.weights_gpu16, l.nweights, weights_gpu_tmp);
			cuda_pull_array(weights_gpu_tmp, l.weights, l.nweights);
			cuda_free(weights_gpu_tmp);
		}
		else
#endif
		{
			cuda_pull_array(l.weights_gpu, l.weights, l.nweights);
		}
		if (l.scales)
		{
			cuda_pull_array(l.scales_gpu, l.scales, l.n);
		}
	}
	else if (l.type == Darknet::ELayerType::CONNECTED)
	{
		cuda_pull_array(l.biases_gpu, l.biases, l.outputs);
		cuda_pull_array(l.weights_gpu, l.weights, l.outputs*l.inputs);
	}
}

void push_weights(Darknet::Layer & l)
{
	TAT(TATPARMS);

	if(l.type == Darknet::ELayerType::CONVOLUTIONAL)
	{
		cuda_push_array(l.biases_gpu, l.biases, l.n);
#if defined(DARKNET_GPU_CUDA) && (CUDART_VERSION >= 11000)
		if (cfg_and_state.use_bf16_master_weights && l.weights_gpu16)
		{
			float *weights_gpu_tmp = cuda_make_array(l.weights, l.nweights);
			cuda_convert_f32_to_bf16(weights_gpu_tmp, l.nweights, l.weights_gpu16);
			cuda_free(weights_gpu_tmp);
			if (l.weight_compensation_gpu)
			{
				CHECK_CUDA(cudaMemsetAsync(l.weight_compensation_gpu, 0, sizeof(float) * (l.nweights / 2 + 1), get_cuda_stream()));
			}
		}
		else
#endif
		{
			cuda_push_array(l.weights_gpu, l.weights, l.nweights);
		}
		if(l.scales)
		{
			cuda_push_array(l.scales_gpu, l.scales, l.n);
		}
	}
	else if(l.type == Darknet::ELayerType::CONNECTED)
	{
		cuda_push_array(l.biases_gpu, l.biases, l.outputs);
		cuda_push_array(l.weights_gpu, l.weights, l.outputs*l.inputs);
	}
}

void distribute_weights(Darknet::Layer & l, Darknet::Layer & base)
{
	TAT(TATPARMS);

	if(l.type == Darknet::ELayerType::CONVOLUTIONAL)
	{
		copy_cpu(l.n, base.biases, 1, l.biases, 1);
		copy_cpu(l.nweights, base.weights, 1, l.weights, 1);
		if (base.scales) copy_cpu(l.n, base.scales, 1, l.scales, 1);
		push_weights(l);
	}
	else if(l.type == Darknet::ELayerType::CONNECTED)
	{
		cuda_push_array(l.biases_gpu, base.biases, l.outputs);
		cuda_push_array(l.weights_gpu, base.weights, l.outputs*l.inputs);
	}
}


void merge_updates(Darknet::Layer & l, Darknet::Layer & base)
{
	TAT(TATPARMS);

	if (l.type == Darknet::ELayerType::CONVOLUTIONAL) {
		axpy_cpu(l.n, 1, l.bias_updates, 1, base.bias_updates, 1);
		axpy_cpu(l.nweights, 1, l.weight_updates, 1, base.weight_updates, 1);
		if (l.scale_updates) {
			axpy_cpu(l.n, 1, l.scale_updates, 1, base.scale_updates, 1);
		}
	} else if(l.type == Darknet::ELayerType::CONNECTED) {
		axpy_cpu(l.outputs, 1, l.bias_updates, 1, base.bias_updates, 1);
		axpy_cpu(l.outputs*l.inputs, 1, l.weight_updates, 1, base.weight_updates, 1);
	}
}

void distribute_updates(Darknet::Layer & l, Darknet::Layer & base)
{
	TAT(TATPARMS);

	if(l.type == Darknet::ELayerType::CONVOLUTIONAL)
	{
		cuda_push_array(l.bias_updates_gpu, base.bias_updates, l.n);
		cuda_push_array(l.weight_updates_gpu, base.weight_updates, l.nweights);
		if(base.scale_updates)
		{
			cuda_push_array(l.scale_updates_gpu, base.scale_updates, l.n);
		}
	}
	else if (l.type == Darknet::ELayerType::CONNECTED)
	{
		cuda_push_array(l.bias_updates_gpu, base.bias_updates, l.outputs);
		cuda_push_array(l.weight_updates_gpu, base.weight_updates, l.outputs*l.inputs);
	}
}

void sync_layer(Darknet::Network * nets, int n, int j)
{
	TAT(TATPARMS);

	Darknet::Network net = nets[0];
	Darknet::Layer & base = net.layers[j];
	cuda_set_device(net.gpu_index);
	pull_weights(base);

	for (int i = 1; i < n; ++i)
	{
		cuda_set_device(nets[i].gpu_index);
		Darknet::Layer & l = nets[i].layers[j];
		pull_weights(l);
		merge_weights(l, base);
	}

	scale_weights(base, 1./n);

	for (int i = 0; i < n; ++i)
	{
		cuda_set_device(nets[i].gpu_index);
		Darknet::Layer & l = nets[i].layers[j];
		distribute_weights(l, base);
	}
}


void sync_nets(Darknet::Network * nets, int n, int interval)
{
	TAT(TATPARMS);

	int layers = nets[0].n;

	std::vector<std::thread> threads;
	threads.reserve(layers);

	*nets[0].seen += interval * (n-1) * nets[0].batch * nets[0].subdivisions;
	for (int j = 0; j < n; ++j)
	{
		*nets[j].seen = *nets[0].seen;
	}

	for (int j = 0; j < layers; ++j)
	{
		threads.emplace_back(
				[nets,n,j]()
				{
					sync_layer(nets, n, j);
				});
	}

	for (auto & t : threads)
	{
		t.join();
	}

	return;
}

float train_networks(Darknet::Network * nets, int n, data d, int interval)
{
	TAT(TATPARMS);

	// IMPORTANT:  If we get here, we already know that n > 1!  This is only called when we have multiple GPUs.
	// There is another similar function called train_network() for single GPU (note singular name!)

#ifdef _DEBUG
	int batch = nets[0].batch;
	int subdivisions = nets[0].subdivisions;
	assert(batch * subdivisions * n == d.X.rows);
#endif

	// "errors"?  This is "loss", right?  We're adding up the loss from training a batch on each GPU?
	float * errors = (float*) calloc(n, sizeof(float));

	std::vector<std::thread> threads;
	threads.reserve(n);
	std::vector<data> p(n);

	for (int i = 0; i < n; ++i)
	{
		 p[i] = get_data_part(d, i, n);

		threads.emplace_back(
			[](Darknet::Network & net, data &d2, float * err)
			{
				TAT(TATPARMS);

				cuda_set_device(net.gpu_index);
				*err = train_network(net, d2); // note this is the "singular" train function (e.g., for a single GPU)
			},
			std::ref(nets[i]), std::ref(p[i]), errors + i);
	}

	float sum = 0.0f;
	for (int i = 0; i < n; ++i)
	{
		threads[i].join();
		sum += errors[i];
	}
	free(errors);

	//cudaDeviceSynchronize();
	*nets[0].cur_iteration += (n - 1);
	*nets[0].seen = nets[0].batch * nets[0].subdivisions * get_current_iteration(nets[0]); // remove this line, when you will save to weights-file both: seen & cur_iteration
	if (get_current_iteration(nets[0]) % interval == 0)
	{
		if (cfg_and_state.is_verbose)
		{
			*cfg_and_state.output << "Syncing..." << std::flush;
		}
		sync_nets(nets, n, interval);
		if (cfg_and_state.is_verbose)
		{
			*cfg_and_state.output << "done!" << std::endl;
		}
	}

	//cudaDeviceSynchronize();
	return sum / n;
}

float *get_network_output_layer_gpu(Darknet::Network & net, int i)
{
	TAT(TATPARMS);

	Darknet::Layer & l = net.layers[i];
	if (l.type != Darknet::ELayerType::REGION && l.type != Darknet::ELayerType::YOLO && (*net.cuda_graph_ready) == 0)
	{
		cuda_pull_array(l.output_gpu, l.output, l.outputs*l.batch);
	}

	return l.output;
}

float *get_network_output_gpu(Darknet::Network & net)
{
	TAT(TATPARMS);

	int i;
	for (i = net.n - 1; i > 0; --i)
	{
		if (net.layers[i].type != Darknet::ELayerType::COST)
		{
			break;
		}
	}

	return get_network_output_layer_gpu(net, i);
}

float *network_predict_gpu(Darknet::Network & net, float *input)
{
	TAT(TATPARMS);

	if (net.gpu_index != cuda_get_device())
	{
		cuda_set_device(net.gpu_index);
	}
	int size = get_network_input_size(net) * net.batch;
	Darknet::NetworkState state;
	state.index = 0;
	state.net = net;
	//state.input = cuda_make_array(input, size);   // memory will be allocated in the parse_network_cfg_custom()
	state.input = net.input_state_gpu;
	memcpy(net.input_pinned_cpu, input, size * sizeof(float));
	state.truth = 0;
	state.train = 0;
	state.delta = 0;

	//cudaGraphExec_t instance = (cudaGraphExec_t)net.cuda_graph_exec;
	static cudaGraphExec_t instance;

	if ((*net.cuda_graph_ready) == 0)
	{
		static cudaGraph_t graph;
		if (net.use_cuda_graph == 1)
		{
			for (int i = 0; i < 16; ++i)
			{
				switch_stream(i);
			}

			cudaStream_t stream0 = switch_stream(0);
			CHECK_CUDA(cudaDeviceSynchronize());
			*cfg_and_state.output << "Try to capture graph..." << std::endl;
			//cudaGraph_t graph = (cudaGraph_t)net.cuda_graph;
			CHECK_CUDA(cudaStreamBeginCapture(stream0, cudaStreamCaptureModeGlobal));
		}

		cuda_push_array(state.input, net.input_pinned_cpu, size);
		forward_network_gpu(net, state);

		if (net.use_cuda_graph == 1)
		{
			cudaStream_t stream0 = switch_stream(0);
			CHECK_CUDA(cudaStreamEndCapture(stream0, &graph));
			CHECK_CUDA(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));
			(*net.cuda_graph_ready) = 1;
			*cfg_and_state.output << "Graph is captured..." << std::endl;
			CHECK_CUDA(cudaDeviceSynchronize());
		}

		CHECK_CUDA(cudaStreamSynchronize(get_cuda_stream()));
	}
	else
	{
		cudaStream_t stream0 = switch_stream(0);
		CHECK_CUDA( cudaGraphLaunch(instance, stream0) );
		CHECK_CUDA( cudaStreamSynchronize(stream0) );
	}

	float *out = get_network_output_gpu(net);
	reset_wait_stream_events();
	//cuda_free(state.input);   // will be freed in the free_network()
	return out;
}
