#include "sppool_layer.h"
#include "cuda.h"
#include <stdio.h>

image get_sppool_image(sppool_layer l)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    return float_to_image(w,h,c,l.output);
}

image get_sppool_delta(sppool_layer l)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    return float_to_image(w,h,c,l.delta);
}

sppool_layer make_sppool_layer(int batch, int h, int w, int c, int sum, int* size, int* stride, int* padding, int* hash, int* cum)
{
    sppool_layer l = {0};
    l.type = SPPOOL;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    //l.out_w = (w + 2*padding)/stride;
    //l.out_h = (h + 2*padding)/stride;
    l.out_w = 1;
    l.out_h = 1;
    l.out_c = h*w*c;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h*w*c;
    l.spp_pad = padding;
    l.spp_size = size;
    l.spp_stride = stride;
    l.spp_hash = hash;
    l.spp_cum = cum;
    l.sum = sum;
    l.pyramids = sizeof(size);

    int output_size = l.out_h * l.out_w * l.out_c * batch;
    l.indexes = calloc(output_size, sizeof(int));
    l.output =  calloc(output_size, sizeof(float));
    l.delta =   calloc(output_size, sizeof(float));
    l.forward = forward_sppool_layer;
    l.backward = backward_sppool_layer;
    #ifdef GPU
    l.forward_gpu = forward_sppool_layer_gpu;
    l.backward_gpu = backward_sppool_layer_gpu;
    l.indexes_gpu = cuda_make_int_array(0, output_size);
    l.output_gpu  = cuda_make_array(l.output, output_size);
    l.delta_gpu   = cuda_make_array(l.delta, output_size);
    #endif
    fprintf(stderr, "spp          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);
    return l;
}

void resize_sppool_layer(sppool_layer *l, int w, int h)
{
	int sum = 0, i;
    l->h = h;
    l->w = w;
    l->inputs = h*w*l->c;

    l->out_h = 1;
    l->out_w = 1;
    l->out_c = l->sum*l->c;
    l->outputs = l->out_w * l->out_h * l->c;
    int output_size = l->outputs * l->batch;

    l->indexes = realloc(l->indexes, output_size * sizeof(int));
    l->output = realloc(l->output, output_size * sizeof(float));
    l->delta = realloc(l->delta, output_size * sizeof(float));

    #ifdef GPU
    cuda_free((float *)l->indexes_gpu);
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    l->indexes_gpu = cuda_make_int_array(0, output_size);
    l->output_gpu  = cuda_make_array(l->output, output_size);
    l->delta_gpu   = cuda_make_array(l->delta,  output_size);
    #endif
}

int compute_last(pad, stride, w, h){
	int siz = (w + 2*pad)/stride;
	return siz*siz;
}


#ifdef NNPACK
struct sppool_params {
	const sppool_layer *l;
	network *net;
};

void sppool_thread(struct sppool_params *params, size_t b, size_t k)
{
	int i, j, m, n, a;
	int c = params->l->c;
	int w = params->l->w;
	int h = params->l->h;
	int x = 0;
	int last_val = compute_last(params->l->spp_pad[x], params->l->spp_stride[x], w, h);
		for(a = 0; a<last_val; ++a){
			int w_offset = -params->l->spp_pad[x];
			int h_offset = -params->l->spp_pad[x];

			int out_index = a + params->l->sum*(k + c*b);
			float max = -FLT_MAX;
			int max_i = -1;
			for(n = 0; n < params->l->size; ++n){
				for(m = 0; m < params->l->size; ++m){
					int cur_h = h_offset + i*params->l->stride + n;
					int cur_w = w_offset + j*params->l->stride + m;
					int index = cur_w + params->l->w*(cur_h + params->l->h*(k + b*params->l->c));
					int valid = (cur_h >= 0 && cur_h < params->l->h &&
								 cur_w >= 0 && cur_w < params->l->w);
					float val = (valid != 0) ? params->net->input[index] : -FLT_MAX;
					max_i = (val > max) ? index : max_i;
					max   = (val > max) ? val   : max;
				}
			}
			params->l->output[out_index] = max;
			params->l->indexes[out_index] = max_i;

			if(a == last_val-1 && last_val != params->l->sum-1){
				last_val = compute_last(params->l->spp_pad[x++], params->l->spp_stride[x++], w, h);
			}
		}
}
#endif

void forward_sppool_layer(const sppool_layer l, network net)
{
#ifdef NNPACK
	struct sppool_params params = { &l, &net };
	pthreadpool_compute_2d(net.threadpool, (pthreadpool_function_2d_t)sppool_thread,
	&params, l.batch, l.c);
#else
	int a,b,i,j,k,m,n,p;

	int c = l.c;
	int start = 0;
	int out_index = 0;

	for(b = 0; b < l.batch; ++b){
		for(k = 0; k < c; ++k){
			int x = 0;
			int last_val = compute_last(l.spp_pad[x], l.spp_stride[x], l.w, l.h);
			for(a = 0; a<last_val; ++a){
				int w_offset = -l.spp_pad[x];
				int h_offset = -l.spp_pad[x];

				int out_index = a + l.sum*(k + c*b);
				float max = -FLT_MAX;
				int max_i = -1;
				for(n = 0; n < l.spp_size[x]; ++n){
					for(m = 0; m < l.spp_size[x]; ++m){
						int cur_h = h_offset + i*l.spp_stride[x] + n;
						int cur_w = w_offset + j*l.spp_stride[x] + m;
						int index = cur_w + l.w*(cur_h + l.h*(k + b*l.c));
						int valid = (cur_h >= 0 && cur_h < l.h &&
									 cur_w >= 0 && cur_w < l.w);
						float val = (valid != 0) ? net.input[index] : -FLT_MAX;
						max_i = (val > max) ? index : max_i;
						max   = (val > max) ? val   : max;
					}
				}
				l.output[out_index] = max;
				l.indexes[out_index] = max_i;

				if(a == last_val-1 && last_val != l.sum-1){
					last_val = compute_last(l.spp_pad[x++], l.spp_stride[x++], l.w, l.h);
				}
			}
		}
	}
#endif
}

void backward_sppool_layer(const sppool_layer l, network net)
{
    int i;
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    for(i = 0; i < h*w*c*l.batch; ++i){
        int index = l.indexes[i];
        net.delta[index] += l.delta[i];
    }
}

