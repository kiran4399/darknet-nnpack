#ifndef SPPOOL_LAYER_H
#define SPPOOL_LAYER_H

#include "image.h"
#include "cuda.h"
#include "layer.h"
#include "network.h"

typedef layer sppool_layer;

image get_sppool_image(sppool_layer l);
sppool_layer make_sppool_layer(int batch, int h, int w, int c, int sum, int* size, int* stride, int* padding, int* hash, int* cum);
void resize_sppool_layer(sppool_layer *l, int w, int h);
void forward_sppool_layer(const sppool_layer l, network net);
void backward_sppool_layer(const sppool_layer l, network net);

#ifdef GPU
void forward_sppool_layer_gpu(sppool_layer l, network net);
void backward_sppool_layer_gpu(sppool_layer l, network net);
#endif

#endif