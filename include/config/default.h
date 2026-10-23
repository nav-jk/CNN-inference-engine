#pragma once
#include <cstddef>

typedef enum layer_type {
    LAYER_CONV,
    LAYER_RELU,
    LAYER_MAXPOOL,
    LAYER_FC
} layer_type;

// All tensors are stored CHW (channel, height, width), row-major, float32.
//
// Conv:    weights [out_c][in_c][filter_h][filter_w] (PyTorch OIHW), bias [out_c]
// MaxPool: filter_w/filter_h = window, stride = step, out_c == in_c
// ReLU:    filter 1x1, stride 1, out_c == in_c (img_w/img_h = current size)
// FC:      in_c = input features, out_c = output features,
//          img 1x1, filter 1x1, stride 1; weights [out_c][in_c], bias [out_c]
typedef struct layer_config {
    int stride;
    int filter_w;
    int filter_h;
    int img_w;        // input width of this layer
    int img_h;        // input height of this layer
    int in_c;         // input channels
    int out_c;        // output channels
    layer_type type;
} layer_config;

typedef struct config {
    int layers;
    layer_config* layer_configs;
} config;

static inline int lc_out_h(const layer_config& lc) { return (lc.img_h - lc.filter_h) / lc.stride + 1; }
static inline int lc_out_w(const layer_config& lc) { return (lc.img_w - lc.filter_w) / lc.stride + 1; }
static inline size_t lc_in_size(const layer_config& lc)  { return (size_t)lc.in_c * lc.img_h * lc.img_w; }
static inline size_t lc_out_size(const layer_config& lc) { return (size_t)lc.out_c * lc_out_h(lc) * lc_out_w(lc); }
static inline bool lc_has_weights(const layer_config& lc) { return lc.type == LAYER_CONV || lc.type == LAYER_FC; }
static inline size_t lc_weight_count(const layer_config& lc) {
    return (size_t)lc.out_c * lc.in_c * lc.filter_h * lc.filter_w;
}
