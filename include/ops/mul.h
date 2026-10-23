#pragma once
#include <cstddef>
#include "config/default.h"

// Straightforward reference kernels (CHW layout, float32). Replace/optimise
// these later; the loader and forward pass don't care how they are implemented.

// "Valid" convolution (cross-correlation, as in PyTorch). w is OIHW, b is [out_c].
void conv2d(const float* in, const float* w, const float* b, float* out, const layer_config& lc);

void maxpool2d(const float* in, float* out, const layer_config& lc);

void relu(float* data, size_t n);

// in has lc.in_c features; w is [out_c][in_c], b is [out_c].
void fully_connected(const float* in, const float* w, const float* b, float* out, const layer_config& lc);
