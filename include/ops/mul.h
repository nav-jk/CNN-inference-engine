#pragma once

#include "config/default.h"

// Placeholder single-channel "valid" convolution so the project builds and the
// loaded weights can be exercised. Replace with real kernel.
void conv2d(const float* img, const float* filter, float* out, const layer_config& lc);
