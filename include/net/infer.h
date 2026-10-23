#pragma once
#include <vector>
#include "config/default.h"
#include "mem/filter.h"

// Runs every layer in order. input must hold lc_in_size(layer 0) floats (CHW).
// Returns the last layer's output. Throws std::runtime_error on a bad config
// or if weights don't match it.
std::vector<float> forward(const config& cfg,
                           const std::vector<LayerWeights>& weights,
                           const float* input);
