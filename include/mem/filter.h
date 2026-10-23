#pragma once
#include <vector>

// Weights for one layer. Empty for layers without weights (ReLU, MaxPool).
struct LayerWeights {
    std::vector<float> filter;   // out_c * in_c * filter_h * filter_w
    std::vector<float> bias;     // out_c
};
