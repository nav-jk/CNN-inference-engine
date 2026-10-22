#pragma once
#include <vector>

struct LayerWeights {
    std::vector<float> filter;   // filter_h * filter_w per layer (extend to out_c*in_c*kh*kw later)
    std::vector<float> bias;     // optional, one per output channel
};
