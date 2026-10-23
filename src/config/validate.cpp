#include "config/validate.h"
#include <stdexcept>
#include <string>

static void fail(int l, const char* msg) {
    throw std::runtime_error("layer " + std::to_string(l) + ": " + msg);
}

void validate_config(const config& cfg) {
    if (cfg.layers <= 0 || !cfg.layer_configs)
        throw std::runtime_error("config has no layers");

    for (int l = 0; l < cfg.layers; ++l) {
        const layer_config& lc = cfg.layer_configs[l];

        if (lc.stride <= 0 || lc.filter_w <= 0 || lc.filter_h <= 0 ||
            lc.img_w <= 0 || lc.img_h <= 0 || lc.in_c <= 0 || lc.out_c <= 0)
            fail(l, "all sizes must be positive");
        if (lc.img_w < lc.filter_w || lc.img_h < lc.filter_h)
            fail(l, "filter is larger than the input");

        switch (lc.type) {
        case LAYER_CONV:
            break;
        case LAYER_RELU:
            if (lc.filter_w != 1 || lc.filter_h != 1 || lc.stride != 1 || lc.out_c != lc.in_c)
                fail(l, "relu needs 1x1 filter, stride 1 and out_c == in_c");
            break;
        case LAYER_MAXPOOL:
            if (lc.out_c != lc.in_c)
                fail(l, "maxpool needs out_c == in_c");
            break;
        case LAYER_FC:
            if (lc.img_w != 1 || lc.img_h != 1 || lc.filter_w != 1 || lc.filter_h != 1 || lc.stride != 1)
                fail(l, "fc needs img 1x1, filter 1x1 and stride 1 (in_c = input features)");
            break;
        }

        if (l > 0 && lc_out_size(cfg.layer_configs[l - 1]) != lc_in_size(lc))
            fail(l, "input size does not match the previous layer's output size");
    }
}
