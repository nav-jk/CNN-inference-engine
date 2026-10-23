#include "net/infer.h"
#include "config/validate.h"
#include "ops/mul.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>

std::vector<float> forward(const config& cfg,
                           const std::vector<LayerWeights>& weights,
                           const float* input) {
    validate_config(cfg);
    if ((int)weights.size() != cfg.layers)
        throw std::runtime_error("weights do not match config (layer count)");

    // Two ping-pong buffers sized for the largest activation: no allocation per layer
    size_t max_elems = lc_in_size(cfg.layer_configs[0]);
    for (int l = 0; l < cfg.layers; ++l)
        max_elems = std::max(max_elems, lc_out_size(cfg.layer_configs[l]));

    std::vector<float> buf_a(max_elems), buf_b(max_elems);
    float* cur = buf_a.data();
    float* nxt = buf_b.data();
    std::memcpy(cur, input, lc_in_size(cfg.layer_configs[0]) * sizeof(float));

    for (int l = 0; l < cfg.layers; ++l) {
        const layer_config& lc = cfg.layer_configs[l];
        const LayerWeights& w = weights[l];

        if (lc_has_weights(lc) &&
            (w.filter.size() != lc_weight_count(lc) || w.bias.size() != (size_t)lc.out_c))
            throw std::runtime_error("layer " + std::to_string(l) + ": weight sizes do not match config");

        switch (lc.type) {
        case LAYER_CONV:
            conv2d(cur, w.filter.data(), w.bias.data(), nxt, lc);
            std::swap(cur, nxt);
            break;
        case LAYER_MAXPOOL:
            maxpool2d(cur, nxt, lc);
            std::swap(cur, nxt);
            break;
        case LAYER_FC:
            fully_connected(cur, w.filter.data(), w.bias.data(), nxt, lc);
            std::swap(cur, nxt);
            break;
        case LAYER_RELU:
            relu(cur, lc_out_size(lc));   // in place, no swap
            break;
        }
    }

    const size_t n_out = lc_out_size(cfg.layer_configs[cfg.layers - 1]);
    return std::vector<float>(cur, cur + n_out);
}
