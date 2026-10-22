#include <cstdio>
#include <vector>
#include <exception>
#include "config/default.h"
#include "utils/load.h"
#include "ops/mul.h"

int main(int argc, char** argv) {
    const char* weights_path = (argc > 1) ? argv[1] : "weights.bin";

    layer_config layers[] = { {/*stride*/1, /*fw*/3, /*fh*/3, /*img_w*/5, /*img_h*/5} };
    config cfg = { 1, layers };

    try {
        std::vector<LayerWeights> weights = load_weights(weights_path, cfg);

        const layer_config& lc = cfg.layer_configs[0];
        std::printf("layer 0 filter (%dx%d):\n", lc.filter_h, lc.filter_w);
        for (int y = 0; y < lc.filter_h; ++y) {
            for (int x = 0; x < lc.filter_w; ++x)
                std::printf("%8.3f", weights[0].filter[y * lc.filter_w + x]);
            std::printf("\n");
        }

        std::vector<float> img(lc.img_w * lc.img_h);
        for (size_t i = 0; i < img.size(); ++i) img[i] = (float)i;

        const int out_h = (lc.img_h - lc.filter_h) / lc.stride + 1;
        const int out_w = (lc.img_w - lc.filter_w) / lc.stride + 1;
        std::vector<float> out(out_h * out_w);
        conv2d(img.data(), weights[0].filter.data(), out.data(), lc);

        std::printf("output (%dx%d):\n", out_h, out_w);
        for (int y = 0; y < out_h; ++y) {
            for (int x = 0; x < out_w; ++x)
                std::printf("%8.3f", out[y * out_w + x]);
            std::printf("\n");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
