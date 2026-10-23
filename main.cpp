#include <cmath>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>
#include "config/default.h"
#include "net/infer.h"
#include "utils/load.h"

static std::vector<float> read_floats(const std::string& path, size_t count) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<float> v(count);
    const size_t got = std::fread(v.data(), sizeof(float), count, f);
    const bool extra = std::fgetc(f) != EOF;
    std::fclose(f);
    if (got != count || extra)
        throw std::runtime_error(path + ": expected exactly " + std::to_string(count) + " floats");
    return v;
}

int main(int argc, char** argv) {
    const std::string model_path    = (argc > 1) ? argv[1] : "lenet.bin";
    const std::string image_path    = (argc > 2) ? argv[2] : "test_image.bin";
    const std::string expected_path = (argc > 3) ? argv[3] : "";

    // LeNet for 28x28 grayscale input (e.g. MNIST).
    //                 stride fw fh  img_w img_h in_c out_c type
    layer_config layers[] = {
        {1, 5, 5, 28, 28,   1,   6, LAYER_CONV},
        {1, 1, 1, 24, 24,   6,   6, LAYER_RELU},
        {2, 2, 2, 24, 24,   6,   6, LAYER_MAXPOOL},
        {1, 5, 5, 12, 12,   6,  16, LAYER_CONV},
        {1, 1, 1,  8,  8,  16,  16, LAYER_RELU},
        {2, 2, 2,  8,  8,  16,  16, LAYER_MAXPOOL},
        {1, 1, 1,  1,  1, 256, 120, LAYER_FC},
        {1, 1, 1,  1,  1, 120, 120, LAYER_RELU},
        {1, 1, 1,  1,  1, 120,  84, LAYER_FC},
        {1, 1, 1,  1,  1,  84,  84, LAYER_RELU},
        {1, 1, 1,  1,  1,  84,  10, LAYER_FC},
    };
    config cfg = { (int)(sizeof(layers) / sizeof(layers[0])), layers };

    try {
        std::vector<LayerWeights> weights = load_weights(model_path, cfg);
        std::vector<float> image = read_floats(image_path, lc_in_size(layers[0]));

        std::vector<float> logits = forward(cfg, weights, image.data());

        int best = 0;
        std::printf("logits:");
        for (size_t i = 0; i < logits.size(); ++i) {
            std::printf(" %.4f", logits[i]);
            if (logits[i] > logits[best]) best = (int)i;
        }
        std::printf("\npredicted class: %d\n", best);

        if (!expected_path.empty()) {
            std::vector<float> expected = read_floats(expected_path, logits.size());
            float max_diff = 0.0f;
            for (size_t i = 0; i < logits.size(); ++i)
                max_diff = std::fmax(max_diff, std::fabs(logits[i] - expected[i]));
            std::printf("max |engine - expected| = %.3e  ->  %s\n", max_diff,
                        max_diff < 1e-3f ? "MATCH" : "MISMATCH");
            return max_diff < 1e-3f ? 0 : 2;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
