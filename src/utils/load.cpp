#include "utils/load.h"
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string>

static void read_exact(FILE* f, void* dst, size_t bytes, const char* what) {
    if (std::fread(dst, 1, bytes, f) != bytes) {
        std::fclose(f);
        throw std::runtime_error(std::string("short read: ") + what);
    }
}

std::vector<LayerWeights> load_weights(const std::string& path, const config& cfg) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);

    // Tiny header: magic + layer count, so a wrong/stale file fails loudly
    uint32_t magic = 0, nlayers = 0;
    read_exact(f, &magic, 4, "magic");
    read_exact(f, &nlayers, 4, "layer count");
    if (magic != 0x574E4E43 /* "CNNW" little-endian */ || (int)nlayers != cfg.layers) {
        std::fclose(f);
        throw std::runtime_error("weight file does not match config");
    }

    std::vector<LayerWeights> out(cfg.layers);
    for (int l = 0; l < cfg.layers; ++l) {
        const layer_config& lc = cfg.layer_configs[l];
        size_t n = (size_t)lc.filter_w * lc.filter_h;
        out[l].filter.resize(n);
        read_exact(f, out[l].filter.data(), n * sizeof(float), "filter");
        // add bias here the same way once your layers use it
    }

    // Anything left over means the file and config disagree
    if (std::fgetc(f) != EOF) {
        std::fclose(f);
        throw std::runtime_error("extra data in weight file");
    }
    std::fclose(f);
    return out;
}
