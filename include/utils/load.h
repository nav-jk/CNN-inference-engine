#pragma once
#include <string>
#include <vector>
#include "config/default.h"
#include "mem/filter.h"

// Weight file format (little-endian):
//   4 bytes   magic "CNNW"
//   uint32    number of layers that HAVE weights (conv + fc), in config order
//   per such layer: float32 weights (lc_weight_count), then float32 bias (out_c)
//
// Returns one LayerWeights per layer in cfg (empty for ReLU/MaxPool).
// Throws std::runtime_error if the config is inconsistent, or the file is
// missing, truncated, has extra data, or doesn't match cfg.
std::vector<LayerWeights> load_weights(const std::string& path, const config& cfg);
