#pragma once
#include <string>
#include <vector>
#include "config/default.h"
#include "mem/filter.h"

// Reads a weight file (see tools/export_weights.py for the format) into one
// LayerWeights per layer, sizes derived from cfg. Throws std::runtime_error
// if the file is missing, truncated, has extra data, or doesn't match cfg.
std::vector<LayerWeights> load_weights(const std::string& path, const config& cfg);
