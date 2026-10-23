#pragma once
#include "config/default.h"

// Throws std::runtime_error with the layer index if the config is inconsistent
// (non-positive sizes, wrong shape rules for a layer type, or a layer whose
// input size doesn't match the previous layer's output size).
void validate_config(const config& cfg);
