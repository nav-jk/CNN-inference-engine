#pragma once

typedef struct layer_config {
    int stride;
    int filter_w;
    int filter_h;
    int img_w;
    int img_h;
} layer_config;

typedef struct config {
    int layers;
    layer_config* layer_configs;
} config;
