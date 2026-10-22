#include "ops/mul.h"

void conv2d(const float* img, const float* filter, float* out, const layer_config& lc) {
    const int out_h = (lc.img_h - lc.filter_h) / lc.stride + 1;
    const int out_w = (lc.img_w - lc.filter_w) / lc.stride + 1;

    for (int oy = 0; oy < out_h; ++oy) {
        for (int ox = 0; ox < out_w; ++ox) {
            float acc = 0.0f;
            for (int ky = 0; ky < lc.filter_h; ++ky) {
                for (int kx = 0; kx < lc.filter_w; ++kx) {
                    acc += img[(oy * lc.stride + ky) * lc.img_w + (ox * lc.stride + kx)]
                         * filter[ky * lc.filter_w + kx];
                }
            }
            out[oy * out_w + ox] = acc;
        }
    }
}
