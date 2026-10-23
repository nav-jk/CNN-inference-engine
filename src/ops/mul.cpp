#include "ops/mul.h"

void conv2d(const float* in, const float* w, const float* b, float* out, const layer_config& lc) {
    const int oh = lc_out_h(lc), ow = lc_out_w(lc);

    for (int oc = 0; oc < lc.out_c; ++oc) {
        for (int oy = 0; oy < oh; ++oy) {
            for (int ox = 0; ox < ow; ++ox) {
                float acc = b ? b[oc] : 0.0f;
                for (int ic = 0; ic < lc.in_c; ++ic) {
                    for (int ky = 0; ky < lc.filter_h; ++ky) {
                        for (int kx = 0; kx < lc.filter_w; ++kx) {
                            const int iy = oy * lc.stride + ky;
                            const int ix = ox * lc.stride + kx;
                            acc += in[((size_t)ic * lc.img_h + iy) * lc.img_w + ix]
                                 * w[(((size_t)oc * lc.in_c + ic) * lc.filter_h + ky) * lc.filter_w + kx];
                        }
                    }
                }
                out[((size_t)oc * oh + oy) * ow + ox] = acc;
            }
        }
    }
}

void maxpool2d(const float* in, float* out, const layer_config& lc) {
    const int oh = lc_out_h(lc), ow = lc_out_w(lc);

    for (int c = 0; c < lc.in_c; ++c) {
        for (int oy = 0; oy < oh; ++oy) {
            for (int ox = 0; ox < ow; ++ox) {
                float best = in[((size_t)c * lc.img_h + oy * lc.stride) * lc.img_w + ox * lc.stride];
                for (int ky = 0; ky < lc.filter_h; ++ky) {
                    for (int kx = 0; kx < lc.filter_w; ++kx) {
                        const float v = in[((size_t)c * lc.img_h + oy * lc.stride + ky) * lc.img_w
                                           + ox * lc.stride + kx];
                        if (v > best) best = v;
                    }
                }
                out[((size_t)c * oh + oy) * ow + ox] = best;
            }
        }
    }
}

void relu(float* data, size_t n) {
    for (size_t i = 0; i < n; ++i)
        if (data[i] < 0.0f) data[i] = 0.0f;
}

void fully_connected(const float* in, const float* w, const float* b, float* out, const layer_config& lc) {
    for (int o = 0; o < lc.out_c; ++o) {
        float acc = b ? b[o] : 0.0f;
        const float* row = w + (size_t)o * lc.in_c;
        for (int i = 0; i < lc.in_c; ++i) acc += row[i] * in[i];
        out[o] = acc;
    }
}
