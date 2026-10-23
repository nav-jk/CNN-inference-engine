"""Generate a RANDOM-weight LeNet plus expected logits, using only NumPy.

Use this to check the C++ engine without PyTorch or any download: the expected
logits come from an independent NumPy implementation, so a match means the
file layout, loader and kernels all agree.

Run from the repo root:   python3 tools/make_test_model.py
Writes lenet.bin, test_image.bin, expected_logits.bin (same files as train_lenet.py).
"""
import struct
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

rng = np.random.default_rng(0)


def conv(x, w, b):                       # x: (C,H,W)  w: (O,C,kh,kw)  b: (O,)
    win = sliding_window_view(x, w.shape[2:], axis=(1, 2))     # (C,oh,ow,kh,kw)
    return np.einsum("cijkl,ockl->oij", win, w) + b[:, None, None]


def pool2(x):                            # 2x2 max pool, stride 2
    c, h, w = x.shape
    return x.reshape(c, h // 2, 2, w // 2, 2).max(axis=(2, 4))


def layer(out_c, in_c, *k):
    fan_in = in_c * int(np.prod(k))
    w = rng.normal(0, np.sqrt(2.0 / fan_in), (out_c, in_c, *k)).astype(np.float32)
    b = rng.normal(0, 0.1, out_c).astype(np.float32)
    return w, b


c1, c2 = layer(6, 1, 5, 5), layer(16, 6, 5, 5)
f1, f2, f3 = layer(120, 256), layer(84, 120), layer(10, 84)
image = rng.random((1, 28, 28)).astype(np.float32)

x = image.astype(np.float64)
x = pool2(np.maximum(conv(x, *map(np.float64, c1)), 0))
x = pool2(np.maximum(conv(x, *map(np.float64, c2)), 0))
x = x.reshape(-1)                        # CHW flatten, same as PyTorch
for (w, b), act in ((f1, True), (f2, True), (f3, False)):
    x = w.astype(np.float64) @ x + b
    if act:
        x = np.maximum(x, 0)

with open("lenet.bin", "wb") as f:
    f.write(b"CNNW")
    f.write(struct.pack("<I", 5))
    for w, b in (c1, c2, f1, f2, f3):
        w.astype("<f4").tofile(f)
        b.astype("<f4").tofile(f)
image.astype("<f4").tofile("test_image.bin")
x.astype("<f4").tofile("expected_logits.bin")
print("wrote lenet.bin, test_image.bin, expected_logits.bin")
