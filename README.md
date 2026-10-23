# CNN Inference Engine

A small CNN inference engine in C++17 with no dependencies. It loads trained
weights from a flat binary file and runs a forward pass on CPU using plain
reference kernels (convolution, ReLU, max-pool, fully connected).

Tensors are float32, stored CHW and row-major.

## Build and run

```bash
make                 # builds build/main
make run             # runs LeNet on lenet.bin / test_image.bin / expected_logits.bin
make clean
```

`make run` needs three files in the repo root. Generate them with either script:

```bash
# Option A: random weights, checked against an independent NumPy implementation.
# Needs only numpy.
python3 tools/make_test_model.py

# Option B: train a real LeNet on MNIST with PyTorch and export it.
# Needs torch, torchvision, numpy (downloads MNIST into ./data).
python3 tools/train_lenet.py
```

Expected output ends with:

```
max |engine - expected| = <small number, ~1e-6>  ->  MATCH
```

`MISMATCH` (exit code 2) means the weight layout or a kernel disagrees with the
reference.

You can also run the binary directly:

```bash
./build/main <weights.bin> <image.bin> [expected_logits.bin]
```

`image.bin` is raw float32 CHW with exactly the first layer's input size
(784 floats for the LeNet config). `expected_logits.bin` is optional.

## Project layout

```
main.cpp                    defines the LeNet config, loads weights, runs, compares
include/config/default.h    layer_config / config structs and size helpers
include/config/validate.h   config consistency checks
include/utils/load.h        weight file format + load_weights()
include/mem/filter.h        LayerWeights (weights + bias for one layer)
include/ops/mul.h           conv2d, maxpool2d, relu, fully_connected
include/net/infer.h         forward()
src/                        implementations of the above
tools/make_test_model.py    NumPy reference: random weights + expected logits
tools/train_lenet.py        PyTorch: train on MNIST and export
```

## Defining a network

A network is an array of `layer_config`, one per layer, in execution order:

```cpp
//                 stride fw fh  img_w img_h in_c out_c type
layer_config layers[] = {
    {1, 5, 5, 28, 28,   1,   6, LAYER_CONV},
    {1, 1, 1, 24, 24,   6,   6, LAYER_RELU},
    {2, 2, 2, 24, 24,   6,   6, LAYER_MAXPOOL},
    ...
};
config cfg = { (int)(sizeof(layers) / sizeof(layers[0])), layers };
```

`img_w` / `img_h` are the layer's input size. Rules per layer type:

| Type          | Rules                                                               |
|---------------|---------------------------------------------------------------------|
| `LAYER_CONV`  | valid convolution (no padding); weights `[out_c][in_c][fh][fw]`     |
| `LAYER_RELU`  | 1x1 filter, stride 1, `out_c == in_c`                               |
| `LAYER_MAXPOOL` | filter = window, stride = step, `out_c == in_c`                   |
| `LAYER_FC`    | input 1x1, filter 1x1, stride 1; `in_c` = input features, weights `[out_c][in_c]` |

`validate_config()` runs automatically when loading and running, and rejects
non-positive sizes, wrong shapes for a layer type, and any layer whose input
size doesn't match the previous layer's output size.

## Weight file format

Little-endian:

```
4 bytes   magic "CNNW"
uint32    number of layers that have weights (conv + fc), in config order
per such layer:
    float32 weights   (out_c * in_c * filter_h * filter_w values)
    float32 bias      (out_c values)
```

ReLU and MaxPool layers have no entry in the file.

`load_weights()` throws `std::runtime_error` if the file is missing, truncated,
has extra data, or its header doesn't match the config.

Exporting from PyTorch: write each weighted layer's `weight` then `bias` with
`.detach().numpy().astype("<f4").tofile(f)`. Conv weights are already OIHW and
linear weights `[out, in]`, which is what the loader expects. See
`tools/train_lenet.py`.

## Current limitations

- Single image (no batching), CPU only, float32 only.
- No padding, dilation, groups, batch norm or residual connections. Fold batch
  norm into the preceding conv when exporting.
- Kernels are straightforward loops, written for correctness rather than speed.

## Ideas for next steps

- Faster conv: better loop order, im2col plus a blocked matrix multiply, SIMD.
- Padding support and more layer types.
- Memory-map the weight file instead of reading it with `fread`.
- Export from ONNX instead of a custom script.