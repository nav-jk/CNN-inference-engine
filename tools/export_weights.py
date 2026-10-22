"""Export per-layer weights to the flat binary format read by load_weights().

Format (little-endian):
    4 bytes   magic "CNNW"
    uint32    number of layers
    per layer: float32 filter values, row-major (for conv: OIHW order)
"""
import struct
import sys
import numpy as np


def export(path, layer_weights):
    with open(path, "wb") as f:
        f.write(b"CNNW")
        f.write(struct.pack("<I", len(layer_weights)))
        for w in layer_weights:
            np.ascontiguousarray(w, dtype="<f4").tofile(f)


if __name__ == "__main__":
    # Demo file matching main.cpp: one 3x3 filter (a simple edge detector)
    out = sys.argv[1] if len(sys.argv) > 1 else "weights.bin"
    demo = np.array([[1, 0, -1],
                     [2, 0, -2],
                     [1, 0, -1]], dtype=np.float32)
    export(out, [demo])
    print(f"wrote {out}")
