#!/usr/bin/env python3

import numpy as np
import os
import json

np.random.seed(0xCA7CAFE)

script_dir = os.path.dirname(os.path.realpath(__file__))


sizes = []

for size_i in [1, 16, 32, 64, 128, 256, 512, 3072]:
    sizes.append((size_i, 3072, 3072))

for size_k in [256, 1024, 8192]:
    sizes.append((256, 256, size_k))

sizes.append((128, 128, 32 << 10))

with open(os.path.join(script_dir, "sizes.json"), "w") as f:
    json.dump(
        [
            {"size_i": size_i, "size_j": size_j, "size_k": size_k}
            for size_i, size_j, size_k in sizes
        ],
        f,
        indent=2,
    )

a_sizes = sorted({(size_i, size_k) for size_i, _, size_k in sizes})
b_sizes = sorted({(size_k, size_j) for _, size_j, size_k in sizes})

a_matrices = {}
b_matrices = {}

for size_i, size_k in a_sizes:
    a = np.random.randn(size_i, size_k).astype(np.float32)
    a_matrices[(size_i, size_k)] = a
    a_fname = os.path.join(script_dir, f"test_a_{size_i}x{size_k}.bin")
    with open(a_fname, "wb") as f:
        f.write(a.tobytes())
    print(f"Wrote {a_fname!r}")

for size_k, size_j in b_sizes:
    b = (np.random.randn(size_k, size_j) / size_k**0.5).astype(np.float32)
    b_matrices[(size_k, size_j)] = b
    b_fname = os.path.join(script_dir, f"test_b_{size_k}x{size_j}.bin")
    with open(b_fname, "wb") as f:
        f.write(b.tobytes())
    print(f"Wrote {b_fname!r}")

for size_i, size_j, size_k in sizes:
    c = a_matrices[(size_i, size_k)] @ b_matrices[(size_k, size_j)]
    c_fname = os.path.join(script_dir, f"test_c_{size_i}x{size_j}x{size_k}.bin")
    with open(c_fname, "wb") as f:
        f.write(c.tobytes())
    print(f"Wrote {c_fname!r}")
