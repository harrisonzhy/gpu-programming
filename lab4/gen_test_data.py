#!/usr/bin/env python3

import numpy as np
import os

np.random.seed(0xca7cafe)

script_dir = os.path.dirname(os.path.realpath(__file__))

def write_example(size_i, size_j, size_k):
    a = (np.random.randn(size_i, size_k) / size_k**0.5).astype(np.float32)
    b = np.random.randn(size_k, size_j).astype(np.float32)
    c = a @ b

    prefix = os.path.join(script_dir, f"test_{size_i}x{size_j}x{size_k}")
    a_fname = f"{prefix}_a.bin"
    b_fname = f"{prefix}_b.bin"
    c_fname = f"{prefix}_c.bin"

    with open(a_fname, "wb") as f:
        f.write(a.tobytes())
    print(f"Wrote {a_fname!r}")

    with open(b_fname, "wb") as f:
        f.write(b.tobytes())
    print(f"Wrote {b_fname!r}")

    with open(c_fname, "wb") as f:
        f.write(c.tobytes())
    print(f"Wrote {c_fname!r}")

write_example(256, 256, 256)
write_example(3072, 3072, 3072)
