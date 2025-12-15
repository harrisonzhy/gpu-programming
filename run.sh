#!/bin/bash

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <file.cu> [args...]"
    exit 1
fi

CUFILE="$1"
shift
OUT="${CUFILE%.cu}"

nvcc -O3 --use_fast_math \
    -gencode arch=compute_89,code=[sm_89,compute_89] \
    --expt-relaxed-constexpr --std=c++20 \
    "$CUFILE" -o "$OUT"

./"$OUT" "$@"

