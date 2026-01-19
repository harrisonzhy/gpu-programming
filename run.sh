#!/bin/bash

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <file.cu> [args...]"
    exit 1
fi

CUFILE="$1"
shift
OUT="${CUFILE%.cu}"

#-gencode arch=compute_90a,code=[sm_90a,compute_90a] \
nvcc -O3 --use_fast_math \
    -arch=sm_90a \
    -lcublas \
    --expt-relaxed-constexpr --std=c++20 \
    "$CUFILE" -o "$OUT" -lcuda

./"$OUT" "$@"

