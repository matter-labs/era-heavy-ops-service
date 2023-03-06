#!/usr/bin/env bash

# get/update the source code and build bellman-cuda library
if cd bellman-cuda; then git pull; else git clone https://github.com/matter-labs/era-bellman-cuda.git --branch main bellman-cuda; fi
cmake -Bbellman-cuda/build -Sbellman-cuda/ -DCMAKE_BUILD_TYPE=Release
cmake --build bellman-cuda/build/

# build testing library
export BELLMAN_CUDA_DIR=$PWD/bellman-cuda
cargo test --package testing --release --no-run
