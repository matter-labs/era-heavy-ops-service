#!/usr/bin/env bash

# get/update the source code and build bellman-cuda library
if cd era-bellman-cuda; then git pull; else git clone https://github.com/matter-labs/era-bellman-cuda.git --branch main era-bellman-cuda; fi
cmake -Bera-bellman-cuda/build -Sera-bellman-cuda/ -DCMAKE_BUILD_TYPE=Release
cmake --build era-bellman-cuda/build/

# build testing library
export BELLMAN_CUDA_DIR=$PWD/era-bellman-cuda
cargo test --package api-testing --release --no-run
# find target/release/deps -type f -executable -iname 'testing-*'
