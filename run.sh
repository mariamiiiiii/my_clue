#! /bin/bash

rm -rf build

CUDA_BASE=/usr/local/cuda-12.8
ROCM_BASE=/opt/rocm-6.4.1

cmake \
  -DCMAKE_CUDA_COMPILER="${CUDA_BASE}"/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -B build \
  -S . \
  -L

cmake --build build

rm -f Results/*.csv

# run without prefetching
for i in {0..10}; do
  ./build/src/clue/main -i data/input/aniso_1000.csv -d 7.0 -r 10.0 -o 2 -e 1000 -v -u -l $i
done

# run with prefetching
for i in {0..10}; do
  ./build/src/clue/main -i data/input/aniso_1000.csv -d 7.0 -r 10.0 -o 2 -e 1000 -v -u -p -l $i
done
