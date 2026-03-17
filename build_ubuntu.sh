#!/bin/bash -e

mkdir -p build
cd build

# Pick just 1 of the following -- either Release or Debug
set BUILD_TYPE=Release
#set BUILD_TYPE=Debug

cmake -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
      -DCMAKE_C_COMPILER=gcc-11 \
      -DCMAKE_CXX_COMPILER=g++-11 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc \
      -DCMAKE_CUDA_HOST_COMPILER=gcc-11 \
      ..
make -j 8
make package

echo Done!
echo Make sure you install the .deb file:
ls -lh *.deb

cd ..
