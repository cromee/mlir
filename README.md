# mlir

This repository contains a simple fully connected PyTorch model defined in
`fc_model.py` and a C++ implementation using AVX-512 intrinsics.

## Build the AVX-512 inference example

The `fc_avx512_inference.cpp` file implements the same model using AVX-512
instructions and measures the average inference time.
A machine and compiler with AVX-512 support are required.

```bash
mkdir build && cd build
cmake ..
make
./fc_avx512_inference
```

The program prints the model output and the average time per inference in
milliseconds.
