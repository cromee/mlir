# mlir

This repository contains a simple fully connected neural network implemented in PyTorch (`fc_model.py`) and a C++ implementation that performs inference using Arm SVE2 intrinsics.

## Exporting weights

If you wish to use the same randomly initialised parameters as the Python model, run:

```bash
python export_weights.py
```

This will generate `fc_weights.h` that is consumed by the C++ source.

## Building the C++ inference code

The C++ implementation requires an Arm server with SVE2 support. Build with CMake:

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

The executable `fc_inference` performs multiple inferences and prints total and average runtime.
