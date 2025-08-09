import subprocess
import time
import numpy as np
import torch
from fc_model import SimpleFCModel

ITERATIONS = 10000


def save_data():
    torch.manual_seed(0)
    model = SimpleFCModel(10, 20, 1)
    inp = torch.randn(1, 10)
    np.savetxt("fc1_weight.txt", model.fc1.weight.detach().numpy().reshape(-1))
    np.savetxt("fc1_bias.txt", model.fc1.bias.detach().numpy())
    np.savetxt("fc2_weight.txt", model.fc2.weight.detach().numpy().reshape(-1))
    np.savetxt("fc2_bias.txt", model.fc2.bias.detach().numpy())
    np.savetxt("input.txt", inp.numpy().reshape(-1))
    return model, inp


def benchmark_python(model, inp):
    start = time.time()
    with torch.no_grad():
        for _ in range(ITERATIONS):
            model(inp)
        output = model(inp).detach().numpy().reshape(-1)
    end = time.time()
    avg_us = (end - start) / ITERATIONS * 1e6
    np.savetxt("py_output.txt", output)
    return avg_us, output


def benchmark_cpp():
    subprocess.run(["g++", "-O3", "fc_inference.cpp", "-o", "fc_inference"], check=True)
    result = subprocess.run(["./fc_inference", str(ITERATIONS)], capture_output=True, text=True, check=True)
    cpp_time = float(result.stdout.strip())
    cpp_output = np.loadtxt("cpp_output.txt").reshape(-1)
    return cpp_time, cpp_output


def main():
    model, inp = save_data()
    py_time, py_output = benchmark_python(model, inp)
    cpp_time, cpp_output = benchmark_cpp()
    speedup = py_time / cpp_time
    print(f"Python avg time: {py_time:.2f} us")
    print(f"C++ avg time: {cpp_time:.2f} us")
    print(f"Speedup: {speedup:.2f}x")
    if np.allclose(py_output, cpp_output, atol=1e-5):
        print("Outputs match within tolerance")
    else:
        max_diff = np.max(np.abs(py_output - cpp_output))
        print(f"Outputs differ (max diff {max_diff})")


if __name__ == "__main__":
    main()

