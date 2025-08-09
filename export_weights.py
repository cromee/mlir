import torch
from fc_model import SimpleFCModel


def export_weights(header_path: str = "fc_weights.h") -> None:
    """Export the randomly initialised fc_model parameters to a C++ header."""
    model = SimpleFCModel(input_dim=10, hidden_dim=20, output_dim=1)
    state = model.state_dict()
    with open(header_path, "w") as f:
        f.write('#pragma once\n')
        f.write('#include <array>\n')
        fc1w = state['fc1.weight'].view(-1).numpy()
        f.write('static const std::array<float, %d> FC1_WEIGHT = {' % fc1w.size)
        f.write(', '.join(f"{v:.8f}f" for v in fc1w))
        f.write('};\n')
        fc1b = state['fc1.bias'].view(-1).numpy()
        f.write('static const std::array<float, %d> FC1_BIAS = {' % fc1b.size)
        f.write(', '.join(f"{v:.8f}f" for v in fc1b))
        f.write('};\n')
        fc2w = state['fc2.weight'].view(-1).numpy()
        f.write('static const std::array<float, %d> FC2_WEIGHT = {' % fc2w.size)
        f.write(', '.join(f"{v:.8f}f" for v in fc2w))
        f.write('};\n')
        fc2b = state['fc2.bias'].view(-1).numpy()
        f.write('static const std::array<float, %d> FC2_BIAS = {' % fc2b.size)
        f.write(', '.join(f"{v:.8f}f" for v in fc2b))
        f.write('};\n')


if __name__ == '__main__':
    export_weights()
