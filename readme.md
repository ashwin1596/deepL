# deepL

## Overview
DeepL is a custom deep learning framework designed for efficient graph optimization and reverse-mode autodifferentiation. The framework provides foundational tools for building and training neural networks with both Python and C++ bindings, enabling flexibility and performance for machine learning tasks.

---

## Key Features
1. Deep Learning Framework with Automatic Differentiation using Reverse AutoGrad.
2. High-Level Architecture:
   - Includes support for training and inference.
   - Forward and backward pass for each iteration.
   - Loss is calculated at the end of each iteration.
   - Gradients are calculated during the backward propagation using the chain rule.
   - Optimizer updates all the parameters based on the calculated gradients.

3. High-Level Components:
   - **Tensors**
   - **Computational Graph**
   - **Optimizers**: SGD
   - **Reverse AutoGrad**
   - **Dataset**
   - **DataLoader**
   - **Layers**: Sequential, Linear, and ReLU
   - **Loss Functions**: Cross Entropy Loss

---

## Library Structure
The project is organized into several components:

- **bindings/python**: Python bindings for easy integration with Python-based workflows.
- **docs**: Documentation for the library and examples.
- **examples**: Python-based examples demonstrating usage of the library.
- **include/deepl**: Header files defining core components, layers, loss functions, optimizers, and utilities.
- **src**: Source files implementing the core functionality.
- **scripts**: Scripts for building and maintaining the project.
- **out/build**: Compiled binaries and build artifacts.

---

## Build Instructions

### Python
To build and use the Python bindings:
1. Navigate to the `bindings/python` directory.
2. Use the following command to build the bindings:
   ```bash
   python setup.py build_ext --inplace
   ```
3. Ensure all dependencies listed in `dependencies.yml` are installed.

### C++
To build the C++ project:
1. Ensure `CMake` and a compatible compiler (e.g., MSVC or GCC) are installed.
2. Run the provided `build.sh` script:
   ```bash
   ./scripts/build.sh
   ```
3. Alternatively, configure and build manually:
   ```bash
   mkdir -p out/build && cd out/build
   cmake ../.. && cmake --build .
   ```

---

## Implementation Details
### Tensors
- Tensors are used as storage objects.
- Tensors can store data on both CPU and GPU devices, depending on the selected device.
- Support for low-level operations:
  - `reshape`, `add`, `elementwise_multiply`, `matmul`, `transpose`, `divide`, `exp`, `binaralize`, `neg`, `log`.
- Support for advanced operations:
  - `sumAlongAxis`, `softmax`, `batchMatmul`.
- All operations are supported on both GPU and CPU devices.

### Computational Graph
- Consists of graph nodes for storing data during the forward pass and adjoints during the backward pass.
- Provides wrappers for all tensor operations.
- Adjoint nodes store adjoint values and dependent nodes along with their partial derivatives.
- During the backward pass, each adjoint node processes the gradient in topological order, propagating the gradients backward.

---

## Tensor Operations
DeepL provides a `Tensor` class implemented in C++, supporting:
- Multidimensional data storage.
- GPU-accelerated operations via CUDA.
- Efficient memory management and computation.

### Key Features:
- Gradient computation using reverse-mode autodifferentiation.
- Vectorized operations for performance.

---

## How It Works
DeepL employs a computation graph to represent neural network operations. The process includes:
1. **Graph Construction**: Nodes represent operations, and edges define dependencies.
2. **Forward Pass**: Compute outputs layer-by-layer.
3. **Backward Pass**: Compute gradients using reverse-mode autodifferentiation.

---

## Example Neural Network
Here is a simple example of building and training a neural network using DeepL:

### Python:
```python
from deepl import Tensor, Layer, Loss, Optimizer

# Define the network
class SimpleNN:
    def __init__(self):
        self.layer1 = Layer(128, 64)
        self.layer2 = Layer(64, 10)

    def forward(self, x):
        x = self.layer1(x).relu()
        return self.layer2(x).softmax()

# Initialize network, loss, and optimizer
nn = SimpleNN()
loss_fn = Loss.CrossEntropy()
optimizer = Optimizer.SGD(nn.parameters(), lr=0.01)

# Training loop
for batch in data_loader:
    inputs, targets = batch
    outputs = nn.forward(inputs)
    loss = loss_fn(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## Execution Details
### C++
- Compile the code by executing the batch script `compile.sh`.
- Setup the environment details such as `batch_size`, `num_epochs`, `device_type`, and `available_devices` in `run.sh`.
- Run the executable using `run.sh`.
- Define the training/testing model in the `main.cpp` file.

---

## Project Structure
```
C:\SEM4\DEEPL
|   .gitignore
|   CMakeLists.txt
|   dependencies.yml
|   readme.md
|
+---bindings
|   \---python
|           deeplearning.cpp
|
+---docs
+---examples
|   \---python
|           gradmtest.py
|           mnist_classifier.py
|           sample1.py
|           trans_test.py
|
+---include
|   \---deepl
|       +---core
|       +---layers
|       +---loss
|       +---optimisers
|       \---utils
+---scripts
|       build.sh
+---src
    +---core
    +---layers
    +---loss
    +---optimisers
    \---utils
```

---

## Dependencies
The project requires the following dependencies:
- **Python**:
  - NumPy
  - Pybind11
  - CMake (for building bindings)
- **C++**:
  - CUDA Toolkit (for GPU acceleration)
  - Eigen (optional for matrix operations)
  - CMake

Install dependencies using:
```bash
conda env create -f dependencies.yml
```

---

## Task List
- [x] Implement Tensor class and operations.
- [x] Add Python bindings.
- [x] Create example neural networks.
- [ ] Expand documentation.
- [ ] Optimize GPU implementations.
- [ ] Add more loss functions and optimizers.

---
