# DeepL: Efficient Deep Learning Framework

A high-performance deep learning framework specifically designed to solve complex graph optimization challenges in machine learning, enabling researchers and developers to build more efficient and precise neural network models.

## Problem Solved

Scientific and industrial machine learning projects often struggle with computational inefficiency and limited flexibility when handling complex neural network architectures. This framework addresses these challenges by providing a unified solution that bridges performance and adaptability.

## Key Features
- Deep Learning Framework with Automatic Differentiation using Reverse AutoGrad
- High-Performance Tensor Operations on both CPU and GPU
- Dynamic Computational Graph
- Python and C++ APIs
- CUDA-accelerated computations
- Built-in Components:
  - Optimizers: SGD
  - Layers: Sequential, Linear, ReLU
  - Loss Functions: Cross Entropy Loss
  - Automatic Differentiation Engine

## Implementation Highlights
- Designed and implemented a custom deep learning framework with reverse-mode auto-differentiation, supporting
 efficient computation graphs for backpropagation in neural networks.
- Optimized tensor operations on CPU and GPU by implementing parallelized matrix operations, fused kernels, and
 CUDA-based execution, improving computational efficiency.
- Developed a Python-C++ bridge using Pybind11, enabling seamless interaction between high-level Python API and
 low-level C++ core for efficient model execution.
- Implemented dynamic memory management and optimized data structures for tensor storage and operations, reduc
ing memory fragmentation and improving runtime performance.

## Quick Start
```bash
# Clone the repository
git clone https://github.com/ashwin1596/deepL.git
cd deepL

# Setup environment
conda env create -f environment.yml
conda activate deepl

# Build the library
bash scripts/build.sh

# Set Python path
export PYTHONPATH="${PWD}/build:${PYTHONPATH}"

# Run example
python examples/python/mnist_classifier.py
```

## Prerequisites
- Python 3.8 or higher
- CUDA Toolkit 11.0 or higher
- CMake 3.15 or higher
- C++17 compatible compiler
- GPU with compute capability 6.0 or higher(for faster computation), if not CPU mode can be used
- Required Python packages:
  - NumPy
  - PyBind11
  - PyTorch (for examples)
  - torchvision (for MNIST example)

## Installation

### Building from Source
1. **Clone the Repository**:
```bash
git clone https://github.com/ashwin1596/deepL.git
cd deepL
```

2. **Create Conda Environment**:
```bash
conda env create -f environment.yml
conda activate deepl
```

3. **Build the Library**:
```bash
bash scripts/build.sh
```

4. **Set Python Path**:
```bash
export PYTHONPATH="${PWD}/build:${PYTHONPATH}"
```

### Common Installation Issues
- **CUDA Not Found**: Verify CUDA installation and ensure `CUDA_HOME` is set correctly
- **Build Fails**: Check compiler compatibility and CMake version
- **Import Errors**: Verify `PYTHONPATH` is set correctly
- **Missing Dependencies**: Ensure all required packages are installed via conda/pip

## Usage Examples

### Simple Neural Network
```python
import deepl as dl
import numpy as np

# Create input and target data
input_data = np.random.randn(10, 2).astype(np.float32)
target_data = np.random.randn(10, 1).astype(np.float32)

# Create input and target nodes
input_node = dl.Tensor(input_data)
target_node = dl.Tensor(target_data)

# Create model
builder = dl.GraphBuilder()
model = dl.Sequential(builder)
model.add_layer(dl.Linear(2, 4, builder))
model.add_layer(dl.ReLU(builder))
model.add_layer(dl.Linear(4, 1, builder))

# Create loss function
loss_fn = dl.CrossEntropyLoss(builder)

# Create optimizer
parameters = model.parameters()
optimizer = dl.SGD(parameters, learning_rate=0.01)

# Training loop
for epoch in range(10):
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model.forward(input_node)
    loss = loss_fn.forward(outputs, target_node)
    
    # Backward pass
    builder.backward(loss)
    optimizer.step()
```

### Using C++ API
```cpp
#include <deepl/core/tensor.cuh>
#include <deepl/layers/sequential.h>
#include <deepl/builder/graph_builder.h>

int main() {
    // Create model
    auto builder = GraphBuilder();
    auto model = Sequential(builder);
    model.add_layer(Linear(784, 128, builder));
    model.add_layer(ReLU(builder));
    model.add_layer(Linear(128, 10, builder));
    
    return 0;
}
```

---

## Library Structure
The project is organized into several components:

- **bindings/python**: Python bindings for easy integration with Python-based workflows.
- **docs**: Documentation for the library and examples.
- **examples**: Python and C++ based examples demonstrating usage of the library.
- **include/deepl**: Header files defining core components, layers, loss functions, optimizers, and utilities.
- **src**: Source files implementing the core functionality.
- **scripts**: Scripts for building and maintaining the project.

---

## Build Instructions

Navigate to scripts and run build.sh
```bash
bash build.sh
```

### Python
For python set the `PYTHONPATH` to build directory
```bash
export PYTHONPATH=$(pwd)/build:$PYTHONPATH
```

To use the library you can import it as `import deeplearning`, see mnist_classifier.py.

### C++(CUDA)
To use the library for C++ code, link the necessary libraries as below.

```bash
nvcc examples/cpp/mnist_classifier.cpp -Iinclude/deepl -Lbuild/ -ldeeplearning_cpp -lcudart -lcublas -o example/cpp
```

---

## Implementation Details
### Tensors
- Tensors are used as storage objects.
- Tensors can store data on both CPU and GPU devices, depending on the selected device.
- Support for low-level operations:
  - `reshape`: Change the shape of the tensor without altering its data.
  - `add`: Perform element-wise addition of two tensors.
  - `elementwise_multiply`: Perform element-wise multiplication of two tensors.
  - `matmul`: Perform matrix multiplication between tensors.
  - `transpose`: Transpose the dimensions of a tensor.
  - `divide`: Perform element-wise division of two tensors.
  - `exp`: Compute the exponential of each element in the tensor.
  - `binaralize`: Convert tensor elements to binary values (e.g., thresholding).
  - `neg`: Negate the elements of a tensor.
  - `log`: Compute the natural logarithm of each element in the tensor.
- Support for advanced operations:
  - `sumAlongAxis`: Sum tensor elements along a specified axis.
  - `softmax`: Apply the softmax function to normalize tensor values.
  - `batchMatmul`: Perform batch matrix multiplication for tensors with batch dimensions.
- All operations are supported on both GPU and CPU devices.
- Operations such as matrix multiplication (`matmul`), addition, subtraction, and negation leverage CuBLAS for GPU-accelerated computation, enhancing the framework's speed and efficiency by utilizing GPU architecture effectively.

### Computational Graph
- Consists of graph nodes for storing data during the forward pass and adjoints during the backward pass.
- Provides wrappers for all tensor operations.
- Adjoint nodes store adjoint values and dependent nodes along with their partial derivatives.
- During the backward pass, each adjoint node processes the gradient in topological order, propagating the gradients backward.

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
import deepl as dl
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


transform = transforms.Compose([
    transforms.ToTensor()
])  

config = dl.Config.get_instance()

config.set_device_type('GPU')
config.set_cuda_devices('0')
config.set_batch_size(32)
config.set_num_epochs(50)

batch_size = config.get_batch_size()
num_epochs = config.get_num_epochs()
num_classes = 10

# Download and transform the MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

builder = dl.GraphBuilder()
model = dl.Sequential(builder)
model.add_layer(dl.Linear(28 * 28, 256, builder, layer_num=1))  # Input: 784, Output: 128
model.add_layer(dl.ReLU(builder, layer_num=2))
model.add_layer(dl.Linear(256, 128, builder, layer_num=3))
model.add_layer(dl.ReLU(builder, layer_num=4))
model.add_layer(dl.Linear(128, 10, builder, layer_num=5))

loss_fn = dl.CrossEntropyLoss(builder)  # Loss function
parameters = model.parameters()  # Get model parameters
optimizer = dl.SGD(parameters, learning_rate=0.001)  # Optimizer

for epoch in range(num_epochs):
    total_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        # Flatten the images to match the input dimension (batch_size, 28*28)
        images = images.view(images.size(0), -1).numpy().astype(np.float32)  # Convert to numpy array
        labels = labels.numpy().astype(int)  # Convert labels to numpy array

        # One-hot encode the labels
        labels = np.eye(num_classes)[labels].astype(np.float32)

        # Convert to Tensor
        input_tensor = dl.Tensor(images, False)
        target_tensor = dl.Tensor(labels, False)

        input_node = builder.createVariable("input", input_tensor.transpose())
        target_node = builder.createVariable("target", target_tensor.transpose())

        # Forward pass
        outputs = model.forward(input_node)

        # Compute loss
        loss = loss_fn.forward(outputs, target_node)

        # Backward pass
        builder.backward(loss)

        optimizer.step()

        # Accumulate loss
        total_loss += loss.value().get_data()[0]

        print(f"Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx+1}, Loss: {loss.value().get_data()[0]}")
        	
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")
```

---

## Project Structure
```
deepL\
|   .gitignore
|   CMakeLists.txt
|   dependencies.yml
|   readme.md
|
+---bindings
|   \---python
|           deeplearning.cpp
|
+---examples
|   \---python
|           gradmtest.py
|           mnist_classifier.py
|           trans_test.py
|   \---cpp
|           mnist_classifier.cpp_
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

## Task List
- [x] Implement Tensor class and operations.
- [x] Add Python bindings.
- [x] Create example neural networks.
- [x] Expand documentation.
- [x] Optimize GPU implementations.
- [x] Add loss functions and optimizers.

---
