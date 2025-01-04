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

### C++
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
import deeplearning as dl
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
- [x] Expand documentation.
- [x] Optimize GPU implementations.
- [ ] Add more loss functions and optimizers.

---
