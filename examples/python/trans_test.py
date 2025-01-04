import deepl as dl
import numpy as np

config = dl.Config.get_instance()

config.set_device_type('GPU')
config.set_cuda_devices('0')
config.set_batch_size(1)
config.set_num_epochs(10)

builder = dl.GraphBuilder()

def test_tensor_transpose():
    # Create a Tensor with some sample data

    data = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

    # Initialize the tensor
    original_tensor = dl.Tensor(data, False)
    transposed_tensor = original_tensor.transpose()
    input_node = builder.createVariable("input", transposed_tensor)

    input_node.value().print()

if __name__ == "__main__":
    test_tensor_transpose()
