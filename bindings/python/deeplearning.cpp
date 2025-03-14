#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<pybind11/stl.h>

#include "Graph.h"
#include "Layer.h"
#include "Optimizer.h"
#include "Loss.h"
#include "GradMode.hpp"

namespace py = pybind11;

PYBIND11_MODULE(deeplearning, m) {

    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
		.def(py::init([](py::object input, bool requires_grad) {
			if (py::isinstance<py::array>(input)) {
				//Handle NumPy array input
				auto array = input.cast<py::array_t<float>>();
				auto buf = array.request();
				std::vector<size_t> shape(buf.ndim);
				for (size_t i = 0; i < buf.ndim; i++) {
					shape[i] = buf.shape[i];
				}
				std::vector<float> data(static_cast<float*>(buf.ptr),
					static_cast<float*>(buf.ptr) + buf.size);
				return std::make_shared<Tensor>(shape, data, requires_grad);
			}
			else if (py::isinstance<py::float_>(input)) {
				//Handle scalar input
				float scalar = input.cast<float>();
				return std::make_shared<Tensor>(scalar, requires_grad);
			}
			else if (py::isinstance<py::tuple>(input)) {
				//Handle tuple for dimension
				auto dims = input.cast<std::vector<size_t>>();
				return std::make_shared<Tensor>(dims, requires_grad);
			}
			else {
				throw std::runtime_error("Unsupported input type. Expected NumPy array, scalar, or tuple.");
			}
		}), py::arg("data"), py::arg("requires_grad")=false)
		.def("get_data", [](const Tensor& self) {
			// Convert std::vector<float> to Python list
			return py::cast(*self.get_data());
		}, "Get the underlying data as a Python list.")
		.def("get_dims", [](const Tensor& self) {
			// Convert std::vector<size_t> to Python list
			return py::cast(*self.get_dims());
		}, "Get the dimensions of the tensor as a Python list.")
		.def("transpose", &Tensor::transpose)
		.def("print", &Tensor::print);

    py::class_<Config>(m, "Config")
        .def_static("get_instance", &Config::getInstance, py::return_value_policy::reference)
        .def("load_from_env", &Config::loadFromEnv)
        .def("get_device_type", &Config::getDeviceType)
        .def("get_cuda_devices", &Config::getCudaDevices)
        .def("get_batch_size", &Config::getBatchSize)
		.def("get_num_epochs", &Config::getNumEpochs)
        .def("set_device_type", &Config::setDeviceType)
        .def("set_cuda_devices", &Config::setCudaDevices)
        .def("set_batch_size", &Config::setBatchSize)
		.def("set_num_epochs", &Config::setNumEpochs);

	py::enum_<DeviceType>(m, "DeviceType")
		.value("CPU", DeviceType::CPU)
		.value("GPU", DeviceType::GPU)
		.export_values();

	py::class_<DeviceManager>(m, "DeviceManager")
        .def_static("device_type_to_string", &DeviceManager::deviceTypeToString, py::arg("device"),
                    "Convert a DeviceType enum to its string representation.");

	py::class_<ComputationGraph, std::shared_ptr<ComputationGraph>>(m, "ComputationGraph")
	   .def(py::init<>())
	   .def("addNode", &ComputationGraph::addNode)
	   .def("backward", &ComputationGraph::backward);

	py::class_<GraphBuilder, std::shared_ptr<GraphBuilder>>(m, "GraphBuilder")
		.def(py::init<>())
		.def("createVariable", &GraphBuilder::createVariable)
		.def("backward", &GraphBuilder::backward);

	py::class_<Layer, std::shared_ptr<Layer>>(m, "Layer")
		.def("forward", &Layer::forward, py::arg("input"))
		.def("parameters", &Layer::parameters)
		.def("get_layer_type", &Layer::getLayerType)
		.def_readwrite("layer_num", &Layer::layer_num_);

	py::class_<Linear, Layer, std::shared_ptr<Linear>>(m, "Linear")
		.def(py::init<size_t, size_t, GraphBuilder&, int>(),
			 py::arg("in_features"), py::arg("out_features"), py::arg("builder"), py::arg("layer_num"))
		.def("forward", &Linear::forward, py::arg("input"))
		.def("parameters", &Linear::parameters)
		.def("get_layer_type", &Linear::getLayerType);

	py::class_<ReLU, Layer, std::shared_ptr<ReLU>>(m, "ReLU")
	.def(py::init<GraphBuilder&, int>(), py::arg("builder"), py::arg("layer_num"))
	.def("forward", &ReLU::forward, py::arg("input"))
	.def("parameters", &ReLU::parameters)
	.def("get_layer_type", &ReLU::getLayerType);

	py::class_<Sequential, std::shared_ptr<Sequential>>(m, "Sequential")
		.def(py::init<GraphBuilder&>(), py::arg("builder"))
		.def("add_layer", &Sequential::addLayer, py::arg("layer"))
		.def("forward", &Sequential::forward, py::arg("input"))
		.def("parameters", &Sequential::parameters);

		py::class_<Node, std::shared_ptr<Node>>(m, "Node")
		.def(py::init<const std::string&, OpType, TensorPtr>())
		.def("value", &Node::value)
		.def("setValue", &Node::setValue)
		.def("name", &Node::name)
		.def("setName", &Node::setName)
		.def("adjoint", &Node::adjoint)
		.def("setAdjoint", &Node::setAdjoint)
		.def("op_type", &Node::op_type);

	py::class_<Loss, std::shared_ptr<Loss>>(m, "Loss")
	.def("forward", &Loss::forward, py::arg("logits"), py::arg("targets"));

	py::class_<CrossEntropyLoss, Loss, std::shared_ptr<CrossEntropyLoss>>(m, "CrossEntropyLoss")
	   .def(py::init<GraphBuilder&>(), py::arg("builder"))
	   .def("forward", &CrossEntropyLoss::forward, py::arg("logits"), py::arg("targets"));

	py::class_<optimizer, std::shared_ptr<optimizer>>(m, "Optimizer")
		.def("step", &optimizer::step, "Performs an optimization step.")
		.def("zero_grad", &optimizer::zero_grad, "Zeros out gradients of all parameters.");

	// SGD class derived from optimizer
	py::class_<SGD, optimizer, std::shared_ptr<SGD>>(m, "SGD")
		.def(py::init<std::vector<NodePtr>, float>(),
			 py::arg("parameters"),
			 py::arg("learning_rate"),
			 "Constructs an SGD optimizer with the given parameters and learning rate.")
		.def("step", &SGD::step, "Performs a single optimization step using SGD.")
		.def("zero_grad", &SGD::zero_grad, "Zeros out gradients of the parameters.");

	// Bind GradMode
	py::class_<GradMode>(m, "GradMode")
		.def_static("is_enabled", &GradMode::isEnabled, "Check if gradient mode is enabled");

	// Bind NoGradGuard
	py::class_<NoGradGuard>(m, "no_grad")
		.def(py::init<>())
		.def("__enter__", [](NoGradGuard& self) { return &self; })
		.def("__exit__", [](NoGradGuard& self, py::object, py::object, py::object) {});

	// Bind EnableGradGuard
	py::class_<EnableGradGuard>(m, "grad")
		.def(py::init<>())
		.def("__enter__", [](EnableGradGuard& self) { return &self; })
		.def("__exit__", [](EnableGradGuard& self, py::object, py::object, py::object) {});
}
