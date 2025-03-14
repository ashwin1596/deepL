#include <iostream>

#include "Graph.h"
#include "Layer.h"
#include "Optimizer.h"
#include "Loss.h"
#include "Utils.h"

int main() {
	auto& config = Config::getInstance();
	size_t batch_size = config.getBatchSize();

	GraphBuilder builder;

	// Create model, optimizer, and loss function as before
	Sequential model(builder);
	model.addLayer(std::make_shared<Linear>(784, 256, builder, 1));
	model.addLayer(std::make_shared<ReLU>(builder, 1));
	model.addLayer(std::make_shared<Linear>(256, 128, builder, 2));
	model.addLayer(std::make_shared<ReLU>(builder, 2));
	model.addLayer(std::make_shared<Linear>(128, 10, builder, 3));

	SGD optimizer(model.parameters(), 0.001);
	CrossEntropyLoss loss_fn(builder);

	size_t total_batches = train_loader.totalBatches(); // Total number of batches in the dataset

	int totalNumEpochs = config.getNumEpochs();
	// Training loop
	for (int epoch = 0; epoch < totalNumEpochs; ++epoch) {
		std::cout << "Epoch " << epoch + 1 << " starting...\n";

		int batch_count = 0;
		for (const auto& batch : train_loader) // Iterator-based batch processing
		{
			// Check if batch is invalid (no more data)
			if (!batch.inputs || !batch.targets) {
				break; // Exit the loop
			}

			batch_count++;

			optimizer.zero_grad();

			NodePtr input = builder.createVariable("input", batch.inputs->transpose());

			// same with the target, following the inputsizexbatchsize
			NodePtr target = builder.createVariable("target", batch.targets);

			// Forward pass
			NodePtr logits = model.forward(input);

			NodePtr loss = loss_fn.forward(logits, target);

			// Backward pass
			builder.backward(loss);

			// Update parameters
			optimizer.step();

			std::cout << "Epoch: " << epoch + 1
				<< "/" << totalNumEpochs
				<< ", Batch: " << batch_count
				<< "/" << total_batches
				<< ", Loss: " << (*loss->value()->data)[0] << "\n";

		}

		std::cout << "Epoch " << epoch + 1 << " completed.\n";
	}

	return 0;
}