#include <torch/torch.h>

#include "../includes/net.h"

#include <iostream>
#include <string>
#include <vector>


// Dataset path
const char* dataset_path = "../data/";

// Batch size for training
const int64_t train_batch_size = 64;

// Batch size for testing
const int64_t test_batch_size = 1000;

// The number of epochs for training
const int64_t epochs = 10;

// Update the loss value after that number of batchs
const int64_t log_interval = 10;


// Forward function description

Net::Net(int64_t num_classes) 
: fully_connected(7*7*32, num_classes) {
    register_module("layer 1", layer1);
    register_module("layer 2", layer2);
    register_module("fc", fully_connected);
}

torch::Tensor Net::forward(torch::Tensor x) {
    x = torch::relu(layer1->forward(x));
    x = torch::relu(layer2->forward(x));
    x = x.view({-1, 7*7*32});
    return fully_connected->forward(x);
}

// Train function    

template <typename DataLoader>
void train(size_t epoch, Net& model, torch::Device device,
           DataLoader& data_loader, torch::optim::Optimizer& optimizer,
           size_t dataset_size)
{
    model.train();
    size_t batch_idx = 0;
    for (auto& batch : data_loader)
    {
        //data = input, label
        auto data = batch.data.to(device), targets = batch.target.to(device);

        optimizer.zero_grad();
        auto output = model.forward(data);
        auto loss = torch::nll_loss(output, targets);

        loss.backward();
        optimizer.step();

        //print some logs
        if (batch_idx++ % log_interval == 0)
        {
            std::cout << "Train Epoch: " << epoch << std::endl
                      << "Loss: " << loss.template item<float>() << std::endl;
        }
    }
}

// Test function

template <typename DataLoader>
void test(Net& model, torch::Device device, DataLoader& data_loader,
          size_t dataset_size)
{
    torch::NoGradGuard no_grad;
    model.eval();
    //some metrics
    double test_loss = 0;
    int32_t correct = 0;

    for (const auto& batch : data_loader)
    {
        //data = input, label
        auto data = batch.data.to(device), targets = batch.target.to(device);

        auto output = model.forward(data);

        test_loss += torch::nll_loss(
                     output,
                     targets,
                     /*weight=*/{},
                     torch::Reduction::Sum)
                     .template item<float>();
        auto pred = output.argmax(1);
        correct += pred.eq(targets).sum().template item<int64_t>();
    }

    test_loss /= dataset_size;
    std::cout << "Test set: "
              << "average loss: " << test_loss << " | "
              << " accuracy: "
              << static_cast<double>(correct) / dataset_size << std::endl;
}

int main() 
{
    torch::manual_seed(1);

    // Define device type

    torch::DeviceType device_type;
    if (torch::cuda::is_available())
    {
        std::cout << "USING CUDA. TRANING ON GPU." << std::endl;
        device_type = torch::kCUDA;
    }
    else
    {
        std::cout << "TRAINING ON CPU." << std::endl;
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);

    // Define model architecture
    Net model;

    // Train the model on the selected device
    model.to(device_type);

    // Define train dataset

    auto train_dataset = torch::data::datasets::MNIST(dataset_path)
                             .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                             .map(torch::data::transforms::Stack<>());

    // Define train dataset size

    const size_t train_dataset_size = train_dataset.size().value();

    // Define multi-thread train loader

    auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(train_dataset),
        train_batch_size);

    // Define test dataset

    auto test_dataset = torch::data::datasets::MNIST(
                            dataset_path, torch::data::datasets::MNIST::Mode::kTest)
                            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                            .map(torch::data::transforms::Stack<>());

    // Define test dataset size

    const size_t test_dataset_size = test_dataset.size().value();

    // Define multi-thread test loader

    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(test_dataset),
        test_batch_size);

    // Define SGD optimizer
    torch::optim::SGD optimizer(
        model.parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));

    // Run training and testing

    for (size_t epoch = 1; epoch <= epochs; ++epoch)
    {
        train(epoch, model, device, *train_loader, optimizer, train_dataset_size);
        test(model, device, *test_loader, test_dataset_size);
    }
    
    return 0;
}