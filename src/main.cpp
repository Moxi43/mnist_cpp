#include <torch/torch.h>

#include <iostream>
#include <string>
#include <vector>

namespace F = torch::nn::functional;

// Dataset path
const char *dataset_path = "data/";

// Batch size for training
const int64_t train_batch_size = 64;

// Batch size for testing
const int64_t test_batch_size = 1000;

// The number of epochs for training
const int64_t epochs = 10;

// Update the loss value after that number of batchs
const int64_t log_interval = 10;

template <typename DataLoader>
void train(size_t epoch, Net& model_arch, torch::Device device,
           DataLoader& data_loader, torch::optim::Optimizer& optimizer, 
           size_t dataset_size)
{
    model_arch.train();
    size_t batch_idx = 0;
    for (auto &batch : data_loader)
    {
        //data = input, label
        auto data = batch.data.to(device), targets = batch.target.to(device);

        optimizer.zero_grad();
        auto output = model.forward(data);
        auto loss = F::nil_loss(output, targets);

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


template <typename DataLoader>
void test(Net& model_arch, torch::Device device, DataLoader& data_loader,
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

        test_loss += F::nil_loss(output, targets, {}, 
            torch::Reduction::Sum).template item<float>();
        auto pred = output.argmax(1);
        correct += pred.eq(targets).sum().template item<int64_t>();
    }

    test_loss /= dataset_size;
    std::cout << "Test set: " << "average loss: " << test_loss << " | " << " accuracy: " 
            << static_cast<double>(correct) / dataset_size << std::endl;
}