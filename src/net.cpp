//Forward funciton describing

#include <torch/torch.h>
#include <net.h>

Net::Net(int64_t num_classes) 
: fully_connected(7*7*32, num_classes) {
    register_module("layer 1", layer1);
    register_module("layer 2", layer2);
    register_module("fc", fully_connected);
}

torch::Tensor Net::forward(torch::Tensor x) {
    x = layer1 -> forward(x);
    x = layer2 -> forward(x);
    x = x.view({-1, 7*7*32});
    return fc->forward(x);
}