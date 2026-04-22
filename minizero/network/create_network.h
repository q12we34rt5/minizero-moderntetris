#pragma once

#include "alphazero_network.h"
#include "muzero_network.h"
#include "network.h"
#include "placement_transformer_network.h"
#include <memory>
#include <string>
#include <torch/script.h>
#include <vector>

namespace minizero::network {

inline std::string probeNetworkTypeName(const std::string& nn_file_name)
{
    auto module = torch::jit::load(nn_file_name, torch::Device("cpu"));
    std::vector<torch::jit::IValue> dummy;
    return module.get_method("get_type_name")(dummy).toString()->string();
}

inline std::shared_ptr<Network> createNetwork(const std::string& nn_file_name, const int gpu_id)
{
    const std::string type_name = probeNetworkTypeName(nn_file_name);

    std::shared_ptr<Network> network;
    if (type_name == "alphazero") {
        network = std::make_shared<AlphaZeroNetwork>();
        std::dynamic_pointer_cast<AlphaZeroNetwork>(network)->loadModel(nn_file_name, gpu_id);
    } else if (type_name == "muzero" || type_name == "muzero_atari") {
        network = std::make_shared<MuZeroNetwork>();
        std::dynamic_pointer_cast<MuZeroNetwork>(network)->loadModel(nn_file_name, gpu_id);
    } else if (type_name == "placement_transformer") {
        network = std::make_shared<PlacementTransformerNetwork>();
        std::dynamic_pointer_cast<PlacementTransformerNetwork>(network)->loadModel(nn_file_name, gpu_id);
    } else {
        // should not be here
        assert(false);
    }

    return network;
}

} // namespace minizero::network
