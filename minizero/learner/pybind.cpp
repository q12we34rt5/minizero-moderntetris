#include "configuration.h"
#include "data_loader.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

namespace py = pybind11;
using namespace minizero;

std::shared_ptr<Environment> kEnvInstance;

Environment& getEnvInstance()
{
    if (!kEnvInstance) { kEnvInstance = std::make_shared<Environment>(); }
    return *kEnvInstance;
}

PYBIND11_MODULE(minizero_py, m)
{
    m.def("load_config_file", [](std::string file_name) {
        minizero::env::setUpEnv();
        minizero::config::ConfigureLoader cl;
        minizero::config::setConfiguration(cl);
        bool success = cl.loadFromFile(file_name);
        if (success) { kEnvInstance = std::make_shared<Environment>(); }
        return success;
    });
    m.def("load_config_string", [](std::string conf_str) {
        minizero::config::ConfigureLoader cl;
        minizero::config::setConfiguration(cl);
        bool success = cl.loadFromString(conf_str);
        if (success) { kEnvInstance = std::make_shared<Environment>(); }
        return success;
    });
    m.def("use_gumbel", []() { return config::actor_use_gumbel; });
    m.def("get_zero_replay_buffer", []() { return config::zero_replay_buffer; });
    m.def("use_per", []() { return config::learner_use_per; });
    m.def("get_training_step", []() { return config::learner_training_step; });
    m.def("get_training_display_step", []() { return config::learner_training_display_step; });
    m.def("get_batch_size", []() { return config::learner_batch_size; });
    m.def("get_muzero_unrolling_step", []() { return config::learner_muzero_unrolling_step; });
    m.def("get_n_step_return", []() { return config::learner_n_step_return; });
    m.def("get_optimizer", []() { return config::learner_optimizer; });
    m.def("get_learning_rate", []() { return config::learner_learning_rate; });
    m.def("get_momentum", []() { return config::learner_momentum; });
    m.def("get_weight_decay", []() { return config::learner_weight_decay; });
    m.def("get_value_loss_scale", []() { return config::learner_value_loss_scale; });
    m.def("get_game_name", []() { return getEnvInstance().name(); });
    m.def("get_nn_num_input_channels", []() { return getEnvInstance().getNumInputChannels(); });
    m.def("get_nn_input_channel_height", []() { return getEnvInstance().getInputChannelHeight(); });
    m.def("get_nn_input_channel_width", []() { return getEnvInstance().getInputChannelWidth(); });
    m.def("get_nn_num_hidden_channels", []() { return config::nn_num_hidden_channels; });
    m.def("get_nn_hidden_channel_height", []() { return getEnvInstance().getHiddenChannelHeight(); });
    m.def("get_nn_hidden_channel_width", []() { return getEnvInstance().getHiddenChannelWidth(); });
    m.def("get_nn_num_action_feature_channels", []() { return getEnvInstance().getNumActionFeatureChannels(); });
    m.def("get_nn_num_blocks", []() { return config::nn_num_blocks; });
    m.def("get_nn_action_size", []() { return getEnvInstance().getPolicySize(); });
    m.def("get_nn_num_value_hidden_channels", []() { return config::nn_num_value_hidden_channels; });
    m.def("get_nn_discrete_value_size", []() { return kEnvInstance->getDiscreteValueSize(); });
    m.def("get_nn_type_name", []() { return config::nn_type_name; });
    m.def("get_nn_placement_d_model", []() { return config::nn_placement_d_model; });
    m.def("get_nn_placement_n_layers", []() { return config::nn_placement_n_layers; });
    m.def("get_nn_placement_n_heads", []() { return config::nn_placement_n_heads; });
    m.def("get_nn_placement_mlp_ratio", []() { return config::nn_placement_mlp_ratio; });
    m.def("get_nn_placement_dropout", []() { return config::nn_placement_dropout; });
    m.def("get_env_modern_tetris_num_preview_piece", []() { return config::env_modern_tetris_num_preview_piece; });

    py::class_<learner::DataLoader>(m, "DataLoader")
        .def(py::init<std::string>())
        .def("initialize", &learner::DataLoader::initialize)
        .def("load_data_from_file", &learner::DataLoader::loadDataFromFile, py::call_guard<py::gil_scoped_release>())
        .def(
            "update_priority", [](learner::DataLoader& data_loader, py::array_t<int>& sampled_index, py::array_t<float>& batch_values) {
                data_loader.updatePriority(static_cast<int*>(sampled_index.request().ptr), static_cast<float*>(batch_values.request().ptr));
            },
            py::call_guard<py::gil_scoped_release>())
        .def(
            "sample_data", [](learner::DataLoader& data_loader, py::array_t<float>& features, py::array_t<float>& action_features, py::array_t<float>& policy, py::array_t<float>& value, py::array_t<float>& reward, py::array_t<float>& loss_scale, py::array_t<int>& sampled_index) {
                data_loader.getSharedData()->getDataPtr()->features_ = static_cast<float*>(features.request().ptr);
                data_loader.getSharedData()->getDataPtr()->action_features_ = static_cast<float*>(action_features.request().ptr);
                data_loader.getSharedData()->getDataPtr()->policy_ = static_cast<float*>(policy.request().ptr);
                data_loader.getSharedData()->getDataPtr()->value_ = static_cast<float*>(value.request().ptr);
                data_loader.getSharedData()->getDataPtr()->reward_ = static_cast<float*>(reward.request().ptr);
                data_loader.getSharedData()->getDataPtr()->loss_scale_ = static_cast<float*>(loss_scale.request().ptr);
                data_loader.getSharedData()->getDataPtr()->sampled_index_ = static_cast<int*>(sampled_index.request().ptr);
                data_loader.sampleData(); }, py::call_guard<py::gil_scoped_release>())
#if MODERNTETRIS_PLACEMENT
        // Mirrors network::kPlacementActionUpperBound (kept inline to avoid dragging torch into pybind TU).
        .def("placement_n_max", []() { return 256; })
        .def(
            "sample_data_placement", [](learner::DataLoader& dl, py::array_t<float>& board, py::array_t<float>& policy, py::array_t<float>& value, py::array_t<float>& loss_scale, py::array_t<int>& sampled_index, py::array_t<int64_t>& current_piece, py::array_t<int64_t>& hold_piece, py::array_t<float>& has_held, py::array_t<int64_t>& preview, py::array_t<float>& was_rotation, py::array_t<int64_t>& srs_index, py::array_t<float>& lifetime, py::array_t<float>& combo, py::array_t<float>& back_to_back, py::array_t<float>& garbage, py::array_t<int64_t>& a_use_hold, py::array_t<int64_t>& a_lock_x, py::array_t<int64_t>& a_lock_y, py::array_t<int64_t>& a_orientation, py::array_t<int64_t>& a_spin, py::array_t<int64_t>& a_piece, py::array_t<int64_t>& a_lines, py::array_t<uint8_t>& a_mask, int n_max, int preview_size) {
                auto dp = dl.getSharedData()->getDataPtr();
                dp->features_ = static_cast<float*>(board.request().ptr);
                dp->policy_ = static_cast<float*>(policy.request().ptr);
                dp->value_ = static_cast<float*>(value.request().ptr);
                dp->loss_scale_ = static_cast<float*>(loss_scale.request().ptr);
                dp->sampled_index_ = static_cast<int*>(sampled_index.request().ptr);
                dp->placement_current_piece_ = static_cast<int64_t*>(current_piece.request().ptr);
                dp->placement_hold_piece_ = static_cast<int64_t*>(hold_piece.request().ptr);
                dp->placement_has_held_ = static_cast<float*>(has_held.request().ptr);
                dp->placement_preview_ = static_cast<int64_t*>(preview.request().ptr);
                dp->placement_was_rotation_ = static_cast<float*>(was_rotation.request().ptr);
                dp->placement_srs_index_ = static_cast<int64_t*>(srs_index.request().ptr);
                dp->placement_lifetime_ = static_cast<float*>(lifetime.request().ptr);
                dp->placement_combo_ = static_cast<float*>(combo.request().ptr);
                dp->placement_back_to_back_ = static_cast<float*>(back_to_back.request().ptr);
                dp->placement_garbage_ = static_cast<float*>(garbage.request().ptr);
                dp->placement_action_use_hold_ = static_cast<int64_t*>(a_use_hold.request().ptr);
                dp->placement_action_lock_x_ = static_cast<int64_t*>(a_lock_x.request().ptr);
                dp->placement_action_lock_y_ = static_cast<int64_t*>(a_lock_y.request().ptr);
                dp->placement_action_orientation_ = static_cast<int64_t*>(a_orientation.request().ptr);
                dp->placement_action_spin_type_ = static_cast<int64_t*>(a_spin.request().ptr);
                dp->placement_action_piece_type_ = static_cast<int64_t*>(a_piece.request().ptr);
                dp->placement_action_lines_cleared_ = static_cast<int64_t*>(a_lines.request().ptr);
                dp->placement_action_mask_ = static_cast<uint8_t*>(a_mask.request().ptr);
                dp->placement_n_max_ = n_max;
                dp->placement_preview_size_ = preview_size;
                dl.sampleData(); }, py::call_guard<py::gil_scoped_release>())
#endif
        ;
}
