#pragma once

#include "network.h"
#include "utils.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace minizero::network {

// Sentinel used in the Python embedding table for "no piece" slots.
// PieceQueueEmbed uses an Embedding of size 8 (7 real + NONE=7).
constexpr int kPlacementNonePieceIndex = 7;

// Safe upper bound on the number of legal placements per state.
// Used to size MCTS tree memory pools that expect a fixed action cap.
// Real-world max is well under this (realistic ~80 including hold branch).
constexpr int kPlacementActionUpperBound = 256;

struct PlacementNetworkInput {
    // Board features (C_board * H * W floats).
    std::vector<float> board_features;
    int board_channels = 1;
    int board_height = 20;
    int board_width = 10;

    // Global state (raw values; normalization/one-hot done by the network).
    int current_piece;        // 0..6, -1 if NONE (will be mapped to 7)
    int hold_piece;           // 0..6, -1 if NONE (will be mapped to 7)
    int has_held;             // 0/1
    std::vector<int> preview; // each 0..6, -1 if NONE -> 7
    int was_rotation;         // 0/1
    int srs_index;            // shifted: original (-1..5) + 1 -> 0..6
    float combo_scaled;       // precomputed scaling on C++ side
    int back_to_back;         // 0/1
    float garbage_scaled;     // precomputed scaling on C++ side

    // Per-placement action descriptors (length N).
    std::vector<int> action_use_hold;
    std::vector<int> action_lock_x;
    std::vector<int> action_lock_y;
    std::vector<int> action_orientation;
    std::vector<int> action_spin_type;
    std::vector<int> action_piece_type;
    std::vector<int> action_lines_cleared;
    // Flattened [N, afterstate_feature_size] summary of the board after each
    // placement. Empty, with afterstate_feature_size == 0, when the env has the
    // feature switched off -- in that case forward() omits the tensor entirely
    // so the TorchScript signature stays the one older models were traced with.
    std::vector<float> action_afterstate;
    int afterstate_feature_size = 0;
};

// Helper that builds a PlacementNetworkInput from a placement env. Templated
// so this header need not include the env header; the caller must pass an
// env that exposes getBoardFeatures() / getGlobalFeatures() / getActionDescriptors().
template <class Env>
inline PlacementNetworkInput buildPlacementNetworkInput(const Env& env,
                                                        int board_channels,
                                                        int board_height,
                                                        int board_width)
{
    PlacementNetworkInput in;
    in.board_features = env.getBoardFeatures();
    in.board_channels = board_channels;
    in.board_height = board_height;
    in.board_width = board_width;
    auto g = env.getGlobalFeatures();
    in.current_piece = g.current_piece;
    in.hold_piece = g.hold_piece;
    in.has_held = g.has_held ? 1 : 0;
    in.preview = g.preview;
    in.was_rotation = g.was_rotation ? 1 : 0;
    in.srs_index = std::clamp(g.srs_index + 1, 0, 6);
    in.combo_scaled = std::clamp((g.combo_count + 1) / 10.0f, 0.0f, 1.0f);
    in.back_to_back = g.back_to_back ? 1 : 0;
    in.garbage_scaled = std::clamp(g.pending_garbage / 20.0f, 0.0f, 1.0f);
    auto descs = env.getActionDescriptors();
    const size_t n = descs.size();
    in.afterstate_feature_size = env.getAfterstateFeatureSize();
    if (in.afterstate_feature_size > 0) { in.action_afterstate.reserve(n * in.afterstate_feature_size); }
    in.action_use_hold.reserve(n);
    in.action_lock_x.reserve(n);
    in.action_lock_y.reserve(n);
    in.action_orientation.reserve(n);
    in.action_spin_type.reserve(n);
    in.action_piece_type.reserve(n);
    in.action_lines_cleared.reserve(n);
    for (const auto& d : descs) {
        in.action_use_hold.push_back(d.use_hold ? 1 : 0);
        in.action_lock_x.push_back(d.lock_x);
        in.action_lock_y.push_back(d.lock_y);
        in.action_orientation.push_back(d.orientation);
        in.action_spin_type.push_back(d.spin_type);
        in.action_piece_type.push_back(d.piece_type);
        in.action_lines_cleared.push_back(d.lines_cleared);
        if (in.afterstate_feature_size > 0) {
            in.action_afterstate.insert(in.action_afterstate.end(),
                                        d.afterstate.begin(),
                                        d.afterstate.begin() + in.afterstate_feature_size);
        }
    }
    return in;
}

class PlacementNetworkOutput : public NetworkOutput {
public:
    float value_;
    std::vector<float> policy_;        // size N (legal placements)
    std::vector<float> policy_logits_; // size N

    explicit PlacementNetworkOutput(int n)
    {
        value_ = 0.0f;
        policy_.resize(n, 0.0f);
        policy_logits_.resize(n, 0.0f);
    }
};

class PlacementTransformerNetwork : public Network {
public:
    PlacementTransformerNetwork() { clear(); }

    // We don't call Network::loadModel() because the TorchScript module
    // does not expose all the fixed-shape getters the base class expects.
    void loadModel(const std::string& nn_file_name, const int gpu_id) override
    {
        assert(batch_size_ == 0);
        gpu_id_ = gpu_id;
        network_file_name_ = nn_file_name;
        try {
            network_ = torch::jit::load(network_file_name_, getDevice());
            network_.eval();
        } catch (const c10::Error& e) {
            std::cerr << e.msg() << std::endl;
            assert(false);
        }
        std::vector<torch::jit::IValue> dummy;
        game_name_ = network_.get_method("get_game_name")(dummy).toString()->string();
        network_type_name_ = network_.get_method("get_type_name")(dummy).toString()->string();
        discrete_value_size_ = network_.get_method("get_discrete_value_size")(dummy).toInt();
        // Placement action space is variable N per state. These fields must not be
        // -1 because callers multiply them into tree-sizing arithmetic (overflow ->
        // bad_alloc). Set to safe upper bounds.
        action_size_ = kPlacementActionUpperBound;
        num_input_channels_ = 1;
        input_channel_height_ = 20;
        input_channel_width_ = 10;
        num_hidden_channels_ = 0;
        hidden_channel_height_ = 0;
        hidden_channel_width_ = 0;
        num_blocks_ = 0;
        num_value_hidden_channels_ = 0;
        clear();
    }

    std::string toString() const override
    {
        std::ostringstream oss;
        oss << "GPU ID: " << gpu_id_ << std::endl;
        oss << "Game name: " << game_name_ << std::endl;
        oss << "Network type name: " << network_type_name_ << std::endl;
        oss << "Discrete value size: " << discrete_value_size_ << std::endl;
        oss << "Network file name: " << network_file_name_ << std::endl;
        return oss.str();
    }

    int pushBack(PlacementNetworkInput input)
    {
        assert(batch_size_ < kReserved_batch_size);
        int index;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            index = batch_size_++;
            batch_inputs_.resize(batch_size_);
        }
        batch_inputs_[index] = std::move(input);
        return index;
    }

    std::vector<std::shared_ptr<NetworkOutput>> forward()
    {
        assert(batch_size_ > 0);
        const int B = batch_size_;
        // Infer N_max, preview_size, and board shape from batched inputs.
        int n_max = 0;
        for (const auto& in : batch_inputs_) {
            n_max = std::max(n_max, static_cast<int>(in.action_use_hold.size()));
        }
        if (n_max == 0) { n_max = 1; } // guard: at least one action slot
        const int preview_size = static_cast<int>(batch_inputs_.front().preview.size());
        const int C = batch_inputs_.front().board_channels;
        const int H = batch_inputs_.front().board_height;
        const int W = batch_inputs_.front().board_width;
        const int F = batch_inputs_.front().afterstate_feature_size;

        // Build per-batch tensors. Fill flat backing storage, then copy it into
        // the tensors via make_tensor() below.
        std::vector<float> board_buf(B * C * H * W, 0.0f);
        std::vector<int64_t> current_buf(B, 0);
        std::vector<int64_t> hold_buf(B, 0);
        std::vector<float> has_held_buf(B, 0.0f);
        std::vector<int64_t> preview_buf(B * preview_size, 0);
        std::vector<float> was_rotation_buf(B, 0.0f);
        std::vector<int64_t> srs_index_buf(B, 0);
        std::vector<float> combo_buf(B, 0.0f);
        std::vector<float> b2b_buf(B, 0.0f);
        std::vector<float> garbage_buf(B, 0.0f);
        std::vector<int64_t> a_use_hold_buf(B * n_max, 0);
        std::vector<int64_t> a_lock_x_buf(B * n_max, 0);
        std::vector<int64_t> a_lock_y_buf(B * n_max, 0);
        std::vector<int64_t> a_orient_buf(B * n_max, 0);
        std::vector<int64_t> a_spin_buf(B * n_max, 0);
        std::vector<int64_t> a_piece_buf(B * n_max, 0);
        std::vector<int64_t> a_clear_buf(B * n_max, 0);
        std::vector<uint8_t> action_mask_buf(B * n_max, 1); // default = padded
        std::vector<float> a_afterstate_buf(F > 0 ? static_cast<size_t>(B) * n_max * F : 0, 0.0f);

        auto normalizePieceIndex = [](int p) -> int64_t {
            if (p < 0 || p >= kPlacementNonePieceIndex) { return kPlacementNonePieceIndex; }
            return static_cast<int64_t>(p);
        };

        for (int i = 0; i < B; ++i) {
            const auto& in = batch_inputs_[i];
            assert(static_cast<int>(in.board_features.size()) == C * H * W);
            std::copy(in.board_features.begin(), in.board_features.end(),
                      board_buf.begin() + i * C * H * W);
            current_buf[i] = normalizePieceIndex(in.current_piece);
            hold_buf[i] = normalizePieceIndex(in.hold_piece);
            has_held_buf[i] = static_cast<float>(in.has_held);
            for (int p = 0; p < preview_size; ++p) {
                const int pv = (p < static_cast<int>(in.preview.size())) ? in.preview[p] : -1;
                preview_buf[i * preview_size + p] = normalizePieceIndex(pv);
            }
            was_rotation_buf[i] = static_cast<float>(in.was_rotation);
            srs_index_buf[i] = static_cast<int64_t>(std::clamp(in.srs_index, 0, 6));
            combo_buf[i] = in.combo_scaled;
            b2b_buf[i] = static_cast<float>(in.back_to_back);
            garbage_buf[i] = in.garbage_scaled;

            const int n = static_cast<int>(in.action_use_hold.size());
            for (int a = 0; a < n; ++a) {
                const int off = i * n_max + a;
                a_use_hold_buf[off] = in.action_use_hold[a];
                a_lock_x_buf[off] = in.action_lock_x[a];
                a_lock_y_buf[off] = in.action_lock_y[a];
                a_orient_buf[off] = in.action_orientation[a];
                a_spin_buf[off] = in.action_spin_type[a];
                a_piece_buf[off] = in.action_piece_type[a];
                a_clear_buf[off] = in.action_lines_cleared[a];
                action_mask_buf[off] = 0; // valid
            }
            if (F > 0) {
                assert(static_cast<int>(in.afterstate_feature_size) == F);
                std::copy(in.action_afterstate.begin(), in.action_afterstate.end(),
                          a_afterstate_buf.begin() + static_cast<size_t>(i) * n_max * F);
            }
        }

        auto opts_long = torch::TensorOptions().dtype(torch::kLong);
        auto opts_float = torch::TensorOptions().dtype(torch::kFloat32);
        auto opts_bool = torch::TensorOptions().dtype(torch::kBool);

        // Build each batch tensor with an explicit serial copy rather than
        // from_blob(...).clone().
        //
        // clone() goes through TensorIterator, which hands any copy larger than
        // at::internal::GRAIN_SIZE (32768 elements) to the intra-op thread pool.
        // That pool is sized to the machine (24 threads on a 48-core box) and its
        // workers spin before parking, so on a loaded machine one oversized clone
        // costs tens of milliseconds of wall time and leaves the whole pool
        // burning CPU between calls -- measured here at 22 ms per clone versus
        // 11 us for the same copy done serially. Every tensor below used to sit
        // under the threshold by luck; B * n_max * afterstate_feature_size is the
        // first that does not, and B * n_max crosses it too once
        // zero_num_parallel_games and the legal-placement count are both large.
        auto make_tensor = [](const auto& buf, c10::IntArrayRef shape, const torch::TensorOptions& opts) {
            auto tensor = torch::empty(shape, opts);
            assert(tensor.nbytes() == buf.size() * sizeof(buf[0]));
            std::memcpy(tensor.data_ptr(), buf.data(), buf.size() * sizeof(buf[0]));
            return tensor;
        };

        auto board_t = make_tensor(board_buf, {B, C, H, W}, opts_float);
        auto current_t = make_tensor(current_buf, {B}, opts_long);
        auto hold_t = make_tensor(hold_buf, {B}, opts_long);
        auto has_held_t = make_tensor(has_held_buf, {B}, opts_float);
        auto preview_t = make_tensor(preview_buf, {B, preview_size}, opts_long);
        auto was_rotation_t = make_tensor(was_rotation_buf, {B}, opts_float);
        auto srs_index_t = make_tensor(srs_index_buf, {B}, opts_long);
        auto combo_t = make_tensor(combo_buf, {B}, opts_float);
        auto b2b_t = make_tensor(b2b_buf, {B}, opts_float);
        auto garbage_t = make_tensor(garbage_buf, {B}, opts_float);
        auto a_use_hold_t = make_tensor(a_use_hold_buf, {B, n_max}, opts_long);
        auto a_lock_x_t = make_tensor(a_lock_x_buf, {B, n_max}, opts_long);
        auto a_lock_y_t = make_tensor(a_lock_y_buf, {B, n_max}, opts_long);
        auto a_orient_t = make_tensor(a_orient_buf, {B, n_max}, opts_long);
        auto a_spin_t = make_tensor(a_spin_buf, {B, n_max}, opts_long);
        auto a_piece_t = make_tensor(a_piece_buf, {B, n_max}, opts_long);
        auto a_clear_t = make_tensor(a_clear_buf, {B, n_max}, opts_long);
        auto mask_t = make_tensor(action_mask_buf, {B, n_max}, opts_bool);

        auto dev = getDevice();
        std::vector<torch::jit::IValue> args{
            board_t.to(dev),
            current_t.to(dev),
            hold_t.to(dev),
            has_held_t.to(dev),
            preview_t.to(dev),
            was_rotation_t.to(dev),
            srs_index_t.to(dev),
            combo_t.to(dev),
            b2b_t.to(dev),
            garbage_t.to(dev),
            a_use_hold_t.to(dev),
            a_lock_x_t.to(dev),
            a_lock_y_t.to(dev),
            a_orient_t.to(dev),
            a_spin_t.to(dev),
            a_piece_t.to(dev),
            a_clear_t.to(dev),
            mask_t.to(dev),
        };
        if (F > 0) {
            auto a_afterstate_t = make_tensor(a_afterstate_buf, {B, n_max, F}, opts_float);
            args.push_back(a_afterstate_t.to(dev));
        }

        auto result = network_.forward(args).toGenericDict();
        auto policy_output = result.at("policy").toTensor().to(at::kCPU).contiguous();
        auto policy_logits_output = result.at("policy_logit").toTensor().to(at::kCPU).contiguous();
        auto value_output = result.at("value").toTensor().to(at::kCPU).contiguous();
        assert(policy_output.size(0) == B && policy_output.size(1) == n_max);

        std::vector<std::shared_ptr<NetworkOutput>> outputs;
        outputs.reserve(B);
        for (int i = 0; i < B; ++i) {
            const int n = static_cast<int>(batch_inputs_[i].action_use_hold.size());
            auto out = std::make_shared<PlacementNetworkOutput>(n);
            const float* policy_row = policy_output.data_ptr<float>() + i * n_max;
            const float* logits_row = policy_logits_output.data_ptr<float>() + i * n_max;
            std::copy(policy_row, policy_row + n, out->policy_.begin());
            std::copy(logits_row, logits_row + n, out->policy_logits_.begin());

            if (discrete_value_size_ == 1) {
                out->value_ = value_output[i].item<float>();
            } else {
                int start_value = -discrete_value_size_ / 2;
                const float* vrow = value_output.data_ptr<float>() + i * discrete_value_size_;
                float v = 0.0f;
                for (int k = 0; k < discrete_value_size_; ++k) {
                    v += vrow[k] * static_cast<float>(start_value + k);
                }
                out->value_ = utils::invertValue(v);
            }
            outputs.push_back(std::move(out));
        }

        clear();
        return outputs;
    }

    inline int getBatchSize() const { return batch_size_; }

protected:
    inline void clear()
    {
        batch_size_ = 0;
        batch_inputs_.clear();
        batch_inputs_.reserve(kReserved_batch_size);
    }

    int batch_size_ = 0;
    std::mutex mutex_;
    std::vector<PlacementNetworkInput> batch_inputs_;

    const int kReserved_batch_size = 4096;
};

} // namespace minizero::network
