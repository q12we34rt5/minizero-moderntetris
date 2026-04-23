#pragma once

#include "environment.h"
#include "paralleler.h"
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace minizero::learner {

class BaseBatchDataPtr {
public:
    BaseBatchDataPtr() {}
    virtual ~BaseBatchDataPtr() = default;
};

class BatchDataPtr : public BaseBatchDataPtr {
public:
    BatchDataPtr() {}
    virtual ~BatchDataPtr() = default;

    float* features_;
    float* action_features_;
    float* policy_;
    float* value_;
    float* reward_;
    float* loss_scale_;
    int* sampled_index_;

#if MODERNTETRIS_PLACEMENT
    // Placement training buffers. `features_` holds board features [B,1,H,W];
    // `policy_` holds padded pi_target [B, N_max]; `value_` / `loss_scale_` /
    // `sampled_index_` are reused as in AlphaZero.
    int64_t* placement_current_piece_ = nullptr;   // [B]
    int64_t* placement_hold_piece_ = nullptr;      // [B]
    float* placement_has_held_ = nullptr;          // [B]
    int64_t* placement_preview_ = nullptr;         // [B * preview_size]
    float* placement_was_rotation_ = nullptr;      // [B]
    int64_t* placement_srs_index_ = nullptr;       // [B]
    float* placement_lifetime_ = nullptr;          // [B]
    float* placement_combo_ = nullptr;             // [B]
    float* placement_back_to_back_ = nullptr;      // [B]
    float* placement_garbage_ = nullptr;           // [B]
    int64_t* placement_action_use_hold_ = nullptr; // [B * N_max]
    int64_t* placement_action_lock_x_ = nullptr;
    int64_t* placement_action_lock_y_ = nullptr;
    int64_t* placement_action_orientation_ = nullptr;
    int64_t* placement_action_spin_type_ = nullptr;
    int64_t* placement_action_piece_type_ = nullptr;
    int64_t* placement_action_lines_cleared_ = nullptr;
    uint8_t* placement_action_mask_ = nullptr; // [B * N_max], 1 = padded
    int placement_n_max_ = 256;
    int placement_preview_size_ = 5;
#endif
};

#if MODERNTETRIS_PLACEMENT
// Per-env replay snapshot cache. snapshots[k] is the env state right before
// applying action k; size matches the env_loader's data range. Filled lazily
// during sampling: a sample at pos P copies the nearest cached snapshot <= P,
// replays forward through the missing steps, and caches each intermediate
// state so future samples land in a hot slot. The mutex serializes concurrent
// fills on the same env_id; contention is rare because samples spread across
// ~1000s of envs with only learner_num_thread workers.
//
// NOTE: relies on Environment's default copy semantics. ModernTetrisPlacementEnv
// has no user-defined copy ctor; if anyone adds non-copyable members (unique_ptr,
// custom resources), update this cache or copies will silently break.
struct EnvCacheEntry {
    std::mutex mutex;
    std::vector<std::shared_ptr<const Environment>> snapshots;
    explicit EnvCacheEntry(int num_positions) : snapshots(num_positions) {}
};
#endif

class ReplayBuffer {
public:
    ReplayBuffer();

    std::mutex mutex_;
    int num_data_;
    float game_priority_sum_;
    std::deque<float> game_priorities_;
    std::deque<std::deque<float>> position_priorities_;
    std::deque<EnvironmentLoader> env_loaders_;
#if MODERNTETRIS_PLACEMENT
    // Parallel-indexed with env_loaders_; pushed/popped together in addData().
    // unique_ptr because EnvCacheEntry holds a non-movable mutex.
    std::deque<std::unique_ptr<EnvCacheEntry>> env_caches_;
#endif

    void addData(const EnvironmentLoader& env_loader);
    std::pair<int, int> sampleEnvAndPos();
    int sampleIndex(const std::deque<float>& weight);
    float getLossScale(const std::pair<int, int>& p);
};

class DataLoaderSharedData : public utils::BaseSharedData {
public:
    std::string getNextEnvString();
    int getNextBatchIndex();

    virtual void createDataPtr() { data_ptr_ = std::make_shared<BatchDataPtr>(); }
    inline std::shared_ptr<BatchDataPtr> getDataPtr() { return std::static_pointer_cast<BatchDataPtr>(data_ptr_); }

    int batch_index_;
    ReplayBuffer replay_buffer_;
    std::mutex mutex_;
    std::deque<std::string> env_strings_;
    std::shared_ptr<BaseBatchDataPtr> data_ptr_;
};

class DataLoaderThread : public utils::BaseSlaveThread {
public:
    DataLoaderThread(int id, std::shared_ptr<utils::BaseSharedData> shared_data)
        : BaseSlaveThread(id, shared_data) {}

    void initialize() override;
    void runJob() override;
    bool isDone() override { return false; }

protected:
    virtual bool addEnvironmentLoader();
    virtual bool sampleData();

    virtual void setAlphaZeroTrainingData(int batch_index);
    virtual void setMuZeroTrainingData(int batch_index);
#if MODERNTETRIS_PLACEMENT
    virtual void setPlacementTrainingData(int batch_index);
#endif

    inline std::shared_ptr<DataLoaderSharedData> getSharedData() { return std::static_pointer_cast<DataLoaderSharedData>(shared_data_); }
};

class DataLoader : public utils::BaseParalleler {
public:
    DataLoader(const std::string& conf_file_name);

    void initialize() override;
    void summarize() override {}
    virtual void loadDataFromFile(const std::string& file_name);
    virtual void sampleData();
    virtual void updatePriority(int* sampled_index, float* batch_values);

    void createSharedData() override { shared_data_ = std::make_shared<DataLoaderSharedData>(); }
    std::shared_ptr<utils::BaseSlaveThread> newSlaveThread(int id) override { return std::make_shared<DataLoaderThread>(id, shared_data_); }
    inline std::shared_ptr<DataLoaderSharedData> getSharedData() { return std::static_pointer_cast<DataLoaderSharedData>(shared_data_); }
};

} // namespace minizero::learner
