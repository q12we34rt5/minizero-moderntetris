#include "data_loader.h"
#include "configuration.h"
#include "environment.h"
#include "random.h"
#include "rotation.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>
#if MODERNTETRIS_PLACEMENT
#include "moderntetris_placement.h"
#endif

namespace minizero::learner {

using namespace minizero;
using namespace minizero::utils;

ReplayBuffer::ReplayBuffer()
{
    num_data_ = 0;
    game_priority_sum_ = 0.0f;
    game_priorities_.clear();
    position_priorities_.clear();
    env_loaders_.clear();
}

void ReplayBuffer::addData(const EnvironmentLoader& env_loader)
{
    std::pair<int, int> data_range = env_loader.getDataRange();
    std::deque<float> position_priorities(data_range.second + 1, 0.0f);
    float game_priority = 0.0f;
    for (int i = data_range.first; i <= data_range.second; ++i) {
        position_priorities[i] = std::pow((config::learner_use_per ? env_loader.getPriority(i) : 1.0f), config::learner_per_alpha);
        game_priority += position_priorities[i];
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // add new data to replay buffer
    num_data_ += (data_range.second - data_range.first + 1);
    position_priorities_.push_back(position_priorities);
    game_priorities_.push_back(game_priority);
    env_loaders_.push_back(env_loader);
#if MODERNTETRIS_PLACEMENT
    // Cache slot covers positions 0..data_range.second (parallel indexing
    // with position_priorities_, which is sized data_range.second + 1).
    env_caches_.push_back(std::make_unique<EnvCacheEntry>(data_range.second + 1));
#endif

    // remove old data if replay buffer is full
    const size_t replay_buffer_max_size = config::zero_replay_buffer * config::zero_num_games_per_iteration;
    while (position_priorities_.size() > replay_buffer_max_size) {
        data_range = env_loaders_.front().getDataRange();
        num_data_ -= (data_range.second - data_range.first + 1);
        position_priorities_.pop_front();
        game_priorities_.pop_front();
        env_loaders_.pop_front();
#if MODERNTETRIS_PLACEMENT
        env_caches_.pop_front();
#endif
    }
}

std::pair<int, int> ReplayBuffer::sampleEnvAndPos()
{
    int env_id = sampleIndex(game_priorities_);
    int pos_id = sampleIndex(position_priorities_[env_id]);
    return {env_id, pos_id};
}

int ReplayBuffer::sampleIndex(const std::deque<float>& weight)
{
    std::discrete_distribution<> dis(weight.begin(), weight.end());
    return dis(Random::generator_);
}

float ReplayBuffer::getLossScale(const std::pair<int, int>& p)
{
    if (!config::learner_use_per) { return 1.0f; }

    // calculate importance sampling ratio
    int env_id = p.first, pos = p.second;
    float prob = position_priorities_[env_id][pos] / game_priority_sum_;
    return std::pow((num_data_ * prob), (-config::learner_per_init_beta));
}

std::string DataLoaderSharedData::getNextEnvString()
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::string env_string = "";
    if (!env_strings_.empty()) {
        env_string = env_strings_.front();
        env_strings_.pop_front();
    }
    return env_string;
}

int DataLoaderSharedData::getNextBatchIndex()
{
    std::lock_guard<std::mutex> lock(mutex_);
    return (batch_index_ < config::learner_batch_size ? batch_index_++ : config::learner_batch_size);
}

void DataLoaderThread::initialize()
{
    int seed = config::program_auto_seed ? std::random_device()() : config::program_seed + id_;
    Random::seed(seed);
}

void DataLoaderThread::runJob()
{
    if (!getSharedData()->env_strings_.empty()) {
        while (addEnvironmentLoader()) {}
    } else {
        while (sampleData()) {}
    }
}

bool DataLoaderThread::addEnvironmentLoader()
{
    std::string env_string = getSharedData()->getNextEnvString();
    if (env_string.empty()) { return false; }

    EnvironmentLoader env_loader;
    if (env_loader.loadFromString(env_string)) { getSharedData()->replay_buffer_.addData(env_loader); }
    return true;
}

bool DataLoaderThread::sampleData()
{
    int batch_index = getSharedData()->getNextBatchIndex();
    if (batch_index >= config::learner_batch_size) { return false; }

    if (config::nn_type_name == "alphazero") {
        setAlphaZeroTrainingData(batch_index);
    } else if (config::nn_type_name == "muzero") {
        setMuZeroTrainingData(batch_index);
#if MODERNTETRIS_PLACEMENT
    } else if (config::nn_type_name == "placement_transformer") {
        setPlacementTrainingData(batch_index);
#endif
    } else {
        return false; // should not be here
    }

    return true;
}

void DataLoaderThread::setAlphaZeroTrainingData(int batch_index)
{
    // random pickup one position
    std::pair<int, int> p = getSharedData()->replay_buffer_.sampleEnvAndPos();
    int env_id = p.first, pos = p.second;

    // AlphaZero training data
    const EnvironmentLoader& env_loader = getSharedData()->replay_buffer_.env_loaders_[env_id];
    Rotation rotation = static_cast<Rotation>(Random::randInt() % static_cast<int>(Rotation::kRotateSize));
    float loss_scale = getSharedData()->replay_buffer_.getLossScale(p);
    std::vector<float> features = env_loader.getFeatures(pos, rotation);
    std::vector<float> policy = env_loader.getPolicy(pos, rotation);
    std::vector<float> value = env_loader.getValue(pos);

    // write data to data_ptr
    getSharedData()->getDataPtr()->loss_scale_[batch_index] = loss_scale;
    getSharedData()->getDataPtr()->sampled_index_[2 * batch_index] = p.first;
    getSharedData()->getDataPtr()->sampled_index_[2 * batch_index + 1] = p.second;
    std::copy(features.begin(), features.end(), getSharedData()->getDataPtr()->features_ + features.size() * batch_index);
    std::copy(policy.begin(), policy.end(), getSharedData()->getDataPtr()->policy_ + policy.size() * batch_index);
    std::copy(value.begin(), value.end(), getSharedData()->getDataPtr()->value_ + value.size() * batch_index);
}

void DataLoaderThread::setMuZeroTrainingData(int batch_index)
{
    // random pickup one position
    std::pair<int, int> p = getSharedData()->replay_buffer_.sampleEnvAndPos();
    int env_id = p.first, pos = p.second;

    // MuZero training data
    const EnvironmentLoader& env_loader = getSharedData()->replay_buffer_.env_loaders_[env_id];
    Rotation rotation = static_cast<Rotation>(Random::randInt() % static_cast<int>(Rotation::kRotateSize));
    float loss_scale = getSharedData()->replay_buffer_.getLossScale(p);
    std::vector<float> features = env_loader.getFeatures(pos, rotation);
    std::vector<float> action_features, policy, value, reward, tmp;
    for (int step = 0; step <= config::learner_muzero_unrolling_step; ++step) {
        // action features
        if (step < config::learner_muzero_unrolling_step) {
            tmp = env_loader.getActionFeatures(pos + step, rotation);
            action_features.insert(action_features.end(), tmp.begin(), tmp.end());
        }

        // policy
        tmp = env_loader.getPolicy(pos + step, rotation);
        policy.insert(policy.end(), tmp.begin(), tmp.end());

        // value
        tmp = env_loader.getValue(pos + step);
        value.insert(value.end(), tmp.begin(), tmp.end());

        // reward
        if (step < config::learner_muzero_unrolling_step) {
            tmp = env_loader.getReward(pos + step);
            reward.insert(reward.end(), tmp.begin(), tmp.end());
        }
    }

    // write data to data_ptr
    getSharedData()->getDataPtr()->loss_scale_[batch_index] = loss_scale;
    getSharedData()->getDataPtr()->sampled_index_[2 * batch_index] = p.first;
    getSharedData()->getDataPtr()->sampled_index_[2 * batch_index + 1] = p.second;
    std::copy(features.begin(), features.end(), getSharedData()->getDataPtr()->features_ + features.size() * batch_index);
    std::copy(action_features.begin(), action_features.end(), getSharedData()->getDataPtr()->action_features_ + action_features.size() * batch_index);
    std::copy(policy.begin(), policy.end(), getSharedData()->getDataPtr()->policy_ + policy.size() * batch_index);
    std::copy(value.begin(), value.end(), getSharedData()->getDataPtr()->value_ + value.size() * batch_index);
    std::copy(reward.begin(), reward.end(), getSharedData()->getDataPtr()->reward_ + reward.size() * batch_index);
}

DataLoader::DataLoader(const std::string& conf_file_name)
{
    env::setUpEnv();
    config::ConfigureLoader cl;
    config::setConfiguration(cl);
    cl.loadFromFile(conf_file_name);
}

void DataLoader::initialize()
{
    createSlaveThreads(config::learner_num_thread);
    getSharedData()->createDataPtr();
}

void DataLoader::loadDataFromFile(const std::string& file_name)
{
    std::ifstream fin(file_name, std::ifstream::in);
    for (std::string content; std::getline(fin, content);) { getSharedData()->env_strings_.push_back(content); }

    for (auto& t : slave_threads_) { t->start(); }
    for (auto& t : slave_threads_) { t->finish(); }
    getSharedData()->replay_buffer_.game_priority_sum_ = std::accumulate(getSharedData()->replay_buffer_.game_priorities_.begin(), getSharedData()->replay_buffer_.game_priorities_.end(), 0.0f);
}

void DataLoader::sampleData()
{
    getSharedData()->batch_index_ = 0;
    for (auto& t : slave_threads_) { t->start(); }
    for (auto& t : slave_threads_) { t->finish(); }
}

void DataLoader::updatePriority(int* sampled_index, float* batch_values)
{
    // TODO: use multiple threads
    for (int batch_index = 0; batch_index < config::learner_batch_size; ++batch_index) {
        int env_id = sampled_index[2 * batch_index];
        int pos_id = sampled_index[2 * batch_index + 1];

        EnvironmentLoader& env_loader = getSharedData()->replay_buffer_.env_loaders_[env_id];
        for (int step = 0; step <= config::learner_muzero_unrolling_step; ++step) {
            float new_value = utils::invertValue(batch_values[step * config::learner_batch_size + batch_index]);
            env_loader.setActionPairInfo(pos_id + step, "V", std::to_string(new_value));
        }
        getSharedData()->replay_buffer_.position_priorities_[env_id][pos_id] = std::pow(env_loader.getPriority(pos_id), config::learner_per_alpha);
    }

    // recalculate priority to correct floating number error (TODO: speedup this)
    for (size_t i = 0; i < getSharedData()->replay_buffer_.game_priorities_.size(); ++i) {
        getSharedData()->replay_buffer_.game_priorities_[i] = std::accumulate(getSharedData()->replay_buffer_.position_priorities_[i].begin(), getSharedData()->replay_buffer_.position_priorities_[i].end(), 0.0f);
    }
    getSharedData()->replay_buffer_.game_priority_sum_ = std::accumulate(getSharedData()->replay_buffer_.game_priorities_.begin(), getSharedData()->replay_buffer_.game_priorities_.end(), 0.0f);
}

#if MODERNTETRIS_PLACEMENT
void DataLoaderThread::setPlacementTrainingData(int batch_index)
{
    using namespace minizero::env::moderntetris_placement;
    std::pair<int, int> p = getSharedData()->replay_buffer_.sampleEnvAndPos();
    int env_id = p.first, pos = p.second;
    const EnvironmentLoader& env_loader = getSharedData()->replay_buffer_.env_loaders_[env_id];
    float loss_scale = getSharedData()->replay_buffer_.getLossScale(p);

    // Replay env up to the sampled position.
    // CRITICAL: placement env is stochastic (7-bag piece RNG). Without reset(seed)
    // the replay runs a DIFFERENT piece sequence than self-play -> legal placements
    // at pos t have different action_ids than those stored in SGF -> dense_policy
    // lookup returns 0 for every descriptor -> pi_sum=0 -> uniform fallback label.
    // (Diagnosed via overfit test showing all label top-5 weights = 1/N.)
    //
    // Optimization: per-env snapshot cache. Each act() runs findPlacements()
    // (BFS), so naive replay-from-zero costs O(pos * BFS) per sample. The
    // cache memoizes the env state at every position; after warmup the same
    // env is sampled at hot slots and we skip the replay entirely (one copy).
    Environment env;
    {
        EnvCacheEntry& cache = *getSharedData()->replay_buffer_.env_caches_[env_id];
        std::lock_guard<std::mutex> lock(cache.mutex);

        const auto& action_pairs = env_loader.getActionPairs();
        const int end_pos = std::min(pos, static_cast<int>(action_pairs.size()));
        const int snap_size = static_cast<int>(cache.snapshots.size());
        // Only write snapshots at positions that are multiples of the
        // interval (pos 0 is always eligible, since 0 % K == 0). Reads stay
        // unchanged -- we walk backwards to find any cached slot, whatever
        // spacing the writer chose.
        const int interval = std::max(1, config::learner_placement_cache_interval);

        // Find largest k <= end_pos with a cached snapshot.
        int start_k = 0;
        std::shared_ptr<const Environment> base;
        const int search_from = std::min(end_pos, snap_size - 1);
        for (int k = search_from; k >= 0; --k) {
            if (cache.snapshots[k]) {
                base = cache.snapshots[k];
                start_k = k;
                break;
            }
        }

        if (base) {
            env = *base;
        } else {
            env.reset(env_loader.getSeed());
            // Free the BFS cache before snapshotting; placements_dirty_=true
            // means the next consumer rebuilds, so this clear is free.
            env.shrinkPlacementCache();
            if (snap_size > 0) {
                cache.snapshots[0] = std::make_shared<const Environment>(env);
            }
            start_k = 0;
        }

        for (int i = start_k; i < end_pos; ++i) {
            env.act(action_pairs[i].first);
            const int next_pos = i + 1;
            if (next_pos < snap_size && next_pos % interval == 0 && !cache.snapshots[next_pos]) {
                env.shrinkPlacementCache();
                cache.snapshots[next_pos] = std::make_shared<const Environment>(env);
            }
        }
    }

    // Deterministic ordering: getActionDescriptors() comes from rebuildLegalPlacements()
    // which calls findPlacements(). The order is fixed by BFS across runs.
    std::vector<PlacementActionDescriptor> descs = env.getActionDescriptors();
    const int N = static_cast<int>(descs.size());

    // Fetch dense SGF policy (size = kMaxPlacementActionId), sparse entries non-zero.
    std::vector<float> dense_policy = env_loader.getPolicy(pos, utils::Rotation::kRotationNone);
    std::vector<float> value = env_loader.getValue(pos);

    // Shared data ptr.
    auto dp = getSharedData()->getDataPtr();
    const int N_max = dp->placement_n_max_;
    const int preview_size = dp->placement_preview_size_;

    // Per-action numpy buffers are sized [B, N_max]. Writing i in [0, N) with
    // N > N_max would stomp the next batch slot (silent corruption). Fail loud
    // so it shows up in training logs instead of as garbage gradients.
    if (N > N_max) {
        std::ostringstream oss;
        oss << "[placement loader] FATAL: legal placement count N=" << N
            << " exceeds N_max=" << N_max
            << " at env_id=" << env_id << " pos=" << pos
            << " (batch_index=" << batch_index << ")."
            << " Raise kPlacementActionUpperBound in"
            << " minizero/network/placement_transformer_network.h,"
            << " placement_n_max() in minizero/learner/pybind.cpp,"
            << " and placement_n_max_ default in minizero/learner/data_loader.h"
            << " (all three must match).";
        std::cerr << oss.str() << std::endl;
        throw std::runtime_error(oss.str());
    }

    // Board features [C=1, H, W].
    const auto board = env.getBoardFeatures();
    std::copy(board.begin(), board.end(), dp->features_ + board.size() * batch_index);

    // Globals (scalar per batch slot).
    auto g = env.getGlobalFeatures();
    auto normalizePiece = [](int p) -> int64_t { return (p < 0 || p >= 7) ? 7 : static_cast<int64_t>(p); };
    dp->placement_current_piece_[batch_index] = normalizePiece(g.current_piece);
    dp->placement_hold_piece_[batch_index] = normalizePiece(g.hold_piece);
    dp->placement_has_held_[batch_index] = g.has_held ? 1.0f : 0.0f;
    for (int j = 0; j < preview_size; ++j) {
        int pv = (j < static_cast<int>(g.preview.size())) ? g.preview[j] : -1;
        dp->placement_preview_[batch_index * preview_size + j] = normalizePiece(pv);
    }
    dp->placement_was_rotation_[batch_index] = g.was_rotation ? 1.0f : 0.0f;
    dp->placement_srs_index_[batch_index] = std::clamp(g.srs_index + 1, 0, 6);
    dp->placement_lifetime_[batch_index] = 0.0f;
    dp->placement_combo_[batch_index] = std::clamp((g.combo_count + 1) / 10.0f, 0.0f, 1.0f);
    dp->placement_back_to_back_[batch_index] = g.back_to_back ? 1.0f : 0.0f;
    dp->placement_garbage_[batch_index] = std::clamp(g.pending_garbage / 20.0f, 0.0f, 1.0f);

    // Per-action tensors and pi_target (zero-initialized by numpy).
    const int action_base = batch_index * N_max;
    float pi_sum = 0.0f;
    int match_count = 0; // replay descriptors whose action_id also appears with non-zero weight in the SGF policy
    for (int i = 0; i < N; ++i) {
        const auto& d = descs[i];
        dp->placement_action_use_hold_[action_base + i] = d.use_hold ? 1 : 0;
        dp->placement_action_lock_x_[action_base + i] = d.lock_x;
        dp->placement_action_lock_y_[action_base + i] = d.lock_y;
        dp->placement_action_orientation_[action_base + i] = d.orientation;
        dp->placement_action_spin_type_[action_base + i] = d.spin_type;
        dp->placement_action_piece_type_[action_base + i] = d.piece_type;
        dp->placement_action_lines_cleared_[action_base + i] = d.lines_cleared;
        dp->placement_action_mask_[action_base + i] = 0; // valid
        const float w = (d.action_id >= 0 && d.action_id < static_cast<int>(dense_policy.size())) ? dense_policy[d.action_id] : 0.0f;
        dp->policy_[action_base + i] = w;
        pi_sum += w;
        if (w > 0.0f) { ++match_count; }
    }
    // Pad tail: masked = 1, π_target = 0. Action field tails are already zero
    // from the numpy allocator but reset to be safe.
    for (int i = N; i < N_max; ++i) {
        dp->placement_action_use_hold_[action_base + i] = 0;
        dp->placement_action_lock_x_[action_base + i] = 0;
        dp->placement_action_lock_y_[action_base + i] = 0;
        dp->placement_action_orientation_[action_base + i] = 0;
        dp->placement_action_spin_type_[action_base + i] = 0;
        dp->placement_action_piece_type_[action_base + i] = 0;
        dp->placement_action_lines_cleared_[action_base + i] = 0;
        dp->placement_action_mask_[action_base + i] = 1; // padded
        dp->policy_[action_base + i] = 0.0f;
    }
    // Strict determinism check: every non-zero entry in the SGF policy must
    // correspond to a legal placement in the replayed env. If any SGF non-zero
    // action_id is missing from the replay's legal set, the replay has diverged
    // from self-play and training on this sample would silently distort
    // pi_target (partial overlap gets re-normalized to a wrong distribution).
    // The old pi_sum == 0 check caught only full divergence; this catches any.
    int sgf_nonzero = 0;
    for (float w : dense_policy) {
        if (w > 0.0f) { ++sgf_nonzero; }
    }
    if (sgf_nonzero == 0) {
        std::ostringstream oss;
        oss << "[placement loader] FATAL: SGF policy has no non-zero entries at env_id=" << env_id
            << " pos=" << pos << " N=" << N
            << ". The stored P[] tag is empty -- likely an SGF write bug or truncated record.";
        std::cerr << oss.str() << std::endl;
        throw std::runtime_error(oss.str());
    }
    if (match_count != sgf_nonzero) {
        std::ostringstream oss;
        oss << "[placement loader] FATAL: replay diverged from self-play at env_id=" << env_id
            << " pos=" << pos << " N=" << N
            << " match=" << match_count << "/" << sgf_nonzero
            << " chosen_action_id=" << env_loader.getActionPairs()[pos].first.getActionID()
            << ". Replay's legal set is missing " << (sgf_nonzero - match_count)
            << " SGF non-zero policy id(s)."
            << " Likely causes: (1) missing env.reset(seed) before replay,"
            << " (2) findPlacements() BFS determinism broke,"
            << " (3) hold-branch merge order changed,"
            << " (4) env config differs between sp and op (e.g. garbage params, piece_lifetime).";
        std::cerr << oss.str() << std::endl;
        throw std::runtime_error(oss.str());
    }
    // match_count > 0 here, so pi_sum > 0.
    for (int i = 0; i < N; ++i) { dp->policy_[action_base + i] /= pi_sum; }

    // Value.
    std::copy(value.begin(), value.end(), dp->value_ + value.size() * batch_index);
    dp->loss_scale_[batch_index] = loss_scale;
    dp->sampled_index_[2 * batch_index] = p.first;
    dp->sampled_index_[2 * batch_index + 1] = p.second;
}
#endif

} // namespace minizero::learner
