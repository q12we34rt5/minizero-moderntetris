#pragma once

#include "alphazero_network.h"
#include "base_actor.h"
#include "configuration.h"
#include "gumbel_zero.h"
#include "mcts.h"
#include "muzero_network.h"
#if MODERNTETRIS_PLACEMENT
#include "placement_transformer_network.h"
#endif
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace minizero::actor {

class MCTSSearchData {
public:
    std::string search_info_;
    MCTSNode* selected_node_;
    std::vector<MCTSNode*> node_path_;
    // Cache of the environment at the leaf of node_path_. Populated by
    // beforeNNEvaluation and reused by the matching afterNNEvaluation.
    // Empty for muzero (which never needs real env transitions inside search).
    std::optional<Environment> env_transition_;
    void clear();
};

class ZeroActor : public BaseActor {
public:
    ZeroActor(uint64_t tree_node_size)
        : tree_node_size_(tree_node_size)
    {
        alphazero_network_ = nullptr;
        muzero_network_ = nullptr;
#if MODERNTETRIS_PLACEMENT
        placement_network_ = nullptr;
#endif
    }

    void reset() override;
    void resetSearch() override;
    Action think(bool with_play = false, bool display_board = false) override;
    void beforeNNEvaluation() override;
    void afterNNEvaluation(const std::shared_ptr<network::NetworkOutput>& network_output) override;
    bool isSearchDone() const override { return getMCTS()->reachMaximumSimulation() && pimc_search_index_ + 1 >= config::actor_mcts_pimc_determinizations && (!config::actor_mcts_privileged_value || privileged_phase_); }
    bool usePimcAggregation() const { return config::actor_mcts_pimc_determinizations > 1 || config::actor_mcts_pimc_force; }
    Action getSearchAction() const override { return mcts_search_data_.selected_node_->getAction(); }
    bool isResign() const override { return enable_resign_ && getMCTS()->isResign(mcts_search_data_.selected_node_); }
    std::string getSearchInfo() const override { return mcts_search_data_.search_info_; }
    void setNetwork(const std::shared_ptr<network::Network>& network) override;
    std::shared_ptr<Search> createSearch() override { return std::make_shared<MCTS>(tree_node_size_); }
    std::shared_ptr<MCTS> getMCTS() { return std::static_pointer_cast<MCTS>(search_); }
    const std::shared_ptr<MCTS> getMCTS() const { return std::static_pointer_cast<MCTS>(search_); }

protected:
    std::vector<std::pair<std::string, std::string>> getActionInfo() const override;
    std::string getMCTSPolicy() const override
    {
        if (config::actor_mcts_privileged_value) { return honest_policy_; }
        if (usePimcAggregation()) { return pimcPolicyString(); }
        return (config::actor_use_gumbel ? gumbel_zero_.getMCTSPolicy(getMCTS()) : getMCTS()->getSearchDistributionString());
    }
    std::string getMCTSValue() const override
    {
        if (usePimcAggregation()) {
            return std::to_string(static_cast<float>(pimc_value_sum_ / (pimc_search_index_ + 1)));
        }
        return std::to_string(getMCTS()->getRootNode()->getMean());
    }
    std::string getEnvReward() const override;

    virtual void step();
    // PIMC (actor_mcts_pimc_determinizations > 1): each move searches several
    // independently resampled futures in turn and decides on their averaged root
    // statistics, which trades searching one sampled future deeply against
    // covering more of them.
    void resampleSearchRootEnv();
    // Critic-only privileged information (actor_mcts_privileged_value): once the
    // honest search is done, keep its move and policy, then search the real future
    // for the value only.
    void startPrivilegedValueSearch();
    void accumulatePimcSearch();
    void startNextPimcSearch();
    std::string pimcPolicyString() const;
    virtual void handleSearchDone();
    virtual MCTSNode* decideActionNode();
    virtual void addNoiseToNodeChildren(MCTSNode* node);
    virtual std::vector<MCTSNode*> selection() { return (config::actor_use_gumbel ? gumbel_zero_.selection(getMCTS()) : getMCTS()->select()); }

    std::vector<MCTS::ActionCandidate> calculateAlphaZeroActionPolicy(const Environment& env_transition, const std::shared_ptr<network::AlphaZeroNetworkOutput>& alphazero_output, const utils::Rotation& rotation);
    std::vector<MCTS::ActionCandidate> calculateMuZeroActionPolicy(MCTSNode* leaf_node, const std::shared_ptr<network::MuZeroNetworkOutput>& muzero_output);
#if MODERNTETRIS_PLACEMENT
    std::vector<MCTS::ActionCandidate> calculatePlacementActionPolicy(const Environment& env_transition, const std::shared_ptr<network::PlacementNetworkOutput>& placement_output);
#endif
    virtual Environment getEnvironmentTransition(const std::vector<MCTSNode*>& node_path);
#if MODERNTETRIS_PLACEMENT
    void checkSearchRootObservation() const;
#endif
    const Environment& getSearchRootEnv() const { return search_root_env_ ? *search_root_env_ : env_; }

    bool enable_resign_;
    GumbelZero gumbel_zero_;
    uint64_t tree_node_size_;
    MCTSSearchData mcts_search_data_;
    // Set when actor_mcts_resample_hidden_future is on: the env the search
    // expands from, i.e. env_ with its unobservable future resampled.
    std::optional<Environment> search_root_env_;
    // PIMC bookkeeping: which determinization we are on, per-action totals over
    // the ones done so far, and their summed root values.
    int pimc_search_index_ = 0;
    std::unordered_map<int, double> pimc_action_score_; // averaged search policy (the training target)
    std::unordered_map<int, double> pimc_count_;        // summed normalized visit counts
    std::unordered_map<int, double> pimc_vote_;         // moves each tree would have played
    double pimc_value_sum_ = 0;
    // Privileged-value bookkeeping: whether the real-future search is running, and
    // what the honest search decided and would have trained on.
    bool privileged_phase_ = false;
    int honest_action_id_ = -1;
    std::string honest_policy_;
    utils::Rotation feature_rotation_;
    std::shared_ptr<network::AlphaZeroNetwork> alphazero_network_;
    std::shared_ptr<network::MuZeroNetwork> muzero_network_;
#if MODERNTETRIS_PLACEMENT
    std::shared_ptr<network::PlacementTransformerNetwork> placement_network_;
#endif
};

} // namespace minizero::actor
