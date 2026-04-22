#include "zero_actor.h"
#include "random.h"
#include "time_system.h"
#include <algorithm>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#if MODERNTETRIS_PLACEMENT
#include "moderntetris_placement.h"
#endif

namespace minizero::actor {

using namespace minizero;
using namespace network;

void MCTSSearchData::clear()
{
    search_info_ = "";
    selected_node_ = nullptr;
    node_path_.clear();
}

void ZeroActor::reset()
{
    BaseActor::reset();
    enable_resign_ = (utils::Random::randReal() < config::zero_disable_resign_ratio ? false : true);
}

void ZeroActor::resetSearch()
{
    BaseActor::resetSearch();
    mcts_search_data_.node_path_.clear();
    getMCTS()->getRootNode()->setAction(Action(-1, env::getPreviousPlayer(env_.getTurn(), env_.getNumPlayer())));
}

Action ZeroActor::think(bool with_play /*= false*/, bool display_board /*= false*/)
{
    resetSearch();
    boost::posix_time::ptime start_ptime = utils::TimeSystem::getLocalTime();
    while (!isSearchDone()) {
        step();
        int spent_million_second = (utils::TimeSystem::getLocalTime() - start_ptime).total_milliseconds();
        if (config::actor_mcts_think_time_limit > 0 && spent_million_second >= config::actor_mcts_think_time_limit * 1000) { break; }
    }
    if (!isSearchDone()) { handleSearchDone(); }
    if (with_play) { act(getSearchAction()); }
    if (display_board) { std::cerr << env_.toString() << mcts_search_data_.search_info_ << std::endl; }
    return getSearchAction();
}

void ZeroActor::beforeNNEvaluation()
{
    mcts_search_data_.node_path_ = selection();
    if (alphazero_network_) {
        Environment env_transition = getEnvironmentTransition(mcts_search_data_.node_path_);
        feature_rotation_ = config::actor_use_random_rotation_features ? static_cast<utils::Rotation>(utils::Random::randInt() % static_cast<int>(utils::Rotation::kRotateSize)) : utils::Rotation::kRotationNone;
        nn_evaluation_batch_id_ = alphazero_network_->pushBack(env_transition.getFeatures(feature_rotation_));
#if MODERNTETRIS_PLACEMENT
    } else if (placement_network_) {
        Environment env_transition = getEnvironmentTransition(mcts_search_data_.node_path_);
        feature_rotation_ = utils::Rotation::kRotationNone;
        using namespace minizero::env::moderntetris_placement;
        auto in = network::buildPlacementNetworkInput(
            env_transition, kPlacementBoardChannels,
            kModernTetrisPlacementBoardHeight, kModernTetrisPlacementBoardWidth);
        nn_evaluation_batch_id_ = placement_network_->pushBack(std::move(in));
#endif
    } else if (muzero_network_) {
        if (getMCTS()->getNumSimulation() == 0) { // initial inference for root node
            nn_evaluation_batch_id_ = muzero_network_->pushBackInitialData(env_.getFeatures());
        } else { // for non-root nodes
            const std::vector<MCTSNode*>& node_path = mcts_search_data_.node_path_;
            MCTSNode* leaf_node = node_path.back();
            MCTSNode* parent_node = node_path[node_path.size() - 2];
            assert(parent_node && parent_node->getHiddenStateDataIndex() != -1);
            const std::vector<float>& hidden_state = getMCTS()->getTreeHiddenStateData().getData(parent_node->getHiddenStateDataIndex()).hidden_state_;
            nn_evaluation_batch_id_ = muzero_network_->pushBackRecurrentData(hidden_state, env_.getActionFeatures(leaf_node->getAction()));
        }
    } else {
        assert(false);
    }
}

void ZeroActor::afterNNEvaluation(const std::shared_ptr<NetworkOutput>& network_output)
{
    const std::vector<MCTSNode*>& node_path = mcts_search_data_.node_path_;
    MCTSNode* leaf_node = node_path.back();
    if (alphazero_network_) {
        Environment env_transition = getEnvironmentTransition(node_path);
        if (!env_transition.isTerminal()) {
            std::shared_ptr<AlphaZeroNetworkOutput> alphazero_output = std::static_pointer_cast<AlphaZeroNetworkOutput>(network_output);
            getMCTS()->expand(leaf_node, calculateAlphaZeroActionPolicy(env_transition, alphazero_output, feature_rotation_));
            getMCTS()->backup(node_path, alphazero_output->value_, env_transition.getReward());
        } else {
            getMCTS()->backup(node_path, env_transition.getEvalScore(), env_transition.getReward());
        }
#if MODERNTETRIS_PLACEMENT
    } else if (placement_network_) {
        Environment env_transition = getEnvironmentTransition(node_path);
        if (!env_transition.isTerminal()) {
            auto placement_output = std::static_pointer_cast<PlacementNetworkOutput>(network_output);
            getMCTS()->expand(leaf_node, calculatePlacementActionPolicy(env_transition, placement_output));
            getMCTS()->backup(node_path, placement_output->value_, env_transition.getReward());
        } else {
            getMCTS()->backup(node_path, env_transition.getEvalScore(), env_transition.getReward());
        }
#endif
    } else if (muzero_network_) {
        std::shared_ptr<MuZeroNetworkOutput> muzero_output = std::static_pointer_cast<MuZeroNetworkOutput>(network_output);
        getMCTS()->expand(leaf_node, calculateMuZeroActionPolicy(leaf_node, muzero_output));
        getMCTS()->backup(node_path, muzero_output->value_, muzero_output->reward_);
        leaf_node->setHiddenStateDataIndex(getMCTS()->getTreeHiddenStateData().store(HiddenStateData(muzero_output->hidden_state_)));
    } else {
        assert(false);
    }
    if (leaf_node == getMCTS()->getRootNode()) { addNoiseToNodeChildren(leaf_node); }
    if (isSearchDone()) { handleSearchDone(); }
    if (config::actor_use_gumbel) { gumbel_zero_.sequentialHalving(getMCTS()); }
}

void ZeroActor::setNetwork(const std::shared_ptr<network::Network>& network)
{
    assert(network);
    alphazero_network_ = nullptr;
    muzero_network_ = nullptr;
#if MODERNTETRIS_PLACEMENT
    placement_network_ = nullptr;
#endif
    if (network->getNetworkTypeName() == "alphazero") {
        alphazero_network_ = std::static_pointer_cast<AlphaZeroNetwork>(network);
    } else if (network->getNetworkTypeName() == "muzero" || network->getNetworkTypeName() == "muzero_atari") {
        muzero_network_ = std::static_pointer_cast<MuZeroNetwork>(network);
#if MODERNTETRIS_PLACEMENT
    } else if (network->getNetworkTypeName() == "placement_transformer") {
        placement_network_ = std::static_pointer_cast<PlacementTransformerNetwork>(network);
#endif
    } else {
        assert(false);
    }
}

std::vector<std::pair<std::string, std::string>> ZeroActor::getActionInfo() const
{
    // ignore recording mcts action info if there is no search
    if (getMCTS()->getRootNode()->getCount() > 0) { return BaseActor::getActionInfo(); }
    return {};
}

std::string ZeroActor::getEnvReward() const
{
    std::ostringstream oss;
    oss << env_.getReward();
    return oss.str();
}

void ZeroActor::step()
{
#if MODERNTETRIS_PLACEMENT
    assert(alphazero_network_ || muzero_network_ || placement_network_);
#else
    assert(alphazero_network_ || muzero_network_);
#endif
    int num_simulation = getMCTS()->getNumSimulation();
    int num_simulation_left = config::actor_num_simulation + 1 - num_simulation;
    bool direct_inference = alphazero_network_ || num_simulation > 0;
#if MODERNTETRIS_PLACEMENT
    direct_inference = direct_inference || placement_network_;
#endif
    int batch_size = std::min(config::actor_mcts_think_batch_size,
                              direct_inference ? num_simulation_left : 1 /* initial inference for root node */);
    assert(batch_size > 0);

    std::vector<std::tuple<int, utils::Rotation, decltype(mcts_search_data_.node_path_)>> batch_queries; // batch id, rotation, search path
    for (int batch_id = 0; batch_id < batch_size; batch_id++) {
        beforeNNEvaluation();
        assert(nn_evaluation_batch_id_ == batch_id);
        if (mcts_search_data_.node_path_.back()->getVirtualLoss() == 0) {
            batch_queries.emplace_back(nn_evaluation_batch_id_, feature_rotation_, mcts_search_data_.node_path_);
        }
        for (auto node : mcts_search_data_.node_path_) { node->addVirtualLoss(); }
    }
    std::vector<std::shared_ptr<network::NetworkOutput>> network_output;
    if (alphazero_network_) {
        network_output = alphazero_network_->forward();
#if MODERNTETRIS_PLACEMENT
    } else if (placement_network_) {
        network_output = placement_network_->forward();
#endif
    } else {
        network_output = (num_simulation == 0 ? muzero_network_->initialInference() : muzero_network_->recurrentInference());
    }
    for (auto& query : batch_queries) {
        nn_evaluation_batch_id_ = std::get<0>(query);
        feature_rotation_ = std::get<1>(query);
        mcts_search_data_.node_path_ = std::get<2>(query);
        afterNNEvaluation(network_output[nn_evaluation_batch_id_]);
        auto virtual_loss = mcts_search_data_.node_path_.back()->getVirtualLoss();
        for (auto node : mcts_search_data_.node_path_) { node->removeVirtualLoss(virtual_loss); }
    }
}

void ZeroActor::handleSearchDone()
{
    mcts_search_data_.selected_node_ = decideActionNode();
    const Action action = getSearchAction();
    std::ostringstream oss;
    oss << "model file name: " << config::nn_file_name << std::endl
        << utils::TimeSystem::getTimeString("[Y/m/d H:i:s.f] ")
        << "move number: " << env_.getActionHistory().size()
        << ", action: " << action.toConsoleString()
        << " (" << action.getActionID() << ")"
        << ", reward: " << env_.getReward()
        << ", player: " << env::playerToChar(action.getPlayer());
    if (config::actor_mcts_value_rescale) { oss << ", value bound: (" << getMCTS()->getTreeValueBound().begin()->first << ", " << getMCTS()->getTreeValueBound().rbegin()->first << ")"; }
    oss << std::endl
        << "  root node info: " << getMCTS()->getRootNode()->toString() << std::endl
        << "action node info: " << mcts_search_data_.selected_node_->toString() << std::endl;
    mcts_search_data_.search_info_ = oss.str();
}

MCTSNode* ZeroActor::decideActionNode()
{
    if (config::actor_use_gumbel) {
        return gumbel_zero_.decideActionNode(getMCTS());
    } else {
        if (config::actor_select_action_by_count) {
            return getMCTS()->selectChildByMaxCount(getMCTS()->getRootNode());
        } else if (config::actor_select_action_by_softmax_count) {
            return getMCTS()->selectChildBySoftmaxCount(getMCTS()->getRootNode(), config::actor_select_action_softmax_temperature);
        }

        assert(false);
        return nullptr;
    }
}

void ZeroActor::addNoiseToNodeChildren(MCTSNode* node)
{
    assert(node && node->getNumChildren() > 0);
    if (config::actor_use_dirichlet_noise) {
        const float epsilon = config::actor_dirichlet_noise_epsilon;
        std::vector<float> dirichlet_noise = utils::Random::randDirichlet(config::actor_dirichlet_noise_alpha, node->getNumChildren());
        for (int i = 0; i < node->getNumChildren(); ++i) {
            MCTSNode* child = node->getChild(i);
            child->setPolicyNoise(dirichlet_noise[i]);
            child->setPolicy((1 - epsilon) * child->getPolicy() + epsilon * dirichlet_noise[i]);
        }
    } else if (config::actor_use_gumbel_noise) {
        std::vector<float> gumbel_noise = utils::Random::randGumbel(node->getNumChildren());
        for (int i = 0; i < node->getNumChildren(); ++i) {
            MCTSNode* child = node->getChild(i);
            child->setPolicyNoise(gumbel_noise[i]);
            child->setPolicyLogit(child->getPolicyLogit() + gumbel_noise[i]);
        }
    }
}

std::vector<MCTS::ActionCandidate> ZeroActor::calculateAlphaZeroActionPolicy(const Environment& env_transition, const std::shared_ptr<network::AlphaZeroNetworkOutput>& alphazero_output, const utils::Rotation& rotation)
{
    assert(alphazero_network_);
    std::vector<MCTS::ActionCandidate> action_candidates;
    for (size_t action_id = 0; action_id < alphazero_output->policy_.size(); ++action_id) {
        Action action(action_id, env_transition.getTurn());
        if (!env_transition.isLegalAction(action)) { continue; }
        int rotated_id = env_transition.getRotateAction(action_id, rotation);
        action_candidates.push_back(MCTS::ActionCandidate(action, alphazero_output->policy_[rotated_id], alphazero_output->policy_logits_[rotated_id]));
    }
    sort(action_candidates.begin(), action_candidates.end(), [](const MCTS::ActionCandidate& lhs, const MCTS::ActionCandidate& rhs) {
        return lhs.policy_ > rhs.policy_;
    });
    return action_candidates;
}

#if MODERNTETRIS_PLACEMENT
std::vector<MCTS::ActionCandidate> ZeroActor::calculatePlacementActionPolicy(const Environment& env_transition, const std::shared_ptr<network::PlacementNetworkOutput>& placement_output)
{
    assert(placement_network_);
    // Env's legal actions and the placement network's policy output are
    // aligned in index: policy_[i] corresponds to legal[i]. No mask / filter needed.
    auto legal = env_transition.getLegalActions();
    assert(legal.size() == placement_output->policy_.size());
    std::vector<MCTS::ActionCandidate> candidates;
    candidates.reserve(legal.size());
    for (size_t i = 0; i < legal.size(); ++i) {
        candidates.emplace_back(legal[i], placement_output->policy_[i], placement_output->policy_logits_[i]);
    }
    sort(candidates.begin(), candidates.end(), [](const MCTS::ActionCandidate& lhs, const MCTS::ActionCandidate& rhs) {
        return lhs.policy_ > rhs.policy_;
    });
    return candidates;
}
#endif

std::vector<MCTS::ActionCandidate> ZeroActor::calculateMuZeroActionPolicy(MCTSNode* leaf_node, const std::shared_ptr<network::MuZeroNetworkOutput>& muzero_output)
{
    assert(muzero_network_);
    std::vector<MCTS::ActionCandidate> action_candidates;
    env::Player turn = leaf_node->getAction().nextPlayer();
    for (size_t action_id = 0; action_id < muzero_output->policy_.size(); ++action_id) {
        const Action action(action_id, turn);
        if (leaf_node == getMCTS()->getRootNode() && !env_.isLegalAction(action)) { continue; }
        action_candidates.push_back(MCTS::ActionCandidate(action, muzero_output->policy_[action_id], muzero_output->policy_logits_[action_id]));
    }
    sort(action_candidates.begin(), action_candidates.end(), [](const MCTS::ActionCandidate& lhs, const MCTS::ActionCandidate& rhs) {
        return lhs.policy_ > rhs.policy_;
    });
    return action_candidates;
}

Environment ZeroActor::getEnvironmentTransition(const std::vector<MCTSNode*>& node_path)
{
    Environment env = env_;
    for (size_t i = 1; i < node_path.size(); ++i) { env.act(node_path[i]->getAction()); }
    return env;
}

} // namespace minizero::actor
