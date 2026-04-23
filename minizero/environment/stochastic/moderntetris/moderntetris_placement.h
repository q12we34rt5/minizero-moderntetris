#pragma once

#include "engine/placement_search.hpp"
#include "engine/step.hpp"
#include "stochastic_env.h"
#include <cmath>
#include <deque>
#include <string>
#include <vector>

namespace minizero::env::moderntetris_placement {

namespace engine = minizero::env::moderntetris::engine;

struct PlacementGlobalFeatures {
    int current_piece; // 0..6, -1 if NONE
    int hold_piece;    // 0..6, -1 if NONE
    bool has_held;
    std::vector<int> preview; // each 0..6, size = preview_size
    bool was_rotation;
    int srs_index; // -1..5
    int combo_count;
    bool back_to_back;
    int pending_garbage;
};

struct PlacementActionDescriptor {
    int action_id;
    bool use_hold;
    int lock_x;        // 0..W-1 (relative to visible board)
    int lock_y;        // 0..H-1
    int orientation;   // 0..3
    int spin_type;     // 0..2
    int piece_type;    // 0..6
    int lines_cleared; // 0..4
};

constexpr char kModernTetrisPlacementName[] = "moderntetris_placement";
constexpr int kModernTetrisPlacementNumPlayer = 1;
constexpr int kModernTetrisPlacementBoardSize = 10;
constexpr int kModernTetrisPlacementBoardWidth = engine::BOARD_RIGHT - engine::BOARD_LEFT + 1;
constexpr int kModernTetrisPlacementBoardHeight = engine::BOARD_BOTTOM - engine::BOARD_TOP + 1;
constexpr int kModernTetrisPlacementChanceEventSize = 1;
constexpr int kModernTetrisPlacementDiscreteValueSize = 601;

// Placement-transformer specific: board tokens only carry locked cells (scheme A).
// Current piece / hold / preview are delivered via global tokens.
constexpr int kPlacementBoardChannels = 1;
constexpr int kPlacementPatchSize = 5;

constexpr int kPackX = 16;
constexpr int kPackY = 32;
constexpr int kPackO = 4;
constexpr int kPackS = 3;
constexpr int kPlacementsPerHoldBranch = kPackX * kPackY * kPackO * kPackS;
constexpr int kMaxPlacementActionId = 2 * kPlacementsPerHoldBranch;
constexpr int kPlacementChanceEventId = kMaxPlacementActionId;

inline int packPlacementId(bool use_hold, int lock_x, int lock_y, int orientation, int spin_type)
{
    return static_cast<int>(use_hold) * kPlacementsPerHoldBranch + lock_x * (kPackY * kPackO * kPackS) + lock_y * (kPackO * kPackS) + orientation * kPackS + spin_type;
}

struct UnpackedPlacement {
    bool use_hold;
    int lock_x;
    int lock_y;
    int orientation;
    int spin_type;
};

inline UnpackedPlacement unpackPlacementId(int action_id)
{
    UnpackedPlacement p;
    p.use_hold = action_id >= kPlacementsPerHoldBranch;
    int rem = action_id % kPlacementsPerHoldBranch;
    p.lock_x = rem / (kPackY * kPackO * kPackS);
    rem %= (kPackY * kPackO * kPackS);
    p.lock_y = rem / (kPackO * kPackS);
    rem %= (kPackO * kPackS);
    p.orientation = rem / kPackS;
    p.spin_type = rem % kPackS;
    return p;
}

void initialize();

class ModernTetrisPlacementAction : public BaseAction {
public:
    ModernTetrisPlacementAction() : BaseAction() {}
    ModernTetrisPlacementAction(int action_id, Player player) : BaseAction(action_id, player) {}
    ModernTetrisPlacementAction(const std::vector<std::string>& action_string_args);

    Player nextPlayer() const override { return player_ == Player::kPlayer1 ? Player::kPlayerNone : Player::kPlayer1; }
    std::string toConsoleString() const override;
};

class ModernTetrisPlacementEnv : public StochasticEnv<ModernTetrisPlacementAction> {
public:
    ModernTetrisPlacementEnv() { reset(); }

    void reset() override { reset(utils::Random::randInt()); }
    void reset(int seed) override;
    bool act(const ModernTetrisPlacementAction& action, bool with_chance = true) override;
    bool act(const std::vector<std::string>& action_string_args, bool with_chance = true) override { return act(ModernTetrisPlacementAction(action_string_args), with_chance); }
    bool actChanceEvent(const ModernTetrisPlacementAction& action) override;
    bool actChanceEvent();

    std::vector<ModernTetrisPlacementAction> getLegalActions() const override;
    std::vector<ModernTetrisPlacementAction> getLegalChanceEvents() const override;
    float getChanceEventProbability(const ModernTetrisPlacementAction& action) const override;
    bool isLegalAction(const ModernTetrisPlacementAction& action) const override;
    bool isLegalChanceEvent(const ModernTetrisPlacementAction& action) const override;
    bool isTerminal() const override;

    int getRotatePosition(int position, utils::Rotation rotation) const override { return position; }
    int getRotateAction(int action_id, utils::Rotation rotation) const override { return action_id; }
    int getRotateChanceEvent(int event_id, utils::Rotation rotation) const override { return event_id; }
    std::vector<float> getFeatures(utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    std::vector<float> getActionFeatures(const ModernTetrisPlacementAction& action, utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    std::vector<float> getChanceEventFeatures(const ModernTetrisPlacementAction& event, utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    int getNumInputChannels() const override;
    int getNumActionFeatureChannels() const override { return 1; }
    int getNumChanceEventFeatureChannels() const override { return kModernTetrisPlacementChanceEventSize; }
    int getInputChannelHeight() const override { return kModernTetrisPlacementBoardHeight; }
    int getInputChannelWidth() const override { return kModernTetrisPlacementBoardWidth; }
    int getHiddenChannelHeight() const override { return kModernTetrisPlacementBoardHeight; }
    int getHiddenChannelWidth() const override { return kModernTetrisPlacementBoardWidth; }
    int getPolicySize() const override { return kMaxPlacementActionId; }
    int getChanceEventSize() const override { return kModernTetrisPlacementChanceEventSize; }
    int getDiscreteValueSize() const override { return kModernTetrisPlacementDiscreteValueSize; }

    std::string toString() const override;
    std::string name() const override { return kModernTetrisPlacementName; }
    int getNumPlayer() const override { return kModernTetrisPlacementNumPlayer; }
    float getReward() const override { return reward_; }
    float getEvalScore(bool is_resign = false) const override { return total_reward_; }

    // --- Placement transformer feature APIs (non-virtual; consumed by new network wrapper) ---
    std::vector<float> getBoardFeatures() const; // size = C_board * H * W, C_board = 1
    PlacementGlobalFeatures getGlobalFeatures() const;
    std::vector<PlacementActionDescriptor> getActionDescriptors() const; // aligned with getLegalActions() order
    int getBoardChannels() const { return kPlacementBoardChannels; }

    // Free the cached BFS placement results. Each cached entry carries a full
    // PlacementSearchResult (including the BFS path and final State), so the
    // cache can grow to tens of KB; clearing it before snapshotting an env in
    // the data loader keeps replay-buffer memory bounded. Safe to call any
    // time -- placements_dirty_ stays true so the next consumer rebuilds.
    void shrinkPlacementCache()
    {
        cached_placements_.clear();
        cached_placements_.shrink_to_fit();
        placements_dirty_ = true;
    }

private:
    struct CachedPlacement {
        int action_id;
        engine::PlacementSearchResult result;
    };

    void rebuildLegalPlacements() const;
    static int toPieceIndex(engine::PieceType piece_type);
    static bool isOccupied(engine::Cell cell);
    static int getChannelCount(int preview_size, int history_length);
    void resetActivePieceHistory();
    void writeBoardFeatures(std::vector<float>& features) const;
    void writePieceFeatures(std::vector<float>& features, int channel_offset, engine::PieceType piece_type) const;
    void fillScalarPlane(std::vector<float>& features, int channel, float value) const;
    void fillOneHotPlane(std::vector<float>& features, int channel_offset, int size, int index) const;

    static engine::step::Action placementActionToStepAction(engine::PlacementAction pa);

private:
    engine::step::Context ctx_;
    std::deque<std::vector<float>> active_piece_history_;
    float reward_ = 0.0f;
    float total_reward_ = 0.0f;
    float reward_prev_potential_ = 0.0f;
    mutable std::vector<CachedPlacement> cached_placements_;
    mutable bool placements_dirty_ = true;
};

class ModernTetrisPlacementEnvLoader : public StochasticEnvLoader<ModernTetrisPlacementAction, ModernTetrisPlacementEnv> {
public:
    std::vector<float> getActionFeatures(const int pos, utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    std::vector<float> getChanceEventFeatures(const int pos, utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    std::vector<float> getChance(const int pos, utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    std::vector<float> getValue(const int pos) const override { return toDiscreteValue(pos < static_cast<int>(action_pairs_.size()) ? utils::transformValue(calculateNStepValue(pos)) : 0.0f); }
    std::vector<float> getAfterstateValue(const int pos) const override { return toDiscreteValue(pos < static_cast<int>(action_pairs_.size()) ? utils::transformValue(calculateNStepValue(pos) - BaseEnvLoader::getReward(pos)[0]) : 0.0f); }
    std::vector<float> getReward(const int pos) const override { return toDiscreteValue(pos < static_cast<int>(action_pairs_.size()) ? utils::transformValue(BaseEnvLoader::getReward(pos)[0]) : 0.0f); }
    float getPriority(const int pos) const override { return std::fabs(calculateNStepValue(pos) - BaseEnvLoader::getValue(pos)[0]); }

    std::string name() const override { return kModernTetrisPlacementName; }
    int getPolicySize() const override { return kMaxPlacementActionId; }
    int getChanceEventSize() const override { return kModernTetrisPlacementChanceEventSize; }
    int getRotatePosition(int position, utils::Rotation rotation) const override { return position; }
    int getRotateAction(int action_id, utils::Rotation rotation) const override { return action_id; }
    int getRotateChanceEvent(int event_id, utils::Rotation rotation) const override { return event_id; }

private:
    float calculateNStepValue(const int pos) const;
    std::vector<float> toDiscreteValue(float value) const;
};

} // namespace minizero::env::moderntetris_placement
