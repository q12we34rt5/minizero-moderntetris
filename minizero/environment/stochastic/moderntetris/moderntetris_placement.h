#pragma once

#include "engine/placement_search.hpp"
#include "engine/step.hpp"
#include "stochastic_env.h"
#include <cmath>
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
constexpr int kModernTetrisPlacementBoardWidth = engine::BOARD_RIGHT - engine::BOARD_LEFT + 1;
constexpr int kModernTetrisPlacementBoardHeight = engine::BOARD_BOTTOM - engine::BOARD_TOP + 1;
constexpr int kModernTetrisPlacementChanceEventSize = 1;
constexpr int kModernTetrisPlacementDiscreteValueSize = 601;

// Placement-transformer specific: board tokens only carry locked cells (scheme A).
// Current piece / hold / preview are delivered via global tokens.
constexpr int kPlacementBoardChannels = 1;

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

// Collapse rotation-symmetric placements onto one canonical (orientation, x, y)
// so the legal-action set matches fusion's: O has a single distinct shape,
// I/S/Z have two (North==South, East==West), L/J/T keep all four. Two
// placements that cover the same cells therefore share one action id, removing
// the ~2x duplicate-shape inflation in moderntetris's enumeration (see
// reports/placement-algorithm.md §6.3 and reports/performance.md §6).
// The function itself lives in engine/placement_search.hpp so the WASM frontend
// can mirror this exact collapse when matching a backend placement to a path.
using engine::canonicalizePlacement;
using engine::CanonicalPlacement;

void initialize();

class ModernTetrisPlacementAction : public BaseAction {
public:
    ModernTetrisPlacementAction() : BaseAction() {}
    ModernTetrisPlacementAction(int action_id, Player player) : BaseAction(action_id, player) {}
    ModernTetrisPlacementAction(const std::vector<std::string>& action_string_args);

    // Positions strictly alternate placement -> chance -> placement. After a
    // placement (kPlayer1/kPlayer2) the next position is a chance event
    // (kPlayerNone). After a chance event the next placement belongs to the
    // other player in two-player mode, or kPlayer1 in single-player mode; the
    // env's turn_ is the authority for that (see actChanceEvent), so a chance
    // action reports kPlayerNone here and lets the env resolve the next mover.
    Player nextPlayer() const override { return player_ == Player::kPlayerNone ? Player::kPlayer1 : Player::kPlayerNone; }
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
    int getNumPlayer() const override;
    float getReward() const override { return reward_; }
    float getEvalScore(bool is_resign = false) const override;
    // Self-play monitoring return: two-player win/loss (getEvalScore) is
    // structurally always +1, so report both players' summed env reward instead
    // — a meaningful "how much did they score this game" signal, comparable to
    // the single-player return. Display only; does not affect training.
    float getSelfPlayGameReturn(bool is_resign = false) const override;

    // --- Placement transformer feature APIs (non-virtual; consumed by new network wrapper) ---
    std::vector<float> getBoardFeatures() const; // size = C_board * H * W; C_board = 1 (single) or 2 (two-player: mover + opponent)
    PlacementGlobalFeatures getGlobalFeatures() const;
    std::vector<PlacementActionDescriptor> getActionDescriptors() const; // aligned with getLegalActions() order
    int getBoardChannels() const;                                        // 1 in single-player mode, 2 in two-player mode

    // Raw engine state of the player to move. The data loader's
    // verify-then-mirror augmentation uses it to run the engine's BFS on the
    // mirrored board as ground truth; the placement_mirror_verify console mode
    // uses it the same way.
    const engine::State& getEngineState() const { return moverCtx().state; }
    const engine::step::Context& getEngineContext() const { return moverCtx(); }

    // Inject a full engine context (board + pieces + garbage + config) directly,
    // bypassing reset/act. Used by the console set_state command to serve AI
    // moves for arbitrary boards (web PvE/EvE). Resets episode bookkeeping so the
    // env behaves as a fresh root at the injected position.
    void setState(const engine::step::Context& ctx);
    // Two-board variant for two-player inference: ctx0 = the player to move
    // (kPlayer1), ctx1 = the opponent. Lets the console/web inject a real
    // opponent board so the AI's two-player search sees it, instead of the
    // empty-opponent fallback of the single-board setState.
    void setState(const engine::step::Context& ctx0, const engine::step::Context& ctx1);

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

    static engine::step::Action placementActionToStepAction(engine::PlacementAction pa);

    // Board index (0 = kPlayer1, 1 = kPlayer2) of the player to move. During a
    // chance ply (turn_ == kPlayerNone) we fall back to the last mover so that
    // feature/tooling getters still resolve to a valid board.
    int moverIndex() const
    {
        Player p = (turn_ == Player::kPlayerNone) ? last_mover_ : turn_;
        return p == Player::kPlayer2 ? 1 : 0;
    }
    engine::step::Context& moverCtx() { return ctx_[moverIndex()]; }
    const engine::step::Context& moverCtx() const { return ctx_[moverIndex()]; }
    engine::step::Context& oppCtx() { return ctx_[1 - moverIndex()]; }
    const engine::step::Context& oppCtx() const { return ctx_[1 - moverIndex()]; }

private:
    // ctx_[0] = kPlayer1's board, ctx_[1] = kPlayer2's board. In single-player
    // mode (env_modern_tetris_two_player == false) only ctx_[0] is ever played;
    // ctx_[1] stays reset-and-idle so isTerminal()'s opponent-death check is
    // inert. total_reward_ / reward_prev_potential_ are per-player.
    engine::step::Context ctx_[2];
    Player last_mover_ = Player::kPlayer1;
    float reward_ = 0.0f; // env reward of the most recent placement (mover's own)
    float total_reward_[2] = {0.0f, 0.0f};
    float reward_prev_potential_[2] = {0.0f, 0.0f};
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

    // Two-player win/loss target: one-hot over {lose, draw, win} (size 3) from
    // the perspective of position pos's mover. Derived from the terminal RE tag
    // (stored from the final to-move / winning player's view) and the absolute
    // winner (= the player who did NOT make the last placement). Only meaningful
    // in two-player mode.
    std::vector<float> getWinLossValue(const int pos) const;

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
