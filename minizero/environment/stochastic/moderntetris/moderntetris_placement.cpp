#include "moderntetris_placement.h"
#include "configuration.h"
#include "random.h"
#include "reward_common.h"
#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace minizero::env::moderntetris_placement {

using namespace minizero::utils;

namespace {

    constexpr int kVisibleCellCount = kModernTetrisPlacementBoardWidth * kModernTetrisPlacementBoardHeight;

    // Mirrors the engine's private xorshf32 (tetris.cpp) so this stream stays
    // in lock-step with the engine's existing garbage_seed usage (hole position).
    inline std::uint32_t xorshift32(std::uint32_t& seed)
    {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        return seed;
    }

} // namespace

void initialize() {}

// --- Action ---

ModernTetrisPlacementAction::ModernTetrisPlacementAction(const std::vector<std::string>& action_string_args)
{
    action_id_ = -1;
    player_ = Player::kPlayerNone;

    std::string token;
    for (const auto& arg : action_string_args) {
        if (!arg.empty()) { token = arg; }
    }
    if (token.empty()) { return; }

    std::transform(token.begin(), token.end(), token.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if (token == "chance") {
        action_id_ = kPlacementChanceEventId;
        player_ = Player::kPlayerNone;
        return;
    }

    if (std::all_of(token.begin(), token.end(), [](unsigned char c) { return std::isdigit(c); })) {
        action_id_ = std::stoi(token);
        player_ = (action_id_ == kPlacementChanceEventId ? Player::kPlayerNone : Player::kPlayer1);
        return;
    }

    // format: "h<0|1>_x<N>_y<N>_o<N>_s<N>"
    bool use_hold = false;
    int lx = 0, ly = 0, orient = 0, spin = 0;
    if (std::sscanf(token.c_str(), "h%d_x%d_y%d_o%d_s%d",
                    reinterpret_cast<int*>(&use_hold), &lx, &ly, &orient, &spin) == 5) {
        action_id_ = packPlacementId(use_hold, lx, ly, orient, spin);
        player_ = Player::kPlayer1;
        return;
    }
}

std::string ModernTetrisPlacementAction::toConsoleString() const
{
    if (action_id_ == kPlacementChanceEventId) { return "chance"; }
    if (action_id_ < 0 || action_id_ >= kMaxPlacementActionId) { return "null"; }
    auto p = unpackPlacementId(action_id_);
    std::ostringstream oss;
    oss << "h" << p.use_hold
        << "_x" << p.lock_x
        << "_y" << p.lock_y
        << "_o" << p.orientation
        << "_s" << p.spin_type;
    return oss.str();
}

// --- Env core ---

void ModernTetrisPlacementEnv::reset(int seed)
{
    random_.seed(seed_ = seed);
    actions_.clear();
    events_.clear();
    observations_.clear();
    reward_ = 0.0f;
    total_reward_ = 0.0f;
    reward_prev_potential_ = 0.0f; // empty board has phi = 0
    placements_dirty_ = true;

    engine::step::Config step_config;
    step_config.piece_life = 0x7fffffff; // effectively disable lifetime expiration
    step_config.auto_drop = 0;
    engine::step::setConfig(&ctx_, step_config);
    engine::step::setSeed(&ctx_, static_cast<std::uint32_t>(seed_), static_cast<std::uint32_t>(seed_ ^ 0x9e3779b9U));
    engine::step::reset(&ctx_);
    turn_ = Player::kPlayer1;
}

void ModernTetrisPlacementEnv::setState(const engine::step::Context& ctx)
{
    ctx_ = ctx;
    actions_.clear();
    events_.clear();
    observations_.clear();
    reward_ = 0.0f;
    total_reward_ = 0.0f;
    placements_dirty_ = true;
    {
        using namespace minizero::env::moderntetris;
        const auto cfg = reward::RewardConfig::fromGlobals();
        reward_prev_potential_ = reward::computeBoardPotential(ctx_.state, cfg);
    }
    turn_ = Player::kPlayer1;
}

bool ModernTetrisPlacementEnv::act(const ModernTetrisPlacementAction& action, bool with_chance /* = true */)
{
    if (turn_ != Player::kPlayer1 || action.getPlayer() != Player::kPlayer1) { return false; }
    if (action.getActionID() < 0 || action.getActionID() >= kMaxPlacementActionId) { return false; }

    rebuildLegalPlacements();
    const CachedPlacement* found = nullptr;
    for (const auto& cp : cached_placements_) {
        if (cp.action_id == action.getActionID()) {
            found = &cp;
            break;
        }
    }
    if (!found) { return false; }

    // replay path through engine
    if (unpackPlacementId(action.getActionID()).use_hold) {
        engine::step::step(&ctx_, engine::step::Action::HOLD);
    }
    for (const auto& pa : found->result.path) {
        engine::step::step(&ctx_, placementActionToStepAction(pa));
    }
    // Capture the piece type + lock y before hard-drop advances state.current
    // and overwrites state.y. lock_y is the piece's settled y (top-left of its
    // 4x4 bbox) — used to index depth-keyed reward terms.
    const engine::PieceType locked_piece = ctx_.state.current;
    const int locked_y = static_cast<int>(found->result.lock_y);
    engine::step::step(&ctx_, engine::step::Action::HARD_DROP);

    // Inject random garbage after the hard-drop so the current piece's attack
    // only counters pre-existing queue entries (new garbage waits until the
    // next placement's processGarbageAndCounterAttack). Deterministic from the
    // state's garbage_seed stream -- loader replay reconstructs it identically.
    if (config::env_modern_tetris_garbage_probability > 0.0f && ctx_.state.is_alive) {
        auto& seed = ctx_.state.garbage_seed;
        const float u = static_cast<float>(xorshift32(seed)) / static_cast<float>(std::numeric_limits<std::uint32_t>::max());
        if (u < config::env_modern_tetris_garbage_probability) {
            const int lo = std::max(1, config::env_modern_tetris_garbage_min_lines);
            const int hi = std::max(lo, config::env_modern_tetris_garbage_max_lines);
            const std::uint32_t range = static_cast<std::uint32_t>(hi - lo + 1);
            const int lines = lo + static_cast<int>(xorshift32(seed) % range);
            const int delay = std::max(0, config::env_modern_tetris_garbage_delay);
            engine::addGarbage(&ctx_.state,
                               static_cast<std::uint8_t>(std::min(lines, 255)),
                               static_cast<std::uint8_t>(std::min(delay, 255)));
        }
    }

    placements_dirty_ = true;

    actions_.push_back(action);
    {
        using namespace minizero::env::moderntetris;
        const auto cfg = reward::RewardConfig::fromGlobals();
        const bool just_died = !ctx_.state.is_alive;
        float base = reward::computeLockBaseReward(ctx_.state, locked_piece, locked_y, just_died, cfg);
        float phi_new = reward::computeBoardPotential(ctx_.state, cfg);
        reward_ = base + (phi_new - reward_prev_potential_);
        reward_prev_potential_ = phi_new;
    }
    total_reward_ += reward_;
    turn_ = Player::kPlayerNone;
    if (with_chance) { return actChanceEvent(); }
    return true;
}

bool ModernTetrisPlacementEnv::actChanceEvent(const ModernTetrisPlacementAction& action)
{
    if (turn_ != Player::kPlayerNone || action.getActionID() != kPlacementChanceEventId || action.getPlayer() != Player::kPlayerNone) { return false; }
    events_.push_back(action);
    turn_ = Player::kPlayer1;
    return true;
}

bool ModernTetrisPlacementEnv::actChanceEvent()
{
    if (turn_ != Player::kPlayerNone) { return false; }
    events_.emplace_back(kPlacementChanceEventId, Player::kPlayerNone);
    turn_ = Player::kPlayer1;
    return true;
}

// --- Legal actions ---

void ModernTetrisPlacementEnv::rebuildLegalPlacements() const
{
    if (!placements_dirty_) { return; }
    cached_placements_.clear();

    // Collapse placements onto one legal action per canonical *position*, matching
    // fusion's canonical move set. Two things are merged:
    //   (1) rotation symmetry: I-North/I-South, O's four rotations, etc. land on
    //       the same cells (handled by canonicalizePlacement).
    //   (2) spin vs non-spin to the same cells: fusion classifies each landing
    //       position once, with spin winning over no-spin (NoSpin &= !spins). So
    //       when the same canonical position is reachable both as a spin and a
    //       plain drop, we keep the highest-value spin (full > mini > none) and
    //       its result (its final_state carries the spin's attack / b2b credit).
    // pos_key is the action id with the spin field zeroed -> the position identity.
    std::unordered_map<int, std::size_t> pos_to_index;

    const auto spinRank = [](engine::SpinType s) -> int {
        switch (s) {
            case engine::SpinType::SPIN: return 2;      // T-spin full
            case engine::SpinType::SPIN_MINI: return 1; // T-spin mini / all-spin mini
            default: return 0;                          // none
        }
    };

    const auto addPlacements = [&](engine::PieceType piece, bool use_hold,
                                   std::vector<engine::PlacementSearchResult>& placements) {
        for (auto& p : placements) {
            const CanonicalPlacement c = canonicalizePlacement(piece, p.lock_x, p.lock_y, p.orientation);
            // Store the canonical geometry so getActionDescriptors() reports the
            // same representative fusion would (the locked board in final_state
            // is unchanged; only this label collapses onto its canonical form).
            p.lock_x = static_cast<std::int8_t>(c.lock_x);
            p.lock_y = static_cast<std::int8_t>(c.lock_y);
            p.orientation = static_cast<std::uint8_t>(c.orientation);
            const int pos_key = packPlacementId(use_hold, c.lock_x, c.lock_y, c.orientation, 0);
            const int aid = packPlacementId(use_hold, c.lock_x, c.lock_y, c.orientation, static_cast<int>(p.spin_type));

            auto it = pos_to_index.find(pos_key);
            if (it == pos_to_index.end()) {
                pos_to_index.emplace(pos_key, cached_placements_.size());
                cached_placements_.push_back({aid, std::move(p)});
            } else if (spinRank(p.spin_type) > spinRank(cached_placements_[it->second].result.spin_type)) {
                cached_placements_[it->second].action_id = aid;
                cached_placements_[it->second].result = std::move(p);
            }
        }
    };

    // non-hold placements
    auto placements = engine::findPlacements(ctx_.state);
    addPlacements(ctx_.state.current, false, placements);

    // hold placements (only if not already held this turn)
    if (!ctx_.state.has_held) {
        engine::State hold_state = ctx_.state;
        engine::hold(&hold_state);
        if (hold_state.current != ctx_.state.current || ctx_.state.hold != engine::PieceType::NONE) {
            auto hold_placements = engine::findPlacements(hold_state);
            addPlacements(hold_state.current, true, hold_placements);
        }
    }

    placements_dirty_ = false;
}

std::vector<ModernTetrisPlacementAction> ModernTetrisPlacementEnv::getLegalActions() const
{
    if (turn_ != Player::kPlayer1 || isTerminal()) { return {}; }

    rebuildLegalPlacements();
    std::vector<ModernTetrisPlacementAction> legal_actions;
    legal_actions.reserve(cached_placements_.size());
    for (const auto& cp : cached_placements_) {
        legal_actions.emplace_back(cp.action_id, Player::kPlayer1);
    }
    return legal_actions;
}

std::vector<ModernTetrisPlacementAction> ModernTetrisPlacementEnv::getLegalChanceEvents() const
{
    if (turn_ != Player::kPlayerNone) { return {}; }
    return {ModernTetrisPlacementAction(kPlacementChanceEventId, Player::kPlayerNone)};
}

float ModernTetrisPlacementEnv::getChanceEventProbability(const ModernTetrisPlacementAction& action) const
{
    return isLegalChanceEvent(action) ? 1.0f : 0.0f;
}

bool ModernTetrisPlacementEnv::isLegalAction(const ModernTetrisPlacementAction& action) const
{
    if (turn_ != Player::kPlayer1 || isTerminal()) { return false; }
    rebuildLegalPlacements();
    for (const auto& cp : cached_placements_) {
        if (cp.action_id == action.getActionID()) { return true; }
    }
    return false;
}

bool ModernTetrisPlacementEnv::isLegalChanceEvent(const ModernTetrisPlacementAction& action) const
{
    return turn_ == Player::kPlayerNone && action.getPlayer() == Player::kPlayerNone && action.getActionID() == kPlacementChanceEventId;
}

bool ModernTetrisPlacementEnv::isTerminal() const
{
    return !ctx_.state.is_alive || static_cast<int>(actions_.size()) >= std::max(1, config::env_modern_tetris_max_episode_steps);
}

// --- Features ---
//
// The placement env feeds the placement_transformer network exclusively, which
// reads getBoardFeatures() / getGlobalFeatures() / getActionDescriptors()
// directly. The dense getFeatures()/getActionFeatures()/getChanceEventFeatures()
// virtuals from the AlphaZero/MuZero base interface are therefore unreachable
// here -- if anything ever invokes them we want a loud failure, not a
// silently-zeroed plane.

std::vector<float> ModernTetrisPlacementEnv::getFeatures(utils::Rotation /*rotation*/) const
{
    throw std::runtime_error{"ModernTetrisPlacementEnv::getFeatures() is not implemented; placement env uses getBoardFeatures()/getGlobalFeatures()/getActionDescriptors()"};
}

std::vector<float> ModernTetrisPlacementEnv::getActionFeatures(const ModernTetrisPlacementAction& /*action*/, utils::Rotation /*rotation*/) const
{
    throw std::runtime_error{"ModernTetrisPlacementEnv::getActionFeatures() is not implemented"};
}

std::vector<float> ModernTetrisPlacementEnv::getChanceEventFeatures(const ModernTetrisPlacementAction& /*event*/, utils::Rotation /*rotation*/) const
{
    throw std::runtime_error{"ModernTetrisPlacementEnv::getChanceEventFeatures() is not implemented"};
}

int ModernTetrisPlacementEnv::getNumInputChannels() const
{
    return kPlacementBoardChannels;
}

std::string ModernTetrisPlacementEnv::toString() const
{
    std::array<char, 4096> buffer{};
    engine::State state = ctx_.state;
    engine::toString(&state, buffer.data(), buffer.size());

    std::string result(buffer.data());
    // result += "\nLegal placements: " + std::to_string(cached_placements_.size());
    // if (!cached_placements_.empty()) {
    //     result += "\n";
    //     for (const auto& cp : cached_placements_) {
    //         auto p = unpackPlacementId(cp.action_id);
    //         result += "  [" + std::to_string(cp.action_id) + "] " + (p.use_hold ? "HOLD " : "") + "x=" + std::to_string(p.lock_x) + " y=" + std::to_string(p.lock_y) + " o=" + std::to_string(p.orientation) + " spin=" + std::to_string(p.spin_type) + "\n";
    //     }
    // }
    return result;
}

// --- Placement transformer feature APIs ---

std::vector<float> ModernTetrisPlacementEnv::getBoardFeatures() const
{
    std::vector<float> features(kPlacementBoardChannels * kVisibleCellCount, 0.0f);
    for (int y = engine::BOARD_TOP; y <= engine::BOARD_BOTTOM; ++y) {
        for (int x = engine::BOARD_LEFT; x <= engine::BOARD_RIGHT; ++x) {
            if (!isOccupied(engine::ops::getCell(ctx_.state.board, x, y))) { continue; }
            const int local_x = x - engine::BOARD_LEFT;
            const int local_y = y - engine::BOARD_TOP;
            features[local_y * kModernTetrisPlacementBoardWidth + local_x] = 1.0f;
        }
    }
    return features;
}

PlacementGlobalFeatures ModernTetrisPlacementEnv::getGlobalFeatures() const
{
    const int preview_size = std::clamp(config::env_modern_tetris_num_preview_piece, 0, 14);
    PlacementGlobalFeatures g;
    const int cur_idx = toPieceIndex(ctx_.state.current);
    g.current_piece = (cur_idx >= 0 && cur_idx < 7) ? cur_idx : -1;
    const int hold_idx = toPieceIndex(ctx_.state.hold);
    g.hold_piece = (hold_idx >= 0 && hold_idx < 7) ? hold_idx : -1;
    g.has_held = ctx_.state.has_held;
    g.preview.reserve(preview_size);
    for (int i = 0; i < preview_size; ++i) {
        const int p = toPieceIndex(ctx_.state.next[i]);
        g.preview.push_back((p >= 0 && p < 7) ? p : -1);
    }
    g.was_rotation = ctx_.state.was_last_rotation;
    g.srs_index = static_cast<int>(ctx_.state.srs_index);
    g.combo_count = ctx_.state.combo_count;
    g.back_to_back = ctx_.state.back_to_back_count > 0;
    int pending_garbage = 0;
    for (int i = 0; i < engine::GARBAGE_QUEUE_SIZE; ++i) { pending_garbage += ctx_.state.garbage_queue[i]; }
    g.pending_garbage = pending_garbage;
    return g;
}

std::vector<PlacementActionDescriptor> ModernTetrisPlacementEnv::getActionDescriptors() const
{
    rebuildLegalPlacements();
    std::vector<PlacementActionDescriptor> descs;
    descs.reserve(cached_placements_.size());
    for (const auto& cp : cached_placements_) {
        auto unpacked = unpackPlacementId(cp.action_id);
        PlacementActionDescriptor d;
        d.action_id = cp.action_id;
        d.use_hold = unpacked.use_hold;
        d.lock_x = static_cast<int>(cp.result.lock_x) - engine::BOARD_LEFT;
        d.lock_y = static_cast<int>(cp.result.lock_y) - engine::BOARD_TOP;
        d.orientation = static_cast<int>(cp.result.orientation);
        d.spin_type = static_cast<int>(cp.result.spin_type);
        // piece_type at the moment of lock = final_state's locked piece; pre-advance piece_type
        // is easier to derive: for no-hold it's ctx_.state.current; for hold it's ctx_.state.hold
        // if hold exists else preview[0].
        engine::PieceType piece_type;
        if (!unpacked.use_hold) {
            piece_type = ctx_.state.current;
        } else if (ctx_.state.hold != engine::PieceType::NONE) {
            piece_type = ctx_.state.hold;
        } else {
            piece_type = ctx_.state.next[0];
        }
        const int pt_idx = toPieceIndex(piece_type);
        d.piece_type = (pt_idx >= 0 && pt_idx < 7) ? pt_idx : 0;
        d.lines_cleared = static_cast<int>(cp.result.final_state.lines_cleared);
        descs.push_back(d);
    }
    return descs;
}

// --- Helpers ---

int ModernTetrisPlacementEnv::toPieceIndex(engine::PieceType piece_type)
{
    return static_cast<int>(piece_type);
}

bool ModernTetrisPlacementEnv::isOccupied(engine::Cell cell)
{
    return cell != engine::Cell::EMPTY;
}

engine::step::Action ModernTetrisPlacementEnv::placementActionToStepAction(engine::PlacementAction pa)
{
    switch (pa) {
        case engine::PlacementAction::LEFT: return engine::step::Action::MOVE_LEFT;
        case engine::PlacementAction::RIGHT: return engine::step::Action::MOVE_RIGHT;
        case engine::PlacementAction::LEFT_WALL: return engine::step::Action::MOVE_LEFT_TO_WALL;
        case engine::PlacementAction::RIGHT_WALL: return engine::step::Action::MOVE_RIGHT_TO_WALL;
        case engine::PlacementAction::SOFT_DROP: return engine::step::Action::SOFT_DROP;
        case engine::PlacementAction::SOFT_DROP_FLOOR: return engine::step::Action::SOFT_DROP_TO_FLOOR;
        case engine::PlacementAction::ROTATE_CW: return engine::step::Action::ROTATE_CW;
        case engine::PlacementAction::ROTATE_CCW: return engine::step::Action::ROTATE_CCW;
        case engine::PlacementAction::ROTATE_180: return engine::step::Action::ROTATE_180;
        default: return engine::step::Action::NOOP;
    }
}

// --- EnvLoader ---
//
// Same rationale as the env-side feature stubs: placement training reads
// board/global/per-action tensors directly via setPlacementTrainingData(),
// never these AlphaZero/MuZero-shaped getters.

std::vector<float> ModernTetrisPlacementEnvLoader::getActionFeatures(const int /*pos*/, utils::Rotation /*rotation*/) const
{
    throw std::runtime_error{"ModernTetrisPlacementEnvLoader::getActionFeatures() is not implemented"};
}

std::vector<float> ModernTetrisPlacementEnvLoader::getChanceEventFeatures(const int /*pos*/, utils::Rotation /*rotation*/) const
{
    throw std::runtime_error{"ModernTetrisPlacementEnvLoader::getChanceEventFeatures() is not implemented"};
}

std::vector<float> ModernTetrisPlacementEnvLoader::getChance(const int /*pos*/, utils::Rotation /*rotation*/) const
{
    throw std::runtime_error{"ModernTetrisPlacementEnvLoader::getChance() is not implemented"};
}

float ModernTetrisPlacementEnvLoader::calculateNStepValue(const int pos) const
{
    assert(pos < static_cast<int>(action_pairs_.size()));

    const int n_step = config::learner_n_step_return;
    const float discount = config::actor_mcts_reward_discount;
    const size_t bootstrap_index = pos + n_step;
    float value = 0.0f;
    const float n_step_value = (bootstrap_index < action_pairs_.size()) ? std::pow(discount, n_step) * BaseEnvLoader::getValue(bootstrap_index)[0] : 0.0f;
    for (size_t index = pos; index < std::min(bootstrap_index, action_pairs_.size()); ++index) {
        value += std::pow(discount, index - pos) * BaseEnvLoader::getReward(index)[0];
    }
    return value + n_step_value;
}

std::vector<float> ModernTetrisPlacementEnvLoader::toDiscreteValue(float value) const
{
    std::vector<float> discrete_value(kModernTetrisPlacementDiscreteValueSize, 0.0f);
    const int value_floor = std::floor(value);
    const int value_ceil = std::ceil(value);
    const int shift = kModernTetrisPlacementDiscreteValueSize / 2;
    const int value_floor_shift = std::clamp(value_floor + shift, 0, kModernTetrisPlacementDiscreteValueSize - 1);
    const int value_ceil_shift = std::clamp(value_ceil + shift, 0, kModernTetrisPlacementDiscreteValueSize - 1);
    if (value_floor == value_ceil) {
        discrete_value[value_floor_shift] = 1.0f;
    } else {
        discrete_value[value_floor_shift] = value_ceil - value;
        discrete_value[value_ceil_shift] = value - value_floor;
    }
    return discrete_value;
}

} // namespace minizero::env::moderntetris_placement
