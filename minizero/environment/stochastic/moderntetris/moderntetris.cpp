#include "moderntetris.h"
#include "configuration.h"
#include "random.h"
#include "reward_common.h"
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>

namespace minizero::env::moderntetris {

using namespace minizero::utils;

namespace {

    constexpr int kVisibleCellCount = kModernTetrisBoardWidth * kModernTetrisBoardHeight;

    std::string normalizeToken(std::string token)
    {
        std::transform(token.begin(), token.end(), token.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return token;
    }

} // namespace

const std::array<std::string, kModernTetrisActionSize + 1> kModernTetrisActionName = {
    "left",
    "right",
    "soft_drop",
    "hard_drop",
    "rotate_cw",
    "rotate_ccw",
    "rotate_180",
    "hold",
    "left_wall",
    "right_wall",
    "soft_drop_floor",
    "chance",
};

void initialize() {}

ModernTetrisAction::ModernTetrisAction(const std::vector<std::string>& action_string_args)
{
    action_id_ = -1;
    player_ = Player::kPlayerNone;

    std::string token;
    for (const auto& arg : action_string_args) {
        if (!arg.empty()) { token = arg; }
    }
    if (token.empty()) { return; }

    token = normalizeToken(token);
    if (std::all_of(token.begin(), token.end(), [](unsigned char c) { return std::isdigit(c); })) {
        action_id_ = std::stoi(token);
        player_ = (action_id_ == kModernTetrisActionSize ? Player::kPlayerNone : Player::kPlayer1);
        return;
    }

    auto it = std::find(kModernTetrisActionName.begin(), kModernTetrisActionName.end(), token);
    if (it == kModernTetrisActionName.end()) { return; }
    action_id_ = static_cast<int>(std::distance(kModernTetrisActionName.begin(), it));
    player_ = (action_id_ == kModernTetrisActionSize ? Player::kPlayerNone : Player::kPlayer1);
}

std::string ModernTetrisAction::toConsoleString() const
{
    if (action_id_ >= 0 && action_id_ < static_cast<int>(kModernTetrisActionName.size())) { return kModernTetrisActionName[action_id_]; }
    return "null";
}

void ModernTetrisEnv::reset(int seed)
{
    random_.seed(seed_ = seed);
    actions_.clear();
    events_.clear();
    observations_.clear();
    reward_ = 0.0f;
    total_reward_ = 0.0f;
    reward_prev_potential_ = 0.0f;

    engine::step::Config step_config;
    step_config.piece_life = std::max(1, config::env_modern_tetris_piece_lifetime);
    step_config.auto_drop = config::env_modern_tetris_auto_drop ? 1 : 0;
    engine::step::setConfig(&ctx_, step_config);
    engine::step::setSeed(&ctx_, static_cast<std::uint32_t>(seed_), static_cast<std::uint32_t>(seed_ ^ 0x9e3779b9U));
    engine::step::reset(&ctx_);
    resetActivePieceHistory();
    turn_ = Player::kPlayer1;
}

bool ModernTetrisEnv::act(const ModernTetrisAction& action, bool with_chance /* = true */)
{
    if (turn_ != Player::kPlayer1 || action.getPlayer() != Player::kPlayer1 || action.getActionID() < 0 || action.getActionID() >= kModernTetrisActionSize) { return false; }

    const std::vector<float> previous_active_plane = captureActivePiecePlane();
    const auto previous_piece_count = ctx_.state.piece_count;
    if (!isActionEffective(action)) { return false; }
    engine::step::Info info = engine::step::step(&ctx_, toEngineAction(action.getActionID()));

    const bool piece_was_replaced = (ctx_.state.piece_count != previous_piece_count) || (action.getActionID() == static_cast<int>(engine::step::Action::HOLD) && info.action_success);
    if (piece_was_replaced) {
        resetActivePieceHistory();
    } else {
        pushActivePieceHistory(previous_active_plane);
    }

    actions_.push_back(action);
    {
        // Reward shaping: only fire at lock events. For step-level env, most
        // steps are navigation (move/rotate/hold) and don't lock a piece. We
        // detect a lock by piece_count advancing.
        const bool lock_happened = (ctx_.state.piece_count != previous_piece_count);
        if (lock_happened) {
            const auto cfg = reward::RewardConfig::fromGlobals();
            const bool just_died = !ctx_.state.is_alive;
            float base = reward::computeLockBaseReward(ctx_.state, just_died, cfg);
            float phi_new = reward::computeBoardPotential(ctx_.state, cfg);
            reward_ = base + (phi_new - reward_prev_potential_);
            reward_prev_potential_ = phi_new;
        } else if (!ctx_.state.is_alive) {
            // Non-lock step that still kills (rare edge case): apply death penalty once.
            const auto cfg = reward::RewardConfig::fromGlobals();
            reward_ = -cfg.death_penalty;
        } else {
            reward_ = 0.0f;
        }
    }
    total_reward_ += reward_;
    turn_ = Player::kPlayerNone;
    if (with_chance) { return actChanceEvent(); }
    return true;
}

bool ModernTetrisEnv::actChanceEvent(const ModernTetrisAction& action)
{
    if (turn_ != Player::kPlayerNone || action.getActionID() != kModernTetrisActionSize || action.getPlayer() != Player::kPlayerNone) { return false; }
    events_.push_back(action);
    turn_ = Player::kPlayer1;
    return true;
}

bool ModernTetrisEnv::actChanceEvent()
{
    if (turn_ != Player::kPlayerNone) { return false; }
    events_.emplace_back(kModernTetrisActionSize, Player::kPlayerNone);
    turn_ = Player::kPlayer1;
    return true;
}

std::vector<ModernTetrisAction> ModernTetrisEnv::getLegalActions() const
{
    if (turn_ != Player::kPlayer1 || isTerminal()) { return {}; }

    std::vector<ModernTetrisAction> legal_actions;
    for (int action_id = 0; action_id < kModernTetrisActionSize; ++action_id) {
        ModernTetrisAction action(action_id, Player::kPlayer1);
        if (isActionEffective(action)) { legal_actions.push_back(action); }
    }
    return legal_actions;
}

std::vector<ModernTetrisAction> ModernTetrisEnv::getLegalChanceEvents() const
{
    if (turn_ != Player::kPlayerNone) { return {}; }
    return {ModernTetrisAction(kModernTetrisActionSize, Player::kPlayerNone)};
}

float ModernTetrisEnv::getChanceEventProbability(const ModernTetrisAction& action) const
{
    return isLegalChanceEvent(action) ? 1.0f : 0.0f;
}

bool ModernTetrisEnv::isLegalAction(const ModernTetrisAction& action) const
{
    if (turn_ != Player::kPlayer1 || isTerminal()) { return false; }
    return isActionEffective(action);
}

bool ModernTetrisEnv::isLegalChanceEvent(const ModernTetrisAction& action) const
{
    return turn_ == Player::kPlayerNone && action.getPlayer() == Player::kPlayerNone && action.getActionID() == kModernTetrisActionSize;
}

bool ModernTetrisEnv::isTerminal() const
{
    return !ctx_.state.is_alive || static_cast<int>(actions_.size()) >= std::max(1, config::env_modern_tetris_max_episode_steps);
}

std::vector<float> ModernTetrisEnv::getFeatures(utils::Rotation rotation /*= utils::Rotation::kRotationNone*/) const
{
    const int preview_size = std::clamp(config::env_modern_tetris_num_preview_piece, 0, 14);
    const int history_length = std::max(0, config::env_modern_tetris_history_length);
    std::vector<float> features(getNumInputChannels() * kVisibleCellCount, 0.0f);

    writeBoardFeatures(features);
    int channel_offset = 2;
    for (int i = 0; i < history_length; ++i) {
        if (i >= static_cast<int>(active_piece_history_.size())) { break; }
        std::copy(active_piece_history_[i].begin(),
                  active_piece_history_[i].end(),
                  features.begin() + (channel_offset + i) * kVisibleCellCount);
    }

    channel_offset += history_length;
    writePieceFeatures(features, channel_offset, ctx_.state.current);
    channel_offset += 7;
    writePieceFeatures(features, channel_offset, ctx_.state.hold);
    channel_offset += 7;
    for (int i = 0; i < preview_size; ++i) {
        writePieceFeatures(features, channel_offset + i * 7, ctx_.state.next[i]);
    }

    const int status_channel = channel_offset + preview_size * 7;
    fillScalarPlane(features, status_channel + 0, ctx_.state.has_held ? 1.0f : 0.0f);
    fillScalarPlane(features, status_channel + 1, ctx_.state.was_last_rotation ? 1.0f : 0.0f);
    fillOneHotPlane(features, status_channel + 2, 7, std::clamp(static_cast<int>(ctx_.state.srs_index) + 1, 0, 6));
    fillScalarPlane(features, status_channel + 9, static_cast<float>(ctx_.lifetime) / std::max(1, config::env_modern_tetris_piece_lifetime));
    fillScalarPlane(features, status_channel + 10, std::clamp(static_cast<float>(ctx_.state.combo_count + 1) / 10.0f, 0.0f, 1.0f));
    fillScalarPlane(features, status_channel + 11, ctx_.state.back_to_back_count > 0 ? 1.0f : 0.0f);
    int pending_garbage = 0;
    for (int i = 0; i < engine::GARBAGE_QUEUE_SIZE; ++i) { pending_garbage += ctx_.state.garbage_queue[i]; }
    fillScalarPlane(features, status_channel + 12, std::clamp(static_cast<float>(pending_garbage) / 20.0f, 0.0f, 1.0f));

    return features;
}

std::vector<float> ModernTetrisEnv::getActionFeatures(const ModernTetrisAction& action, utils::Rotation rotation /*= utils::Rotation::kRotationNone*/) const
{
    std::vector<float> action_features(kModernTetrisActionSize * kVisibleCellCount, 0.0f);
    if (action.getActionID() >= 0 && action.getActionID() < kModernTetrisActionSize) {
        std::fill(action_features.begin() + action.getActionID() * kVisibleCellCount,
                  action_features.begin() + (action.getActionID() + 1) * kVisibleCellCount,
                  1.0f);
    }
    return action_features;
}

std::vector<float> ModernTetrisEnv::getChanceEventFeatures(const ModernTetrisAction& event, utils::Rotation rotation /*= utils::Rotation::kRotationNone*/) const
{
    return std::vector<float>(kVisibleCellCount, 1.0f);
}

int ModernTetrisEnv::getNumInputChannels() const
{
    return getChannelCount(std::clamp(config::env_modern_tetris_num_preview_piece, 0, 14), std::max(0, config::env_modern_tetris_history_length));
}

std::string ModernTetrisEnv::toString() const
{
    std::array<char, 4096> buffer{};
    engine::State state = ctx_.state;
    engine::toString(&state, buffer.data(), buffer.size());
    return std::string(buffer.data());
}

engine::step::Action ModernTetrisEnv::toEngineAction(int action_id)
{
    static const std::array<engine::step::Action, kModernTetrisActionSize> kActionMap = {
        engine::step::Action::MOVE_LEFT,
        engine::step::Action::MOVE_RIGHT,
        engine::step::Action::SOFT_DROP,
        engine::step::Action::HARD_DROP,
        engine::step::Action::ROTATE_CW,
        engine::step::Action::ROTATE_CCW,
        engine::step::Action::ROTATE_180,
        engine::step::Action::HOLD,
        engine::step::Action::MOVE_LEFT_TO_WALL,
        engine::step::Action::MOVE_RIGHT_TO_WALL,
        engine::step::Action::SOFT_DROP_TO_FLOOR,
    };
    return kActionMap[action_id];
}

int ModernTetrisEnv::toPieceIndex(engine::PieceType piece_type)
{
    return static_cast<int>(piece_type);
}

bool ModernTetrisEnv::isOccupied(engine::Cell cell)
{
    return cell != engine::Cell::EMPTY;
}

int ModernTetrisEnv::getChannelCount(int preview_size, int history_length)
{
    return 2 + history_length + 7 + 7 + preview_size * 7 + 13;
}

bool ModernTetrisEnv::isActionEffective(const ModernTetrisAction& action) const
{
    if (action.getPlayer() != Player::kPlayer1 || action.getActionID() < 0 || action.getActionID() >= kModernTetrisActionSize) { return false; }
    engine::step::Context transition = ctx_;
    engine::step::Action engine_action = toEngineAction(action.getActionID());
    engine::step::Info info = engine::step::step(&transition, engine_action);
    return info.action_success || engine_action == engine::step::Action::HARD_DROP;
}

std::vector<float> ModernTetrisEnv::captureActivePiecePlane() const
{
    std::vector<float> plane(kVisibleCellCount, 0.0f);
    if (ctx_.state.current == engine::PieceType::NONE) { return plane; }

    const engine::Piece& active_piece = engine::ops::getPiece(ctx_.state.current, ctx_.state.orientation);
    for (int piece_y = 0; piece_y < 4; ++piece_y) {
        for (int piece_x = 0; piece_x < 4; ++piece_x) {
            if (!isOccupied(engine::ops::getCell(active_piece, piece_x, piece_y))) { continue; }
            const int board_x = ctx_.state.x + piece_x;
            const int board_y = ctx_.state.y + piece_y;
            if (board_x < engine::BOARD_LEFT || board_x > engine::BOARD_RIGHT || board_y < engine::BOARD_TOP || board_y > engine::BOARD_BOTTOM) { continue; }
            const int local_x = board_x - engine::BOARD_LEFT;
            const int local_y = board_y - engine::BOARD_TOP;
            plane[local_y * kModernTetrisBoardWidth + local_x] = 1.0f;
        }
    }
    return plane;
}

void ModernTetrisEnv::resetActivePieceHistory()
{
    active_piece_history_.clear();
}

void ModernTetrisEnv::pushActivePieceHistory(const std::vector<float>& plane)
{
    const int history_length = std::max(0, config::env_modern_tetris_history_length);
    if (history_length == 0) { return; }
    active_piece_history_.push_front(plane);
    while (static_cast<int>(active_piece_history_.size()) > history_length) { active_piece_history_.pop_back(); }
}

void ModernTetrisEnv::writeBoardFeatures(std::vector<float>& features) const
{
    for (int y = engine::BOARD_TOP; y <= engine::BOARD_BOTTOM; ++y) {
        for (int x = engine::BOARD_LEFT; x <= engine::BOARD_RIGHT; ++x) {
            const int local_x = x - engine::BOARD_LEFT;
            const int local_y = y - engine::BOARD_TOP;
            const int position = local_y * kModernTetrisBoardWidth + local_x;
            if (isOccupied(engine::ops::getCell(ctx_.state.board, x, y))) { features[position] = 1.0f; }
        }
    }

    const std::vector<float> current_active_plane = captureActivePiecePlane();
    std::copy(current_active_plane.begin(), current_active_plane.end(), features.begin() + kVisibleCellCount);
}

void ModernTetrisEnv::writePieceFeatures(std::vector<float>& features, int channel_offset, engine::PieceType piece_type) const
{
    const int piece_index = toPieceIndex(piece_type);
    if (piece_index < 0 || piece_index >= 7) { return; }
    std::fill(features.begin() + (channel_offset + piece_index) * kVisibleCellCount,
              features.begin() + (channel_offset + piece_index + 1) * kVisibleCellCount,
              1.0f);
}

void ModernTetrisEnv::fillScalarPlane(std::vector<float>& features, int channel, float value) const
{
    std::fill(features.begin() + channel * kVisibleCellCount,
              features.begin() + (channel + 1) * kVisibleCellCount,
              value);
}

void ModernTetrisEnv::fillOneHotPlane(std::vector<float>& features, int channel_offset, int size, int index) const
{
    if (index < 0 || index >= size) { return; }
    std::fill(features.begin() + (channel_offset + index) * kVisibleCellCount,
              features.begin() + (channel_offset + index + 1) * kVisibleCellCount,
              1.0f);
}

std::vector<float> ModernTetrisEnvLoader::getActionFeatures(const int pos, utils::Rotation rotation /*= utils::Rotation::kRotationNone*/) const
{
    ModernTetrisAction action = (pos < static_cast<int>(action_pairs_.size())) ? action_pairs_[pos].first : ModernTetrisAction(0, Player::kPlayer1);
    return ModernTetrisEnv().getActionFeatures(action, rotation);
}

std::vector<float> ModernTetrisEnvLoader::getChanceEventFeatures(const int pos, utils::Rotation rotation /*= utils::Rotation::kRotationNone*/) const
{
    return ModernTetrisEnv().getChanceEventFeatures(ModernTetrisAction(kModernTetrisActionSize, Player::kPlayerNone), rotation);
}

std::vector<float> ModernTetrisEnvLoader::getChance(const int pos, utils::Rotation rotation /*= utils::Rotation::kRotationNone*/) const
{
    return std::vector<float>(kModernTetrisChanceEventSize, 1.0f);
}

float ModernTetrisEnvLoader::calculateNStepValue(const int pos) const
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

std::vector<float> ModernTetrisEnvLoader::toDiscreteValue(float value) const
{
    std::vector<float> discrete_value(kModernTetrisDiscreteValueSize, 0.0f);
    const int value_floor = std::floor(value);
    const int value_ceil = std::ceil(value);
    const int shift = kModernTetrisDiscreteValueSize / 2;
    const int value_floor_shift = std::clamp(value_floor + shift, 0, kModernTetrisDiscreteValueSize - 1);
    const int value_ceil_shift = std::clamp(value_ceil + shift, 0, kModernTetrisDiscreteValueSize - 1);
    if (value_floor == value_ceil) {
        discrete_value[value_floor_shift] = 1.0f;
    } else {
        discrete_value[value_floor_shift] = value_ceil - value;
        discrete_value[value_ceil_shift] = value - value_floor;
    }
    return discrete_value;
}

} // namespace minizero::env::moderntetris
