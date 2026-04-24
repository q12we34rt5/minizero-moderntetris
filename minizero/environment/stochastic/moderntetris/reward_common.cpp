#include "reward_common.h"
#include "configuration.h"

namespace minizero::env::moderntetris::reward {

RewardConfig RewardConfig::fromGlobals()
{
    RewardConfig c;
    c.survival_bonus = config::env_modern_tetris_reward_survival_bonus;
    c.death_penalty = config::env_modern_tetris_reward_death_penalty;
    c.lines_sent_weight = config::env_modern_tetris_reward_lines_sent_weight;
    c.clear_1 = config::env_modern_tetris_reward_clear_1;
    c.clear_2 = config::env_modern_tetris_reward_clear_2;
    c.clear_3 = config::env_modern_tetris_reward_clear_3;
    c.clear_4 = config::env_modern_tetris_reward_clear_4;
    c.tspin_bonus = config::env_modern_tetris_reward_tspin_bonus;
    c.tspin_mini_bonus = config::env_modern_tetris_reward_tspin_mini_bonus;
    c.all_spin_bonus = config::env_modern_tetris_reward_all_spin_bonus;
    c.b2b_bonus = config::env_modern_tetris_reward_b2b_bonus;
    c.combo_bonus = config::env_modern_tetris_reward_combo_bonus;
    c.perfect_clear_bonus = config::env_modern_tetris_reward_perfect_clear_bonus;
    c.clear_depth_weight_bottom = config::env_modern_tetris_reward_clear_depth_weight_bottom;
    c.clear_depth_weight_top = config::env_modern_tetris_reward_clear_depth_weight_top;
    c.height_weight = config::env_modern_tetris_reward_height_weight;
    c.hole_weight = config::env_modern_tetris_reward_hole_weight;
    return c;
}

float computeLockBaseReward(const engine::State& post, engine::PieceType locked_piece, int locked_y, bool just_died, const RewardConfig& cfg)
{
    float r = cfg.survival_bonus;
    if (just_died) { r -= cfg.death_penalty; }

    // Accumulate every reward term tied to clearing lines. This whole bucket is
    // scaled by a depth multiplier below, so the agent learns to prefer
    // clearing deep over clearing high.
    float clear_reward = cfg.lines_sent_weight * static_cast<float>(post.lines_sent);

    switch (post.lines_cleared) {
        case 1: clear_reward += cfg.clear_1; break;
        case 2: clear_reward += cfg.clear_2; break;
        case 3: clear_reward += cfg.clear_3; break;
        case 4: clear_reward += cfg.clear_4; break;
        default: break;
    }

    if (post.lines_cleared > 0) {
        // Engine convention: for T pieces, spin_type is SPIN (full T-spin) or
        // SPIN_MINI (T-spin mini). For non-T pieces, any detected spin is
        // reported as SPIN_MINI — we must inspect locked_piece to classify.
        if (post.spin_type == engine::SpinType::SPIN || post.spin_type == engine::SpinType::SPIN_MINI) {
            if (locked_piece == engine::PieceType::T) {
                clear_reward += (post.spin_type == engine::SpinType::SPIN) ? cfg.tspin_bonus : cfg.tspin_mini_bonus;
            } else {
                clear_reward += cfg.all_spin_bonus;
            }
        }
        if (post.back_to_back_count > 0) { clear_reward += cfg.b2b_bonus; }
        if (post.combo_count > 0) { clear_reward += cfg.combo_bonus * static_cast<float>(post.combo_count); }
    }

    if (post.perfect_clear) { clear_reward += cfg.perfect_clear_bonus; }

    // Depth-keyed multiplier on the whole clear bucket. locked_y < 0 leaves
    // the weight at 1.0 (callers without a pre-lock y are unaffected).
    float clear_weight = 1.0f;
    if (locked_y >= 0) {
        constexpr float kSpan = static_cast<float>(engine::BOARD_BOTTOM - engine::BOARD_TOP);
        float t = static_cast<float>(locked_y - engine::BOARD_TOP) / kSpan;
        if (t < 0.0f) { t = 0.0f; }
        if (t > 1.0f) { t = 1.0f; }
        clear_weight = cfg.clear_depth_weight_top + (cfg.clear_depth_weight_bottom - cfg.clear_depth_weight_top) * t;
    }
    r += clear_weight * clear_reward;

    return r;
}

float computeBoardPotential(const engine::State& state, const RewardConfig& cfg)
{
    if (cfg.height_weight == 0.0f && cfg.hole_weight == 0.0f) { return 0.0f; }

    int max_height = 0;
    int total_holes = 0;
    for (int x = engine::BOARD_LEFT; x <= engine::BOARD_RIGHT; ++x) {
        // Find top-most occupied cell in column.
        int top_y = engine::BOARD_BOTTOM + 1;
        for (int y = engine::BOARD_TOP; y <= engine::BOARD_BOTTOM; ++y) {
            if (engine::ops::getCell(state.board, x, y) != engine::Cell::EMPTY) {
                top_y = y;
                break;
            }
        }
        const int col_height = engine::BOARD_BOTTOM + 1 - top_y;
        if (col_height > max_height) { max_height = col_height; }
        // Holes below the top-most occupied cell.
        for (int y = top_y + 1; y <= engine::BOARD_BOTTOM; ++y) {
            if (engine::ops::getCell(state.board, x, y) == engine::Cell::EMPTY) {
                ++total_holes;
            }
        }
    }
    return -(cfg.height_weight * static_cast<float>(max_height) + cfg.hole_weight * static_cast<float>(total_holes));
}

} // namespace minizero::env::moderntetris::reward
