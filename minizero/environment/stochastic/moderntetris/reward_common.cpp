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
    c.b2b_bonus = config::env_modern_tetris_reward_b2b_bonus;
    c.combo_bonus = config::env_modern_tetris_reward_combo_bonus;
    c.perfect_clear_bonus = config::env_modern_tetris_reward_perfect_clear_bonus;
    c.height_weight = config::env_modern_tetris_reward_height_weight;
    c.hole_weight = config::env_modern_tetris_reward_hole_weight;
    return c;
}

float computeLockBaseReward(const engine::State& post, bool just_died, const RewardConfig& cfg)
{
    float r = cfg.survival_bonus;
    if (just_died) { r -= cfg.death_penalty; }

    r += cfg.lines_sent_weight * static_cast<float>(post.lines_sent);

    switch (post.lines_cleared) {
        case 1: r += cfg.clear_1; break;
        case 2: r += cfg.clear_2; break;
        case 3: r += cfg.clear_3; break;
        case 4: r += cfg.clear_4; break;
        default: break;
    }

    if (post.lines_cleared > 0) {
        if (post.spin_type == engine::SpinType::SPIN) {
            r += cfg.tspin_bonus;
        } else if (post.spin_type == engine::SpinType::SPIN_MINI) {
            r += cfg.tspin_mini_bonus;
        }
        if (post.back_to_back_count > 0) { r += cfg.b2b_bonus; }
        if (post.combo_count > 0) { r += cfg.combo_bonus * static_cast<float>(post.combo_count); }
    }

    if (post.perfect_clear) { r += cfg.perfect_clear_bonus; }

    return r;
}

float computeBoardPotential(const engine::State& state, const RewardConfig& cfg)
{
    if (cfg.height_weight == 0.0f && cfg.hole_weight == 0.0f) { return 0.0f; }

    int total_height = 0;
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
        total_height += (engine::BOARD_BOTTOM + 1 - top_y);
        // Holes below the top-most occupied cell.
        for (int y = top_y + 1; y <= engine::BOARD_BOTTOM; ++y) {
            if (engine::ops::getCell(state.board, x, y) == engine::Cell::EMPTY) {
                ++total_holes;
            }
        }
    }
    return -(cfg.height_weight * static_cast<float>(total_height) + cfg.hole_weight * static_cast<float>(total_holes));
}

} // namespace minizero::env::moderntetris::reward
