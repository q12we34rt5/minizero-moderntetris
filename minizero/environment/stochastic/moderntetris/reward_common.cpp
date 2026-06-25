#include "reward_common.h"
#include "configuration.h"

#include <cassert>

namespace minizero::env::moderntetris::reward {

namespace {
    // Jstris combo table, mirroring engine::calculateAttack (the engine keeps that
    // function private and the table is frozen, so a small copy here is fine).
    // combo_count: -1 = no combo, 0 = first clear, 1 = second consecutive, ...
    int comboValue(int combo_count)
    {
        constexpr int combo_table[] = {0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5};
        constexpr int n = sizeof(combo_table) / sizeof(combo_table[0]);
        if (combo_count < 0) { return 0; }
        return combo_table[combo_count < n ? combo_count : n - 1];
    }
} // namespace

RewardConfig RewardConfig::fromGlobals()
{
    RewardConfig c;
    c.survival_bonus = config::env_modern_tetris_reward_survival_bonus;
    c.death_penalty = config::env_modern_tetris_reward_death_penalty;
    c.attack_single = config::env_modern_tetris_reward_attack_single;
    c.attack_double = config::env_modern_tetris_reward_attack_double;
    c.attack_triple = config::env_modern_tetris_reward_attack_triple;
    c.attack_tetris = config::env_modern_tetris_reward_attack_tetris;
    c.attack_tspin_single = config::env_modern_tetris_reward_attack_tspin_single;
    c.attack_tspin_double = config::env_modern_tetris_reward_attack_tspin_double;
    c.attack_tspin_triple = config::env_modern_tetris_reward_attack_tspin_triple;
    c.attack_tspin_mini_single = config::env_modern_tetris_reward_attack_tspin_mini_single;
    c.attack_tspin_mini_double = config::env_modern_tetris_reward_attack_tspin_mini_double;
    c.attack_allspin = config::env_modern_tetris_reward_attack_allspin;
    c.attack_pc = config::env_modern_tetris_reward_attack_pc;
    c.attack_b2b = config::env_modern_tetris_reward_attack_b2b;
    c.attack_combo_weight = config::env_modern_tetris_reward_attack_combo_weight;
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

    // Weighted decomposition of the engine's raw (pre-garbage-counter) attack,
    // read straight from the post-lock state plus the piece that locked. With
    // every attack_* at its engine default this reproduces state->attack exactly;
    // zero components to ablate. (engine::calculateAttack is the canonical rule.)
    // The bucket is scaled by a depth multiplier below, so the agent learns to
    // prefer clearing deep over clearing high.
    float clear_reward = 0.0f;
    const int lines = post.lines_cleared;
    if (lines > 0) {
        if (post.perfect_clear) {
            // Perfect clear overrides the line/spin value (mirrors calculateAttack).
            clear_reward += cfg.attack_pc;
        } else if (post.spin_type == engine::SpinType::SPIN) {
            // Full T-spin (SpinType::SPIN is only ever produced for a T).
            assert(locked_piece == engine::PieceType::T);
            assert(lines >= 1 && lines <= 3);
            clear_reward += (lines == 1) ? cfg.attack_tspin_single
                                         : (lines == 2) ? cfg.attack_tspin_double
                                                        : cfg.attack_tspin_triple;
        } else if (post.spin_type == engine::SpinType::SPIN_MINI && locked_piece == engine::PieceType::T) {
            // Mini T-spin (a real mini, or an immobile T under the all-spin ruleset).
            // No mini-triple knob: a 3-line clear is never a T mini, so use the triple.
            clear_reward += (lines == 1) ? cfg.attack_tspin_mini_single
                                         : (lines == 2) ? cfg.attack_tspin_mini_double
                                                        : cfg.attack_triple;
        } else {
            // Normal clear, and non-T all-spins (SPIN_MINI on a non-T, which the
            // engine also scores as a normal clear) that additionally earn
            // attack_allspin on top.
            clear_reward += (lines == 1) ? cfg.attack_single
                                         : (lines == 2) ? cfg.attack_double
                                                        : (lines == 3) ? cfg.attack_triple
                                                                       : cfg.attack_tetris;
            if (post.spin_type == engine::SpinType::SPIN_MINI) { clear_reward += cfg.attack_allspin; }
        }
        // Qualifying back-to-back: a Tetris or any spin while a streak is active.
        if (post.back_to_back_count > 0 && (lines == 4 || post.spin_type != engine::SpinType::NONE)) {
            clear_reward += cfg.attack_b2b;
        }
        // Combo: the engine's saturating combo-table value, weighted.
        clear_reward += cfg.attack_combo_weight * static_cast<float>(comboValue(post.combo_count));
    }

    // Depth-keyed multiplier on the whole attack bucket. locked_y < 0 leaves
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
