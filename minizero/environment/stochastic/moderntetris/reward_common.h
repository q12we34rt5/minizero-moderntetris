#pragma once

#include "engine/tetris.hpp"

namespace minizero::env::moderntetris::reward {

/**
 * Per-lock reward shaping configuration. All weights are additive and can be
 * set to zero to disable that term. Caller is responsible for reading the
 * config globals via RewardConfig::fromGlobals().
 *
 * Structure of a typical reward per lock event:
 *
 *   base    = survival_bonus - (just_died ? death_penalty : 0)
 *   attack  = lines_sent_weight * post.lines_sent
 *   shape   = clear_N[post.lines_cleared] + spin_bonus(post) + b2b_bonus
 *           + combo_bonus * post.combo_count + perfect_clear_bonus?
 *   pot     = phi(post) - phi(pre)     where phi is potential-based shaping
 *
 *   total   = base + attack + shape + pot
 *
 * base + attack + shape is computed by computeLockBaseReward().
 * phi is computed by computeBoardPotential(). Caller maintains prev phi.
 */
struct RewardConfig {
    // Survival / death
    float survival_bonus;
    float death_penalty; // positive; caller subtracts on is_alive flip

    // Baseline (engine's attack value already includes line count, spin, b2b, combo multipliers)
    float lines_sent_weight;

    // Explicit segmented line-clear bonus (on top of lines_sent). All zero by default
    // to avoid double counting. Index 0 = no clear; unused.
    float clear_1;
    float clear_2;
    float clear_3;
    float clear_4;

    // Spin bonuses (applied only when the lock also cleared >=1 line)
    float tspin_bonus;
    float tspin_mini_bonus;

    // B2B / combo (applied only on successful clear)
    float b2b_bonus;   // flat, added once when b2b_count > 0
    float combo_bonus; // multiplied by combo_count

    // Perfect clear
    float perfect_clear_bonus;

    // Potential-based shaping (delta of phi per lock): lower stack / fewer holes = higher phi
    float height_weight; // phi term: -height_weight * sum_col_heights
    float hole_weight;   // phi term: -hole_weight * hole_count

    static RewardConfig fromGlobals();
};

// Base reward for a single lock event. Does NOT include potential delta.
// post_state : engine state right after the hard-drop (lines_cleared, spin_type,
//              lines_sent, combo, b2b, perfect_clear are all valid).
// just_died  : true if this lock caused is_alive to flip to false.
float computeLockBaseReward(const engine::State& post_state, bool just_died, const RewardConfig& cfg);

// Potential function phi(state). Returns 0 if both weights are zero (fast path).
// phi = -(height_weight * total_height + hole_weight * hole_count)
//       total_height = sum over columns of column stack height
//       hole_count   = number of empty cells below the top-most filled cell, per column
float computeBoardPotential(const engine::State& state, const RewardConfig& cfg);

} // namespace minizero::env::moderntetris::reward
