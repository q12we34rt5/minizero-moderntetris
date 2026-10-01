#pragma once

#include "engine/tetris.hpp"

namespace minizero::env::moderntetris::reward {

/**
 * Per-lock reward shaping configuration. Caller reads the config globals via
 * RewardConfig::fromGlobals().
 *
 * The attack reward is a weighted decomposition of the engine's raw attack, read
 * from the post-lock state plus the locked piece, one value per attack component.
 * With every attack_* knob at its default (the engine's Jstris value) and the
 * combo weight at 1, the attack reward equals the engine's raw attack
 * (state->attack) exactly -- this is the faithful baseline. Zero out components
 * to run ablations (e.g. "only combo": set everything but attack_combo_weight to 0).
 *
 * NOTE: this is the raw, pre-garbage-counter attack -- it does NOT use
 * post.lines_sent. The two differ only when pending garbage is countered; raw
 * attack is the per-action offensive value and is what the components decompose.
 *
 * Structure of a reward per lock event:
 *
 *   base    = survival_bonus - (just_died ? death_penalty : 0)
 *   attack  = depth_weight * weighted_decomposition(post, locked_piece)
 *   pot     = phi(post) - phi(pre)     where phi is potential-based shaping
 *   total   = base + attack + pot
 *
 * base + attack is computed by computeLockBaseReward().
 * phi is computed by computeBoardPotential(). Caller maintains prev phi.
 */
struct RewardConfig {
    // Survival / death
    float survival_bonus;
    float death_penalty; // positive; caller subtracts on is_alive flip

    // Attack components. Each is the attack value contributed when that event
    // fires; defaults equal the engine's Jstris table so all-defaults reproduce
    // state->attack. A perfect clear overrides the line/spin value with
    // attack_pc. attack_allspin is an extra added on top of the normal-clear
    // value for non-T spins (the engine itself scores them as normal clears).
    float attack_single;            // 1-line clear (engine 0)
    float attack_double;            // 2-line clear (engine 1)
    float attack_triple;            // 3-line clear (engine 2)
    float attack_tetris;            // 4-line clear (engine 4)
    float attack_tspin_single;      // full T-spin, 1 line (engine 2)
    float attack_tspin_double;      // full T-spin, 2 lines (engine 4)
    float attack_tspin_triple;      // full T-spin, 3 lines (engine 6)
    float attack_tspin_mini_single; // mini T-spin, 1 line (engine 0)
    float attack_tspin_mini_double; // mini T-spin, 2 lines (engine 4, jstris: scored as a full T-spin double)
    float attack_allspin;           // extra for non-T spin clears (engine 0)
    float attack_pc;                // perfect clear, overrides base (engine 10)
    float attack_b2b;               // qualifying back-to-back +1 (engine 1)
    float attack_combo_weight;      // multiplies the engine combo-table value
    float b2b_break_penalty;        // subtracted when a clear breaks a running back-to-back chain
    // Height-keyed combo weight: attack_combo_weight at a pre-lock stack of
    // combo_height_low or less, attack_combo_weight_high at combo_height_high or
    // more, linear in between. Negative attack_combo_weight_high disables it.
    float attack_combo_weight_high;
    int combo_height_low;
    int combo_height_high;

    // Depth-keyed multiplier applied to the whole attack bucket. Indexed by the
    // piece's y-coordinate just before the lock (for the placement env:
    // PlacementSearchResult::lock_y). Weight = _top at y = BOARD_TOP, = _bottom
    // at y = BOARD_BOTTOM, linearly interpolated between. Out-of-range y is
    // clamped; locked_y < 0 disables the weight (defaults to 1.0).
    // Setting _top < _bottom makes the agent prefer clearing deep.
    //     weight(y) = lerp(_top, _bottom, clamp((y - BOARD_TOP)/span, 0, 1))
    float clear_depth_weight_bottom;
    float clear_depth_weight_top;

    // Potential-based shaping (delta of phi per lock): lower stack / fewer holes = higher phi
    float height_weight; // phi term: -height_weight * max_col_height
    float hole_weight;   // phi term: -hole_weight * hole_count

    static RewardConfig fromGlobals();
};

// Base reward for a single lock event. Does NOT include potential delta.
// post_state  : engine state right after the hard-drop (lines_cleared, spin_type,
//               lines_sent, combo, b2b, perfect_clear are all valid).
// locked_piece : the piece that just locked (captured from state.current BEFORE
//                hard-drop advances it to the next piece). Required to tell a
//                T-spin apart from a non-T all-spin, since the engine reuses
//                SpinType::SPIN_MINI for all non-T all-spins.
// locked_y    : piece's y-coordinate just before the lock, used to index the
//               depth-keyed low_clear bonus. Pass <0 to skip that term.
// just_died   : true if this lock caused is_alive to flip to false.
float computeLockBaseReward(const engine::State& post_state, engine::PieceType locked_piece, int locked_y, bool just_died, const RewardConfig& cfg);

// Penalty for a clear that breaks a running back-to-back chain: pre_b2b is the
// engine's back_to_back_count before the lock (>= 0 while a chain is running,
// -1 otherwise). Clears made with no chain running cost nothing, so clearing
// lines to survive stays free. Returns the (non-negative) amount to subtract.
float computeB2bBreakPenalty(int pre_b2b, const engine::State& post_state, const RewardConfig& cfg);

// Max column height of the stack.
int maxColumnHeight(const engine::State& state);

// Extra combo reward from the height-keyed combo weight: (weight(pre_height) -
// attack_combo_weight) * combo value, so the total combo term follows the
// height-keyed weight. Not scaled by the clear-depth multiplier. 0 when disabled.
float computeComboHeightBonus(int pre_height, const engine::State& post_state, const RewardConfig& cfg);

// Potential function phi(state). Returns 0 if both weights are zero (fast path).
// phi = -(height_weight * max_height + hole_weight * hole_count)
//       max_height = max over columns of column stack height
//       hole_count = number of empty cells below the top-most filled cell, per column
float computeBoardPotential(const engine::State& state, const RewardConfig& cfg);

} // namespace minizero::env::moderntetris::reward
