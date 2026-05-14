// WASM wrapper around the moderntetris engine.
//
// Compiled by build.sh together with the in-tree engine source (tetris.cpp +
// placement_search.cpp). Exposes the step-level API for local human play plus
// placement-level APIs (apply_placement) and full-state serialization
// (serialize_full / codec) for the Phase 2 AI backend.

#include "../../minizero/environment/stochastic/moderntetris/engine/placement_search.hpp"
#include "../../minizero/environment/stochastic/moderntetris/engine/state_codec.hpp"
#include "../../minizero/environment/stochastic/moderntetris/engine/step.hpp"
#include <cstdint>
#include <emscripten/emscripten.h>

namespace eng = minizero::env::moderntetris::engine;
namespace step = minizero::env::moderntetris::engine::step;
namespace codec = minizero::env::moderntetris::engine::codec;

// ---- Serialized view layout (flat int32 array, read by JS via Int32Array) ----
// Keep in sync with web/src/engine/view.ts
enum : int {
    ET_BOARD_W = eng::BOARD_RIGHT - eng::BOARD_LEFT + 1, // 10
    ET_BOARD_H = eng::BOARD_BOTTOM - eng::BOARD_TOP + 1, // 20
    ET_BOARD_CELLS = ET_BOARD_W * ET_BOARD_H,            // 200
    ET_NEXT_COUNT = 5,
    ET_GARBAGE_SLOTS = eng::GARBAGE_QUEUE_SIZE, // 20
};
enum : int {
    I_BOARD = 0,                                          // [0..199]
    I_CURRENT = ET_BOARD_CELLS,                           // 200
    I_ORIENTATION,                                        // 201
    I_X,                                                  // 202
    I_Y,                                                  // 203
    I_GHOST_Y,                                            // 204
    I_HOLD,                                               // 205
    I_HAS_HELD,                                           // 206
    I_NEXT,                                               // 207..211
    I_IS_ALIVE = I_NEXT + ET_NEXT_COUNT,                  // 212
    I_PIECE_COUNT,                                        // 213
    I_COMBO_COUNT,                                        // 214
    I_B2B_COUNT,                                          // 215
    I_LINES_CLEARED,                                      // 216
    I_ATTACK,                                             // 217
    I_LINES_SENT,                                         // 218
    I_TOTAL_LINES_CLEARED,                                // 219
    I_TOTAL_ATTACK,                                       // 220
    I_TOTAL_LINES_SENT,                                   // 221
    I_SPIN_TYPE,                                          // 222
    I_SRS_INDEX,                                          // 223
    I_PERFECT_CLEAR,                                      // 224
    I_PENDING_GARBAGE,                                    // 225
    I_LIFETIME,                                           // 226
    I_GARBAGE_QUEUE,                                      // 227..246  (per-entry queued lines)
    I_GARBAGE_DELAY = I_GARBAGE_QUEUE + ET_GARBAGE_SLOTS, // 247..266  (per-entry delay)
    ET_VIEW_SIZE = I_GARBAGE_DELAY + ET_GARBAGE_SLOTS,    // 267
};

static int computeGhostY(const eng::State& s)
{
    if (s.current == eng::PieceType::NONE) { return s.y; }
    const eng::Piece& piece = eng::ops::getPiece(s.current, s.orientation);
    int gy = s.y;
    while (eng::ops::canPlacePiece(s.board, piece, s.x, gy + 1)) { gy++; }
    return gy;
}

static step::Action placementActionToStepAction(eng::PlacementAction pa)
{
    switch (pa) {
        case eng::PlacementAction::LEFT: return step::Action::MOVE_LEFT;
        case eng::PlacementAction::RIGHT: return step::Action::MOVE_RIGHT;
        case eng::PlacementAction::LEFT_WALL: return step::Action::MOVE_LEFT_TO_WALL;
        case eng::PlacementAction::RIGHT_WALL: return step::Action::MOVE_RIGHT_TO_WALL;
        case eng::PlacementAction::SOFT_DROP: return step::Action::SOFT_DROP;
        case eng::PlacementAction::SOFT_DROP_FLOOR: return step::Action::SOFT_DROP_TO_FLOOR;
        case eng::PlacementAction::ROTATE_CW: return step::Action::ROTATE_CW;
        case eng::PlacementAction::ROTATE_CCW: return step::Action::ROTATE_CCW;
        case eng::PlacementAction::ROTATE_180: return step::Action::ROTATE_180;
        default: return step::Action::NOOP;
    }
}

extern "C" {

EMSCRIPTEN_KEEPALIVE
int et_view_size() { return ET_VIEW_SIZE; }

EMSCRIPTEN_KEEPALIVE
step::Context* et_create() { return new step::Context(); }

EMSCRIPTEN_KEEPALIVE
void et_free(step::Context* ctx) { delete ctx; }

// piece_life <= 0 disables forced hard drop (treated as effectively infinite).
EMSCRIPTEN_KEEPALIVE
void et_set_config(step::Context* ctx, int piece_life, int auto_drop)
{
    step::Config cfg;
    cfg.piece_life = (piece_life > 0) ? piece_life : 0x7fffffff;
    cfg.auto_drop = auto_drop ? 1 : 0;
    step::setConfig(ctx, cfg);
}

EMSCRIPTEN_KEEPALIVE
void et_reset(step::Context* ctx, unsigned int seed)
{
    step::setSeed(ctx, seed, seed ^ 0x9e3779b9u);
    step::reset(ctx);
}

// Returns packed info: bit0 = action_success, bit1 = forced_hard_drop.
EMSCRIPTEN_KEEPALIVE
int et_step(step::Context* ctx, int action)
{
    if (action < 0 || action >= static_cast<int>(step::Action::SIZE)) { return 0; }
    step::Info info = step::step(ctx, static_cast<step::Action>(action));
    return (info.action_success ? 1 : 0) | (info.forced_hard_drop ? 2 : 0);
}

// Queue garbage lines (used in later phases for PvE/EvE).
EMSCRIPTEN_KEEPALIVE
int et_add_garbage(step::Context* ctx, int lines, int delay)
{
    if (lines <= 0) { return 0; }
    return eng::addGarbage(&ctx->state, static_cast<std::uint8_t>(lines > 255 ? 255 : lines),
                           static_cast<std::uint8_t>(delay > 255 ? 255 : (delay < 0 ? 0 : delay)))
               ? 1
               : 0;
}

EMSCRIPTEN_KEEPALIVE
void et_serialize(step::Context* ctx, int* out)
{
    const eng::State& s = ctx->state;
    // board (active piece NOT placed; JS renders it from piece fields)
    for (int y = 0; y < ET_BOARD_H; ++y) {
        for (int x = 0; x < ET_BOARD_W; ++x) {
            eng::Cell cell = eng::ops::getCell(s.board, x + eng::BOARD_LEFT, y + eng::BOARD_TOP);
            out[I_BOARD + y * ET_BOARD_W + x] = static_cast<int>(static_cast<eng::Row>(cell) >> 30);
        }
    }
    out[I_CURRENT] = static_cast<int>(s.current);
    out[I_ORIENTATION] = static_cast<int>(s.orientation);
    out[I_X] = static_cast<int>(s.x) - eng::BOARD_LEFT;
    out[I_Y] = static_cast<int>(s.y) - eng::BOARD_TOP;
    out[I_GHOST_Y] = computeGhostY(s) - eng::BOARD_TOP;
    out[I_HOLD] = static_cast<int>(s.hold);
    out[I_HAS_HELD] = s.has_held ? 1 : 0;
    for (int i = 0; i < ET_NEXT_COUNT; ++i) { out[I_NEXT + i] = static_cast<int>(s.next[i]); }
    out[I_IS_ALIVE] = s.is_alive ? 1 : 0;
    out[I_PIECE_COUNT] = static_cast<int>(s.piece_count);
    out[I_COMBO_COUNT] = s.combo_count;
    out[I_B2B_COUNT] = s.back_to_back_count;
    out[I_LINES_CLEARED] = s.lines_cleared;
    out[I_ATTACK] = s.attack;
    out[I_LINES_SENT] = s.lines_sent;
    out[I_TOTAL_LINES_CLEARED] = static_cast<int>(s.total_lines_cleared);
    out[I_TOTAL_ATTACK] = static_cast<int>(s.total_attack);
    out[I_TOTAL_LINES_SENT] = static_cast<int>(s.total_lines_sent);
    out[I_SPIN_TYPE] = static_cast<int>(s.spin_type);
    out[I_SRS_INDEX] = static_cast<int>(s.srs_index);
    out[I_PERFECT_CLEAR] = s.perfect_clear ? 1 : 0;
    int pending = 0;
    for (int i = 0; i < eng::GARBAGE_QUEUE_SIZE; ++i) { pending += s.garbage_queue[i]; }
    out[I_PENDING_GARBAGE] = pending;
    out[I_LIFETIME] = ctx->lifetime;
    for (int i = 0; i < ET_GARBAGE_SLOTS; ++i) {
        out[I_GARBAGE_QUEUE + i] = s.garbage_queue[i];
        out[I_GARBAGE_DELAY + i] = s.garbage_delay[i];
    }
}

// ---- Placement-level API (Phase 2: AI backend) ----

// Number of int32 slots a full-state serialization occupies.
EMSCRIPTEN_KEEPALIVE
int et_codec_size() { return codec::STATE_CODEC_SIZE; }

// Serialize the complete engine context (board + pieces + garbage + config)
// for the AI backend. out must hold et_codec_size() int32 values.
EMSCRIPTEN_KEEPALIVE
void et_serialize_full(step::Context* ctx, int* out)
{
    codec::serialize(*ctx, out);
}

// Enumerate the legal placements for the current piece (no-hold branch only).
// Writes up to max_count entries of 4 ints each: lock_x, lock_y, orientation,
// spin_type (engine coordinates). Returns the number of placements written.
EMSCRIPTEN_KEEPALIVE
int et_find_placements(step::Context* ctx, int* out, int max_count)
{
    const auto placements = eng::findPlacements(ctx->state);
    int n = 0;
    for (const auto& p : placements) {
        if (n >= max_count) { break; }
        out[n * 4 + 0] = p.lock_x;
        out[n * 4 + 1] = p.lock_y;
        out[n * 4 + 2] = p.orientation;
        out[n * 4 + 3] = static_cast<int>(p.spin_type);
        ++n;
    }
    return n;
}

// Apply a placement-level move: optionally hold, then replay the BFS path to
// the locked position and hard-drop. The (lock_x, lock_y) are engine board
// coordinates, matching the console string emitted by minizero's genmove.
// Returns 1 on success, 0 if no matching placement was found.
EMSCRIPTEN_KEEPALIVE
int et_apply_placement(step::Context* ctx, int use_hold, int lock_x, int lock_y, int orientation, int spin_type)
{
    if (use_hold) {
        if (!step::step(ctx, step::Action::HOLD).action_success) { return 0; }
    }
    const auto placements = eng::findPlacements(ctx->state);
    for (const auto& p : placements) {
        if (p.lock_x == lock_x && p.lock_y == lock_y &&
            p.orientation == orientation && static_cast<int>(p.spin_type) == spin_type) {
            for (const auto pa : p.path) { step::step(ctx, placementActionToStepAction(pa)); }
            step::step(ctx, step::Action::HARD_DROP);
            return 1;
        }
    }
    return 0;
}

} // extern "C"
