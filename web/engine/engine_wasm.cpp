// WASM wrapper around the moderntetris engine (step-level).
//
// Compiled by build.sh together with the in-tree engine source
// (minizero/environment/stochastic/moderntetris/engine/tetris.cpp).
// Phase 1 exposes the step-level API needed for local human play.
// Placement-level APIs (findPlacements / applyPlacement) are added in
// Phase 2 when the AI backend lands.

#include "../../minizero/environment/stochastic/moderntetris/engine/step.hpp"
#include <cstdint>
#include <emscripten/emscripten.h>

namespace eng = minizero::env::moderntetris::engine;
namespace step = minizero::env::moderntetris::engine::step;

// ---- Serialized view layout (flat int32 array, read by JS via Int32Array) ----
// Keep in sync with web/src/engine/view.ts
enum : int {
    ET_BOARD_W = eng::BOARD_RIGHT - eng::BOARD_LEFT + 1, // 10
    ET_BOARD_H = eng::BOARD_BOTTOM - eng::BOARD_TOP + 1, // 20
    ET_BOARD_CELLS = ET_BOARD_W * ET_BOARD_H,            // 200
    ET_NEXT_COUNT = 5,
};
enum : int {
    I_BOARD = 0,                         // [0..199]
    I_CURRENT = ET_BOARD_CELLS,          // 200
    I_ORIENTATION,                       // 201
    I_X,                                 // 202
    I_Y,                                 // 203
    I_GHOST_Y,                           // 204
    I_HOLD,                              // 205
    I_HAS_HELD,                          // 206
    I_NEXT,                              // 207..211
    I_IS_ALIVE = I_NEXT + ET_NEXT_COUNT, // 212
    I_PIECE_COUNT,                       // 213
    I_COMBO_COUNT,                       // 214
    I_B2B_COUNT,                         // 215
    I_LINES_CLEARED,                     // 216
    I_ATTACK,                            // 217
    I_LINES_SENT,                        // 218
    I_TOTAL_LINES_CLEARED,               // 219
    I_TOTAL_ATTACK,                      // 220
    I_TOTAL_LINES_SENT,                  // 221
    I_SPIN_TYPE,                         // 222
    I_SRS_INDEX,                         // 223
    I_PERFECT_CLEAR,                     // 224
    I_PENDING_GARBAGE,                   // 225
    I_LIFETIME,                          // 226
    ET_VIEW_SIZE,                        // 227
};

static int computeGhostY(const eng::State& s)
{
    if (s.current == eng::PieceType::NONE) { return s.y; }
    const eng::Piece& piece = eng::ops::getPiece(s.current, s.orientation);
    int gy = s.y;
    while (eng::ops::canPlacePiece(s.board, piece, s.x, gy + 1)) { gy++; }
    return gy;
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
}

} // extern "C"
