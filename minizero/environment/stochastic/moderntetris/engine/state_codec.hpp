#pragma once
#include "step.hpp"
#include <cstdint>

// Flat int32 serialization of a full step::Context (engine State + lifetime +
// config). Used to ship a board between the WASM frontend and the minizero
// backend without replay drift.
//
// serialize() and deserialize() both walk the SAME field list in visit(), so
// the two directions cannot desync. The field order in visit() is the single
// source of truth. WriteCursor / ReadCursor / visit are codec internals --
// callers use serialize() / deserialize() (visit() is also handy for tests).

namespace minizero::env::moderntetris::engine::codec {

// 32 board rows + 14 next + 20 garbage_queue + 20 garbage_delay + 28 scalars.
constexpr int STATE_CODEC_SIZE = BOARD_HEIGHT + 14 + 2 * GARBAGE_QUEUE_SIZE + 28;

struct WriteCursor {
    std::int32_t* p;
    template <typename T>
    void operator()(const T& v) { *p++ = static_cast<std::int32_t>(v); }
};

struct ReadCursor {
    const std::int32_t* p;
    template <typename T>
    void operator()(T& v) { v = static_cast<T>(*p++); }
};

// Visits every serialized field of ctx in a fixed order. The cursor either
// writes (WriteCursor) or reads (ReadCursor) each field it is handed.
template <typename Ctx, typename Cursor>
void visit(Ctx& ctx, Cursor&& cur)
{
    auto& s = ctx.state;
    for (int i = 0; i < BOARD_HEIGHT; ++i) { cur(s.board.data[i]); }
    cur(s.is_alive);
    for (int i = 0; i < 14; ++i) { cur(s.next[i]); }
    cur(s.hold);
    cur(s.has_held);
    cur(s.current);
    cur(s.orientation);
    cur(s.x);
    cur(s.y);
    cur(s.seed);
    cur(s.srs_index);
    cur(s.piece_count);
    cur(s.was_last_rotation);
    cur(s.spin_type);
    cur(s.perfect_clear);
    cur(s.back_to_back_count);
    cur(s.combo_count);
    cur(s.lines_cleared);
    cur(s.attack);
    cur(s.lines_sent);
    cur(s.total_lines_cleared);
    cur(s.total_attack);
    cur(s.total_lines_sent);
    cur(s.garbage_seed);
    for (int i = 0; i < GARBAGE_QUEUE_SIZE; ++i) { cur(s.garbage_queue[i]); }
    for (int i = 0; i < GARBAGE_QUEUE_SIZE; ++i) { cur(s.garbage_delay[i]); }
    cur(s.max_garbage_spawn);
    cur(s.garbage_blocking);
    cur(s.all_spin);
    cur(ctx.lifetime);
    cur(ctx.config.piece_life);
    cur(ctx.config.auto_drop);
}

// Writes STATE_CODEC_SIZE int32 values into out.
inline void serialize(const step::Context& ctx, std::int32_t* out)
{
    visit(ctx, WriteCursor{out});
}

// Reads STATE_CODEC_SIZE int32 values from in into ctx.
inline void deserialize(const std::int32_t* in, step::Context& ctx)
{
    visit(ctx, ReadCursor{in});
}

} // namespace minizero::env::moderntetris::engine::codec
