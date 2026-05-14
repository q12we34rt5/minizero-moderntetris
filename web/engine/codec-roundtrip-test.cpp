// Standalone round-trip test for the shared engine state codec.
//
// state_codec.hpp is the single serialization format used by BOTH the WASM
// frontend (et_serialize_full) and the minizero console (set_state). This test
// verifies serialize -> deserialize is identity. It only needs the engine
// headers (no libtorch), so it builds with a plain compiler. Run via:
//
//   npm run test:codec      (from web/)

#include "state_codec.hpp"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace codec = minizero::env::moderntetris::engine::codec;
namespace eng = minizero::env::moderntetris::engine;
namespace step = minizero::env::moderntetris::engine::step;

static int failures = 0;
static void check(const char* name, bool ok)
{
    std::printf("  %s %s\n", ok ? "ok  " : "FAIL", name);
    if (!ok) { ++failures; }
}

int main()
{
    // Build a context with distinctive, non-trivial values in every field so a
    // dropped or misordered field would show up as a mismatch.
    step::Context ctx{};
    eng::State& s = ctx.state;
    for (int i = 0; i < eng::BOARD_HEIGHT; ++i) {
        s.board.data[i] = static_cast<eng::Row>(0x9E3779B9u * (i + 1)); // includes high-bit values
    }
    s.is_alive = 1;
    for (int i = 0; i < 14; ++i) { s.next[i] = static_cast<eng::PieceType>(i % 7); }
    s.hold = eng::PieceType::T;
    s.has_held = 1;
    s.current = eng::PieceType::I;
    s.orientation = 2;
    s.x = 7;
    s.y = -3;
    s.seed = 0xDEADBEEFu;
    s.srs_index = 4;
    s.piece_count = 123456;
    s.was_last_rotation = 1;
    s.spin_type = eng::SpinType::SPIN_MINI;
    s.perfect_clear = 1;
    s.back_to_back_count = -1;
    s.combo_count = 9;
    s.lines_cleared = 4;
    s.attack = 12;
    s.lines_sent = 5;
    s.total_lines_cleared = 999;
    s.total_attack = 4242;
    s.total_lines_sent = 314;
    s.garbage_seed = 0x12345678u;
    for (int i = 0; i < eng::GARBAGE_QUEUE_SIZE; ++i) {
        s.garbage_queue[i] = static_cast<std::uint8_t>(i * 3 + 1);
        s.garbage_delay[i] = static_cast<std::uint8_t>(i * 2);
    }
    s.max_garbage_spawn = 6;
    s.garbage_blocking = 1;
    ctx.lifetime = 20;
    ctx.config.piece_life = 0x7fffffff;
    ctx.config.auto_drop = 0;

    std::vector<std::int32_t> buf(codec::STATE_CODEC_SIZE);
    codec::serialize(ctx, buf.data());

    step::Context out{};
    codec::deserialize(buf.data(), out);

    check("board rows round-trip (incl. high-bit values)",
          std::memcmp(s.board.data, out.state.board.data, sizeof(s.board.data)) == 0);
    check("next queue round-trips",
          std::memcmp(s.next, out.state.next, sizeof(s.next)) == 0);
    check("garbage_queue round-trips",
          std::memcmp(s.garbage_queue, out.state.garbage_queue, sizeof(s.garbage_queue)) == 0);
    check("garbage_delay round-trips",
          std::memcmp(s.garbage_delay, out.state.garbage_delay, sizeof(s.garbage_delay)) == 0);
    check("current/hold/orientation",
          out.state.current == s.current && out.state.hold == s.hold &&
              out.state.orientation == s.orientation && out.state.has_held == s.has_held);
    check("x/y (signed) round-trip", out.state.x == s.x && out.state.y == s.y);
    check("seeds round-trip", out.state.seed == s.seed && out.state.garbage_seed == s.garbage_seed);
    check("srs_index (signed) + spin_type",
          out.state.srs_index == s.srs_index && out.state.spin_type == s.spin_type);
    check("back_to_back_count (negative) round-trips",
          out.state.back_to_back_count == s.back_to_back_count);
    check("combo/lines/attack counters",
          out.state.combo_count == s.combo_count && out.state.lines_cleared == s.lines_cleared &&
              out.state.attack == s.attack && out.state.lines_sent == s.lines_sent);
    check("totals round-trip",
          out.state.total_lines_cleared == s.total_lines_cleared &&
              out.state.total_attack == s.total_attack &&
              out.state.total_lines_sent == s.total_lines_sent);
    check("piece_count / is_alive / perfect_clear / was_last_rotation",
          out.state.piece_count == s.piece_count && out.state.is_alive == s.is_alive &&
              out.state.perfect_clear == s.perfect_clear &&
              out.state.was_last_rotation == s.was_last_rotation);
    check("max_garbage_spawn / garbage_blocking",
          out.state.max_garbage_spawn == s.max_garbage_spawn &&
              out.state.garbage_blocking == s.garbage_blocking);
    check("ctx lifetime + config",
          out.lifetime == ctx.lifetime && out.config.piece_life == ctx.config.piece_life &&
              out.config.auto_drop == ctx.config.auto_drop);

    // STATE_CODEC_SIZE must exactly match what visit() walks: a counting cursor.
    int counted = 0;
    codec::visit(ctx, [&counted](const auto&) { ++counted; });
    check("STATE_CODEC_SIZE matches the visited field count", counted == codec::STATE_CODEC_SIZE);

    std::printf(failures == 0 ? "\nALL PASSED\n" : "\n%d FAILED\n", failures);
    return failures == 0 ? 0 : 1;
}
