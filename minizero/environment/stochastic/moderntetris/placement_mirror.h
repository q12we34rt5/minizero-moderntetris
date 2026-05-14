#pragma once

#include "engine/tetris.hpp"
#include <cstdint>

namespace minizero::env::moderntetris_placement::mirror {

namespace engine = minizero::env::moderntetris::engine;

// Chiral pair under horizontal mirror in the 16-wide engine board (axis at
// col 7.5). Z(0)<->S(3), L(1)<->J(5), O/I/T self.
constexpr std::int8_t kPieceChiral[7] = {3, 5, 2, 0, 4, 1, 6};

inline engine::PieceType mirrorPieceType(engine::PieceType p)
{
    if (p == engine::PieceType::NONE) { return engine::PieceType::NONE; }
    const int idx = static_cast<int>(p);
    if (idx < 0 || idx >= 7) { return engine::PieceType::NONE; }
    return static_cast<engine::PieceType>(kPieceChiral[idx]);
}

// Per-(piece, orientation) analytic mirror. Each entry says: when the
// original placement is (piece, orient) with engine_lock_x = X, the mirror
// placement is (chiral_piece, mirror_orient) with engine_lock_x = x_const - X.
//
// All pieces follow the same orient mapping: 0->0, 1->3, 2->2, 3->1.
// Horizontal mirror swaps CW with CCW rotation, so a piece that ended up at
// rot 1 (one CW from spawn) on the original board has its mirror end at
// rot 3 (one CCW from spawn) on the mirror board. For chiral pairs whose
// rot-1 and rot-3 cell sets differ visibly within the 4x4 (L/J/T), this
// orient-swap is forced by cell match. For Z/S/I/O whose rot-1 and rot-3
// cells are 180-equivalent (same cells, lock_x differs by 1), cell-only
// verification can't distinguish the two choices; we still pick the swap
// because (a) it matches the natural CW<->CCW semantics, (b) it gives the
// (orient, lock_x) representation the engine's BFS is more likely to emit
// for the chiral piece on the mirror board (avoiding train/inference
// distribution shift on the augmented samples).
//
// x_const is NOT a single global constant: the engine packs each piece's
// mask left-aligned in a 4x4, so the bbox width determines the offset.
// x_const = (BOARD_WIDTH - 1) - max_dx_orig - min_dx_chiral, which works
// out to 13 for the 3-wide pieces (Z/S/L/J/T) and 12 for the centered
// pieces (O/I). See runPlacementMirrorVerify in mode_handler.cpp for the
// empirical per-descriptor cell-equality check.
struct MirrorOrientEntry {
    std::uint8_t mirror_orient;
    std::int8_t x_const;
};

constexpr MirrorOrientEntry kMirrorTable[7][4] = {
    {{0, 13}, {3, 13}, {2, 13}, {1, 13}}, // Z (0) -> S (3)
    {{0, 13}, {3, 13}, {2, 13}, {1, 13}}, // L (1) -> J (5)
    {{0, 12}, {3, 12}, {2, 12}, {1, 12}}, // O (2) -> O
    {{0, 13}, {3, 13}, {2, 13}, {1, 13}}, // S (3) -> Z (0)
    {{0, 12}, {3, 12}, {2, 12}, {1, 12}}, // I (4) -> I
    {{0, 13}, {3, 13}, {2, 13}, {1, 13}}, // J (5) -> L (1)
    {{0, 13}, {3, 13}, {2, 13}, {1, 13}}, // T (6) -> T
};

struct MirrorPlacementResult {
    int piece_type;  // 0..6
    int orientation; // 0..3
    int lock_x;      // descriptor coords (engine_lock_x - BOARD_LEFT)
};

// Mirror a placement descriptor's spatial fields. lock_x is in descriptor
// coords (engine_lock_x - BOARD_LEFT); the result is in the same coords.
// lock_y, spin_type, lines_cleared, use_hold are unaffected by horizontal
// mirror and should be copied verbatim by the caller.
inline MirrorPlacementResult mirrorPlacement(int piece_type, int orientation, int lock_x)
{
    const auto& e = kMirrorTable[piece_type][orientation];
    const int engine_x = lock_x + engine::BOARD_LEFT;
    const int mirror_engine_x = e.x_const - engine_x;
    return MirrorPlacementResult{
        static_cast<int>(kPieceChiral[piece_type]),
        static_cast<int>(e.mirror_orient),
        mirror_engine_x - engine::BOARD_LEFT,
    };
}

// Mirror one engine row (16 cells, 2 bits each). Cell at column c moves to
// column (15 - c). Walls in cols 0..2 swap with walls in cols 13..15 so the
// mirrored row keeps walls in the wall region.
inline engine::Row mirrorRow(engine::Row r)
{
    engine::Row out = 0;
    for (int c = 0; c < 16; ++c) {
        engine::ops::setCell(out, 15 - c, engine::ops::getCell(r, c));
    }
    return out;
}

// Mirror a whole engine State: flip every board row and swap current / hold /
// preview pieces to their chiral partners. The active piece's (x, y,
// orientation) are deliberately left UNTOUCHED -- the placement env only runs
// findPlacements between placements, when the piece is freshly spawned at
// (PIECE_SPAWN_X, PIECE_SPAWN_Y, 0), and the mirror game spawns at that same
// off-axis PIECE_SPAWN_X. Mirroring the spawn would hide the very asymmetry
// the differential verifier is meant to measure. Used only by the verifier.
inline engine::State mirrorState(const engine::State& s)
{
    engine::State out = s;
    for (int r = 0; r < engine::BOARD_HEIGHT; ++r) {
        out.board.data[r] = mirrorRow(s.board.data[r]);
    }
    out.current = mirrorPieceType(s.current);
    out.hold = mirrorPieceType(s.hold);
    for (std::size_t i = 0; i < sizeof(s.next) / sizeof(s.next[0]); ++i) {
        out.next[i] = mirrorPieceType(s.next[i]);
    }
    return out;
}

} // namespace minizero::env::moderntetris_placement::mirror
