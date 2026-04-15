#include "tetris.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>

#include <cassert>

namespace minizero::env::moderntetris::engine {

template <template <int...> class Container, int n, int... args>
struct IndexGenerator : IndexGenerator<Container, n - 1, n - 1, args...> {
};
template <template <int...> class Container, int... args>
struct IndexGenerator<Container, 0, args...> {
    using result = Container<args...>;
};
template <int... args>
struct Wrapper {
    template <int height, int floor, Row row, Row wall>
    struct BoardInitializer {
        template <bool condition, typename x>
        struct If {
        };
        template <typename x>
        struct If<true, x> {
            static constexpr Row value = wall;
        };
        template <typename x>
        struct If<false, x> {
            static constexpr Row value = row;
        };

        static const Rows<height> board;
    };
};
template <int... args>
template <int height, int floor, Row row, Row wall>
const Rows<height> Wrapper<args...>::BoardInitializer<height, floor, row, wall>::board = {
    Wrapper<args...>::template BoardInitializer<height, floor, row, wall>::template If<(args >= height - floor), int>::value...};

inline static SRSKickData getSRSKickData(PieceType type, std::uint8_t orientation, Rotation rot)
{
    // [<piece-orientation>][<rotate-direction>][<srs-index>]
    static const SRSKickData::Kick JLSTZ[4][2][5] = {
        {
            {{0, 0}, {-1, 0}, {-1, 1}, {0, -2}, {-1, -2}}, // 0 -> 1
            {{0, 0}, {1, 0}, {1, 1}, {0, -2}, {1, -2}},    // 0 -> 3
        },
        {
            {{0, 0}, {1, 0}, {1, -1}, {0, 2}, {1, 2}}, // 1 -> 2
            {{0, 0}, {1, 0}, {1, -1}, {0, 2}, {1, 2}}, // 1 -> 0
        },
        {
            {{0, 0}, {1, 0}, {1, 1}, {0, -2}, {1, -2}},    // 2 -> 3
            {{0, 0}, {-1, 0}, {-1, 1}, {0, -2}, {-1, -2}}, // 2 -> 1
        },
        {
            {{0, 0}, {-1, 0}, {-1, -1}, {0, 2}, {-1, 2}}, // 3 -> 0
            {{0, 0}, {-1, 0}, {-1, -1}, {0, 2}, {-1, 2}}, // 3 -> 2
        }};
    static const SRSKickData::Kick JLSTZ180[4][1][6] = {
        {
            {{0, 0}, {0, 1}, {1, 1}, {-1, 1}, {1, 0}, {-1, 0}}, // 0 -> 2
        },
        {
            {{0, 0}, {1, 0}, {1, 2}, {1, 1}, {0, 2}, {0, 1}}, // 1 -> 3
        },
        {
            {{0, 0}, {0, -1}, {-1, -1}, {1, -1}, {-1, 0}, {1, 0}}, // 2 -> 0
        },
        {
            {{0, 0}, {-1, 0}, {-1, 2}, {-1, 1}, {0, 2}, {0, 1}}, // 3 -> 1
        }};
    static const SRSKickData::Kick I[4][2][5] = {
        {
            {{0, 0}, {-2, 0}, {1, 0}, {-2, -1}, {1, 2}}, // 0 -> 1
            {{0, 0}, {-1, 0}, {2, 0}, {-1, 2}, {2, -1}}, // 0 -> 3
        },
        {
            {{0, 0}, {-1, 0}, {2, 0}, {-1, 2}, {2, -1}}, // 1 -> 2
            {{0, 0}, {2, 0}, {-1, 0}, {2, 1}, {-1, -2}}, // 1 -> 0
        },
        {
            {{0, 0}, {2, 0}, {-1, 0}, {2, 1}, {-1, -2}}, // 2 -> 3
            {{0, 0}, {1, 0}, {-2, 0}, {1, -2}, {-2, 1}}, // 2 -> 1
        },
        {
            {{0, 0}, {1, 0}, {-2, 0}, {1, -2}, {-2, 1}}, // 3 -> 0
            {{0, 0}, {-2, 0}, {1, 0}, {-2, -1}, {1, 2}}, // 3 -> 2
        }};
    static const SRSKickData::Kick O[4][2][1] = {
        {
            {{0, 0}}, // 0 -> 1
            {{0, 0}}  // 0 -> 3
        },
        {
            {{0, 0}}, // 1 -> 2
            {{0, 0}}  // 1 -> 0
        },
        {
            {{0, 0}}, // 2 -> 3
            {{0, 0}}  // 2 -> 1
        },
        {
            {{0, 0}}, // 3 -> 0
            {{0, 0}}  // 3 -> 2
        }};
    static const SRSKickData::Kick IO180[4][1][1] = {
        {
            {{0, 0}}, // 0 -> 2
        },
        {
            {{0, 0}}, // 1 -> 3
        },
        {
            {{0, 0}}, // 2 -> 0
        },
        {
            {{0, 0}}, // 3 -> 1
        }};
    switch (type) {
        case PieceType::J:
        case PieceType::L:
        case PieceType::S:
        case PieceType::T:
        case PieceType::Z:
            if (rot == Rotation::CW || rot == Rotation::CCW) {
                return {
                    .kicks = JLSTZ[orientation][static_cast<std::underlying_type_t<Rotation>>(rot)],
                    .length = 5};
            } else if (rot == Rotation::HALF) {
                return {
                    .kicks = JLSTZ180[orientation][0],
                    .length = 6};
            }
            break;
        case PieceType::I:
            if (rot == Rotation::CW || rot == Rotation::CCW) {
                return {
                    .kicks = I[orientation][static_cast<std::underlying_type_t<Rotation>>(rot)],
                    .length = 5};
            } else if (rot == Rotation::HALF) {
                return {
                    .kicks = IO180[orientation][0],
                    .length = 1};
            }
            break;
        case PieceType::O:
            if (rot == Rotation::CW || rot == Rotation::CCW) {
                return {
                    .kicks = O[orientation][static_cast<std::underlying_type_t<Rotation>>(rot)],
                    .length = 1};
            } else if (rot == Rotation::HALF) {
                return {
                    .kicks = IO180[orientation][0],
                    .length = 1};
            }
            break;
        default:
            break;
    }
    return {
        .kicks = nullptr,
        .length = 0};
}

// srs_table[<piece-type>][<piece-orientation>][<rotate-direction>].kicks[<srs-index>]
const SRSKickData srs_table[static_cast<std::underlying_type_t<PieceType>>(PieceType::SIZE)][4][static_cast<std::underlying_type_t<Rotation>>(Rotation::SIZE)] = {
    {{getSRSKickData(PieceType::Z, 0, Rotation::CW), getSRSKickData(PieceType::Z, 0, Rotation::CCW), getSRSKickData(PieceType::Z, 0, Rotation::HALF)},
     {getSRSKickData(PieceType::Z, 1, Rotation::CW), getSRSKickData(PieceType::Z, 1, Rotation::CCW), getSRSKickData(PieceType::Z, 1, Rotation::HALF)},
     {getSRSKickData(PieceType::Z, 2, Rotation::CW), getSRSKickData(PieceType::Z, 2, Rotation::CCW), getSRSKickData(PieceType::Z, 2, Rotation::HALF)},
     {getSRSKickData(PieceType::Z, 3, Rotation::CW), getSRSKickData(PieceType::Z, 3, Rotation::CCW), getSRSKickData(PieceType::Z, 3, Rotation::HALF)}},
    {{getSRSKickData(PieceType::L, 0, Rotation::CW), getSRSKickData(PieceType::L, 0, Rotation::CCW), getSRSKickData(PieceType::L, 0, Rotation::HALF)},
     {getSRSKickData(PieceType::L, 1, Rotation::CW), getSRSKickData(PieceType::L, 1, Rotation::CCW), getSRSKickData(PieceType::L, 1, Rotation::HALF)},
     {getSRSKickData(PieceType::L, 2, Rotation::CW), getSRSKickData(PieceType::L, 2, Rotation::CCW), getSRSKickData(PieceType::L, 2, Rotation::HALF)},
     {getSRSKickData(PieceType::L, 3, Rotation::CW), getSRSKickData(PieceType::L, 3, Rotation::CCW), getSRSKickData(PieceType::L, 3, Rotation::HALF)}},
    {{getSRSKickData(PieceType::O, 0, Rotation::CW), getSRSKickData(PieceType::O, 0, Rotation::CCW), getSRSKickData(PieceType::O, 0, Rotation::HALF)},
     {getSRSKickData(PieceType::O, 1, Rotation::CW), getSRSKickData(PieceType::O, 1, Rotation::CCW), getSRSKickData(PieceType::O, 1, Rotation::HALF)},
     {getSRSKickData(PieceType::O, 2, Rotation::CW), getSRSKickData(PieceType::O, 2, Rotation::CCW), getSRSKickData(PieceType::O, 2, Rotation::HALF)},
     {getSRSKickData(PieceType::O, 3, Rotation::CW), getSRSKickData(PieceType::O, 3, Rotation::CCW), getSRSKickData(PieceType::O, 3, Rotation::HALF)}},
    {{getSRSKickData(PieceType::S, 0, Rotation::CW), getSRSKickData(PieceType::S, 0, Rotation::CCW), getSRSKickData(PieceType::S, 0, Rotation::HALF)},
     {getSRSKickData(PieceType::S, 1, Rotation::CW), getSRSKickData(PieceType::S, 1, Rotation::CCW), getSRSKickData(PieceType::S, 1, Rotation::HALF)},
     {getSRSKickData(PieceType::S, 2, Rotation::CW), getSRSKickData(PieceType::S, 2, Rotation::CCW), getSRSKickData(PieceType::S, 2, Rotation::HALF)},
     {getSRSKickData(PieceType::S, 3, Rotation::CW), getSRSKickData(PieceType::S, 3, Rotation::CCW), getSRSKickData(PieceType::S, 3, Rotation::HALF)}},
    {{getSRSKickData(PieceType::I, 0, Rotation::CW), getSRSKickData(PieceType::I, 0, Rotation::CCW), getSRSKickData(PieceType::I, 0, Rotation::HALF)},
     {getSRSKickData(PieceType::I, 1, Rotation::CW), getSRSKickData(PieceType::I, 1, Rotation::CCW), getSRSKickData(PieceType::I, 1, Rotation::HALF)},
     {getSRSKickData(PieceType::I, 2, Rotation::CW), getSRSKickData(PieceType::I, 2, Rotation::CCW), getSRSKickData(PieceType::I, 2, Rotation::HALF)},
     {getSRSKickData(PieceType::I, 3, Rotation::CW), getSRSKickData(PieceType::I, 3, Rotation::CCW), getSRSKickData(PieceType::I, 3, Rotation::HALF)}},
    {{getSRSKickData(PieceType::J, 0, Rotation::CW), getSRSKickData(PieceType::J, 0, Rotation::CCW), getSRSKickData(PieceType::J, 0, Rotation::HALF)},
     {getSRSKickData(PieceType::J, 1, Rotation::CW), getSRSKickData(PieceType::J, 1, Rotation::CCW), getSRSKickData(PieceType::J, 1, Rotation::HALF)},
     {getSRSKickData(PieceType::J, 2, Rotation::CW), getSRSKickData(PieceType::J, 2, Rotation::CCW), getSRSKickData(PieceType::J, 2, Rotation::HALF)},
     {getSRSKickData(PieceType::J, 3, Rotation::CW), getSRSKickData(PieceType::J, 3, Rotation::CCW), getSRSKickData(PieceType::J, 3, Rotation::HALF)}},
    {{getSRSKickData(PieceType::T, 0, Rotation::CW), getSRSKickData(PieceType::T, 0, Rotation::CCW), getSRSKickData(PieceType::T, 0, Rotation::HALF)},
     {getSRSKickData(PieceType::T, 1, Rotation::CW), getSRSKickData(PieceType::T, 1, Rotation::CCW), getSRSKickData(PieceType::T, 1, Rotation::HALF)},
     {getSRSKickData(PieceType::T, 2, Rotation::CW), getSRSKickData(PieceType::T, 2, Rotation::CCW), getSRSKickData(PieceType::T, 2, Rotation::HALF)},
     {getSRSKickData(PieceType::T, 3, Rotation::CW), getSRSKickData(PieceType::T, 3, Rotation::CCW), getSRSKickData(PieceType::T, 3, Rotation::HALF)}},
};

inline static std::uint32_t xorshf32(std::uint32_t& seed)
{
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    return seed;
}

struct SevenBagPermutations {
    struct PackedSevenBag {
        std::uint32_t b0 : 3;
        std::uint32_t b1 : 3;
        std::uint32_t b2 : 3;
        std::uint32_t b3 : 3;
        std::uint32_t b4 : 3;
        std::uint32_t b5 : 3;
        std::uint32_t b6 : 3;
    };
    // 7! = 5040
    enum { SIZE = 5040 };
    PackedSevenBag data[SIZE];
};
inline static SevenBagPermutations generateSevenBagPermutations()
{
    SevenBagPermutations permutations;
    PieceType next[] = {PieceType::Z, PieceType::L, PieceType::O, PieceType::S, PieceType::I, PieceType::J, PieceType::T};
    int index = 0;
    do {
        auto& ref = permutations.data[index++];
        ref.b0 = static_cast<std::uint32_t>(next[0]);
        ref.b1 = static_cast<std::uint32_t>(next[1]);
        ref.b2 = static_cast<std::uint32_t>(next[2]);
        ref.b3 = static_cast<std::uint32_t>(next[3]);
        ref.b4 = static_cast<std::uint32_t>(next[4]);
        ref.b5 = static_cast<std::uint32_t>(next[5]);
        ref.b6 = static_cast<std::uint32_t>(next[6]);
    } while (std::next_permutation(next, next + 7));
    assert(index == SevenBagPermutations::SIZE);
    return permutations;
}

inline static void randomPieces(PieceType dest[], std::uint32_t& seed)
{
    static const SevenBagPermutations seven_bag_permutations = generateSevenBagPermutations();
    auto& ref = seven_bag_permutations.data[xorshf32(seed) % SevenBagPermutations::SIZE];
    dest[0] = static_cast<PieceType>(ref.b0);
    dest[1] = static_cast<PieceType>(ref.b1);
    dest[2] = static_cast<PieceType>(ref.b2);
    dest[3] = static_cast<PieceType>(ref.b3);
    dest[4] = static_cast<PieceType>(ref.b4);
    dest[5] = static_cast<PieceType>(ref.b5);
    dest[6] = static_cast<PieceType>(ref.b6);
}

inline static void initializeBoard(Board& board)
{
    using Initializer = IndexGenerator<Wrapper, BOARD_HEIGHT>::result::BoardInitializer<BOARD_HEIGHT, BOARD_FLOOR,
                                                                                        ROW_EMPTY, // row data
                                                                                        ROW_FULL>; // wall data
    board = Initializer::board;
}

// returns {is_tspin, is_mini_tspin}
inline static std::pair<bool, bool> isTspin(State* state)
{
    if (state->current != PieceType::T || !state->was_last_rotation) { return {false, false}; }
    // check corners around the T piece
    // |0# | #1| # | # |
    // |###|###|###|###|
    // |   |   |2  |  3|
    bool corners[4] = {
        !ops::canPlaceRows(state->board, Rows<1>{mkrow("B  ")}, state->x, state->y),
        !ops::canPlaceRows(state->board, Rows<1>{mkrow("  B")}, state->x, state->y),
        !ops::canPlaceRows(state->board, Rows<1>{mkrow("B  ")}, state->x, state->y + 2),
        !ops::canPlaceRows(state->board, Rows<1>{mkrow("  B")}, state->x, state->y + 2),
    };
    int count = 0;
    for (int i = 0; i < 4; ++i) { count += corners[i]; }
    if (count < 3) { return {false, false}; }
    // check for mini tspin (https://tetris.wiki/T-Spin)
    // orientation -> two corner indices
    constexpr std::pair<std::uint32_t, std::uint32_t> mini_check_corner_table[] = {
        {0, 1},
        {1, 3},
        {2, 3},
        {0, 2},
    };
    bool is_mini = state->srs_index < 4 // not mini if used 5th kick
                   && (corners[mini_check_corner_table[state->orientation].first] == 0 || corners[mini_check_corner_table[state->orientation].second] == 0);
    return {true, is_mini};
}
inline static bool isAllSpin(State* state)
{
    if (state->current == PieceType::T || !state->was_last_rotation) { return false; }
    auto& piece = ops::getPiece(state->current, state->orientation);
    // check all-spin (piece cannot move left, right, up or down)
    bool collisions[4] = {
        !ops::canPlacePiece(state->board, piece, state->x - 1, state->y), // left
        !ops::canPlacePiece(state->board, piece, state->x + 1, state->y), // right
        !ops::canPlacePiece(state->board, piece, state->x, state->y - 1), // up
        !ops::canPlacePiece(state->board, piece, state->x, state->y + 1), // down
    };
    return collisions[0] && collisions[1] && collisions[2] && collisions[3];
}
inline static SpinType getSpinType(State* state)
{
    if (state->current == PieceType::T) {
        auto [is_tspin, is_mini] = isTspin(state);
        if (is_tspin) { return is_mini ? SpinType::SPIN_MINI : SpinType::SPIN; }
    } else if (isAllSpin(state)) {
        return SpinType::SPIN_MINI;
    }
    return SpinType::NONE;
}
inline static bool isBackToBackSpinType(PieceType piece_type, SpinType spin_type)
{
    // current ruleset: tspin only
    if (piece_type != PieceType::T) { return false; }
    if (spin_type == SpinType::SPIN || spin_type == SpinType::SPIN_MINI) { return true; }
    return false;
}

inline static int calculateAttack(const State* state)
{
    // Jstris attack table (https://jstris.jezevec10.com/guide#attack-and-combo-table, https://tetris.wiki/Jstris#Details)
    // | Attack Type        | Lines Sent || Combo # | Lines Sent |
    // | 0 lines            |          0 ||       0 |          0 |
    // | 1 line  (single)   |          0 ||       1 |          0 |
    // | 2 lines (double)   |          1 ||       2 |          1 |
    // | 3 lines (triple)   |          2 ||       3 |          1 |
    // | 4 lines            |          4 ||       4 |          1 |
    // | T-spin Double      |          4 ||       5 |          2 |
    // | T-spin Triple      |          6 ||       6 |          2 |
    // | T-spin Single      |          2 ||       7 |          3 |
    // | T-spin Mini Single |          0 ||       8 |          3 |
    // | Perfect Clear      |         10 ||       9 |          4 |
    // | Back-to-Back       |         +1 ||      10 |          4 |
    // |                    |            ||      11 |          4 |
    // |                    |            ||     12+ |          5 |
    if (state->lines_cleared == 0) { return 0; }
    bool is_tspin = state->current == PieceType::T && state->spin_type == SpinType::SPIN;
    bool is_mini_tspin = state->current == PieceType::T && state->spin_type == SpinType::SPIN_MINI;
    bool is_b2b = state->back_to_back_count > 0;
    // --- Base attack ---
    int base = 0;
    if (is_tspin) {
        // T-spin Single = 2, Double = 4, Triple = 6
        base = state->lines_cleared << 1;
    } else if (is_mini_tspin) {
        // T-spin Mini Single = 0, Double = 4
        base = (state->lines_cleared == 2) << 2;
    } else {
        // normal clears
        switch (state->lines_cleared) {
            case 1: base = 0; break;
            case 2: base = 1; break;
            case 3: base = 2; break;
            case 4: base = 4; break;
            default: assert(false); break;
        }
    }
    // --- Perfect Clear ---
    if (state->perfect_clear) { base = 10; }
    // --- Back-to-Back bonus ---
    // B2B applies to Tetris and T-spins, but NOT Mini T-spin Singles
    if (is_b2b && (state->lines_cleared >= 4 || is_tspin)) { base += 1; }
    // --- Combo bonus ---
    // combo_count: -1 = no combo, 0 = first clear (combo 0), 1 = second consecutive (combo 1), ...
    constexpr int combo_table[] = {0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5};
    constexpr int combo_table_size = sizeof(combo_table) / sizeof(combo_table[0]);
    if (state->combo_count >= 0) {
        int combo_index = state->combo_count;
        if (combo_index >= combo_table_size) { combo_index = combo_table_size - 1; }
        base += combo_table[combo_index];
    }
    return base;
}

inline static void applyGarbage(Board& board, int lines, int hole_position)
{
    if (lines <= 0) { return; }
    // shift up
    for (int i = 0; i + lines <= BOARD_BOTTOM; ++i) {
        board.data[i] = board.data[i + lines];
    }
    // add garbage rows
    const Row row = ROW_GARBAGE & ~ops::shift(CELL_MASK, hole_position);
    for (int i = 0; i < lines; ++i) {
        board.data[BOARD_BOTTOM - i] = row;
    }
}

inline static int processGarbageAndCounterAttack(State* state, int attack)
{
    int garbage_begin = 0, garbage_end = 0; // [begin, end) of pending garbage segments
    // decrement delays and records the range of pending garbage
    for (int i = 0; i < GARBAGE_QUEUE_SIZE; ++i) {
        if (state->garbage_queue[i] > 0) {
            if (state->garbage_delay[i] > 0) { state->garbage_delay[i]--; }
            garbage_end = i + 1; // update end to include this entry
        } else {
            break; // stop at first empty entry
        }
    }
    // counter garbage with attack and update the range
    int remaining_attack = attack;
    for (int i = garbage_begin; remaining_attack > 0 && i < garbage_end; ++i) {
        assert(state->garbage_queue[i] > 0);
        if (state->garbage_queue[i] <= remaining_attack) { // fully countered
            remaining_attack -= state->garbage_queue[i];
            state->garbage_queue[i] = 0;
            state->garbage_delay[i] = 0;
            garbage_begin++; // track begin index
        } else {             // partially countered
            state->garbage_queue[i] -= remaining_attack;
            remaining_attack = 0;
        }
    }
    // apply pending garbage with zero delay if no garbage blocking or no lines cleared
    if (!state->garbage_blocking || state->lines_cleared == 0) {
        std::uint16_t total_garbage_spawned = 0;
        for (int i = garbage_begin; i < garbage_end; ++i) {
            assert(state->garbage_queue[i] > 0);
            if (state->garbage_delay[i] > 0) {
                assert(garbage_begin == i);
                break; // stop at first delayed garbage (assume delays are in order)
            }
            std::uint8_t lines_to_spawn = state->garbage_queue[i];
            if (total_garbage_spawned + lines_to_spawn > state->max_garbage_spawn) {
                lines_to_spawn = static_cast<std::uint8_t>(state->max_garbage_spawn - total_garbage_spawned);
            }
            assert(lines_to_spawn > 0);
            total_garbage_spawned += lines_to_spawn;
            // generate random hole position
            int hole_position = xorshf32(state->garbage_seed) % (BOARD_RIGHT - BOARD_LEFT + 1) + BOARD_LEFT;
            // apply this segment of garbage
            applyGarbage(state->board, lines_to_spawn, hole_position);
            // update remaining garbage in this entry
            state->garbage_queue[i] -= lines_to_spawn;
            if (state->garbage_queue[i] == 0) {
                state->garbage_delay[i] = 0;
                garbage_begin++; // track begin index
                assert(garbage_begin == i + 1);
            } else {
                assert(garbage_begin == i);
                break; // stop if this entry is not fully applied (only happens when max_garbage_spawn is reached)
            }
        }
    }
    // shift remaining garbage entries to the front of the queue [garbage_begin, garbage_end) -> [0, garbage_end - garbage_begin)
    if (garbage_begin > 0) {
        int write_index = 0;
        for (int read_index = garbage_begin; read_index < garbage_end; ++read_index) {
            assert(state->garbage_queue[read_index] > 0);
            if (write_index != read_index) {
                state->garbage_queue[write_index] = state->garbage_queue[read_index];
                state->garbage_delay[write_index] = state->garbage_delay[read_index];
            }
            write_index++;
        }
        // clear remaining entries after compacted data
        for (int i = write_index; i < garbage_end; ++i) {
            state->garbage_queue[i] = 0;
            state->garbage_delay[i] = 0;
        }
    }
    return remaining_attack;
}

inline static std::uint16_t clearLines(State* state)
{
    int count = 0;
    for (int i = BOARD_BOTTOM; i >= 0; i--) {
        while (i - count >= 0 && (state->board.data[i - count] & ROW_FULL) == ROW_FULL) {
            count++;
        }
        if (i - count >= 0) {
            state->board.data[i] = state->board.data[i - count];
        } else {
            state->board.data[i] = ROW_EMPTY;
        }
    }
    return static_cast<std::uint16_t>(count);
}
inline static void processPiecePlacement(State* state)
{
    // place the current piece on the board
    auto& piece = ops::getPiece(state->current, state->orientation);
    ops::placePiece(state->board, piece, state->x, state->y);
    // clear lines and update state
    state->lines_cleared = clearLines(state);
    state->total_lines_cleared += state->lines_cleared;
    state->piece_count++;
    if (state->lines_cleared > 0) {
        // check perfect clear
        state->perfect_clear = true;
        for (int i = 0; i <= BOARD_BOTTOM; ++i) {
            if (state->board.data[i] != ROW_EMPTY) {
                state->perfect_clear = false;
                break;
            }
        }
        // update combo and back-to-back counts
        state->combo_count++;
        state->back_to_back_count = (isBackToBackSpinType(state->current, state->spin_type) || state->lines_cleared == 4) // T-spin or Tetris
                                        ? state->back_to_back_count + 1
                                        : -1;
    } else {
        state->combo_count = -1;
    }
    // calculate attack and counter garbage
    int attack = calculateAttack(state);
    int lines_sent = processGarbageAndCounterAttack(state, attack);
    // update attack and lines sent in state (capped at max values for uint16_t)
    assert(attack <= std::numeric_limits<std::uint16_t>::max());
    assert(lines_sent <= std::numeric_limits<std::uint16_t>::max());
    state->attack = static_cast<std::uint16_t>(attack);
    state->lines_sent = static_cast<std::uint16_t>(lines_sent);
    state->total_attack += static_cast<std::uint32_t>(attack);
    state->total_lines_sent += static_cast<std::uint32_t>(lines_sent);
}
inline static PieceType fetchNextPiece(State* state)
{
    PieceType next_piece = state->next[0];
    // shift the next pieces
    for (int i = 0; i < 13; i++) { state->next[i] = state->next[i + 1]; }
    state->next[13] = PieceType::NONE;
    // generate new random pieces if needed
    if (state->next[7] == PieceType::NONE) { randomPieces(state->next + 7, state->seed); }
    return next_piece;
}
inline static bool newCurrentPiece(State* state, PieceType piece_type)
{
    // set new current piece
    state->current = piece_type;
    state->orientation = 0;
    state->x = PIECE_SPAWN_X;
    state->y = PIECE_SPAWN_Y;
    // reset state for new piece
    state->srs_index = -1;
    state->was_last_rotation = false;
    // state->spin_type = SpinType::NONE; // postpone for tracking last spin type
    // spawn the new piece
    auto& piece = ops::getPiece(state->current, state->orientation);
    bool can_place = ops::canPlacePiece(state->board, piece, state->x, state->y);
    // Top out rule (https://tetris.wiki/Top_out)
    if (!can_place) {
        state->is_alive = false;
        return false;
    }
    return true;
}

inline static bool movePiece(State* state, int new_x, int new_y)
{
    auto& piece = ops::getPiece(state->current, state->orientation);
    bool can_place = ops::canPlacePiece(state->board, piece, new_x, new_y);
    if (can_place) {
        state->x = static_cast<std::int8_t>(new_x);
        state->y = static_cast<std::int8_t>(new_y);
    }
    return can_place;
}
inline static bool rotatePiece(State* state, Rotation rot)
{
    constexpr std::uint8_t orientation_delta_table[static_cast<std::underlying_type_t<Rotation>>(Rotation::SIZE)] = {
        1, // CW  -> orientation + 1
        3, // CCW -> orientation - 1 (= +3 mod 4)
        2, // 180 -> orientation + 2
    };
    const std::uint8_t new_orientation = (state->orientation + orientation_delta_table[static_cast<std::underlying_type_t<Rotation>>(rot)]) % 4;
    auto& new_piece = ops::getPiece(state->current, new_orientation);
    // SRS kicks for CW/CCW/180
    auto& [kicks, len] = srs_table[static_cast<std::underlying_type_t<PieceType>>(state->current)][state->orientation][static_cast<std::underlying_type_t<Rotation>>(rot)];
    // try SRS kicks
    for (int i = 0; i < len; ++i) {
        const int test_x = state->x + kicks[i].x;
        const int test_y = state->y - kicks[i].y;
        if (ops::canPlacePiece(state->board, new_piece, test_x, test_y)) {
            // commit rotation + kick
            state->orientation = new_orientation;
            state->x = static_cast<std::int8_t>(test_x);
            state->y = static_cast<std::int8_t>(test_y);
            state->srs_index = static_cast<std::int8_t>(i);
            return true;
        }
    }
    return false;
}

void setSeed(State* state, std::uint32_t seed, std::uint32_t garbage_seed)
{
    state->seed = seed;
    state->garbage_seed = garbage_seed;
}

void reset(State* state)
{
    initializeBoard(state->board);
    state->is_alive = true;
    randomPieces(state->next, state->seed);
    randomPieces(state->next + 7, state->seed);
    state->hold = PieceType::NONE;
    state->has_held = false;
    state->current = PieceType::NONE;
    state->orientation = 0;
    state->x = -1;
    state->y = -1;
    state->srs_index = -1;
    state->piece_count = 0;
    state->was_last_rotation = false;
    state->spin_type = SpinType::NONE;
    state->perfect_clear = false;
    // -1 because the first clear is not considered a back-to-back or a combo
    state->back_to_back_count = -1;
    state->combo_count = -1;
    state->lines_cleared = 0;
    state->attack = 0;
    state->lines_sent = 0;
    state->total_lines_cleared = 0;
    state->total_attack = 0;
    state->total_lines_sent = 0;
    // initialize garbage queues
    std::fill(std::begin(state->garbage_queue), std::end(state->garbage_queue), 0);
    std::fill(std::begin(state->garbage_delay), std::end(state->garbage_delay), 0);
    // spawn current piece
    PieceType next_piece = fetchNextPiece(state);
    newCurrentPiece(state, next_piece);
}

// reset output fields from the last piece placement
inline static void clearLastPlacementResult(State* state)
{
    state->perfect_clear = false;
    state->lines_cleared = 0;
    state->attack = 0;
    state->lines_sent = 0;
}

bool moveLeft(State* state)
{
    clearLastPlacementResult(state);
    bool moved = movePiece(state, state->x - 1, state->y);
    if (moved) {
        state->was_last_rotation = false;
        state->spin_type = SpinType::NONE;
    }
    return moved;
}
bool moveRight(State* state)
{
    clearLastPlacementResult(state);
    bool moved = movePiece(state, state->x + 1, state->y);
    if (moved) {
        state->was_last_rotation = false;
        state->spin_type = SpinType::NONE;
    }
    return moved;
}
bool moveLeftToWall(State* state)
{
    clearLastPlacementResult(state);
    bool moved = false;
    while (movePiece(state, state->x - 1, state->y)) { moved = true; }
    if (moved) {
        state->was_last_rotation = false;
        state->spin_type = SpinType::NONE;
    }
    return moved;
}
bool moveRightToWall(State* state)
{
    clearLastPlacementResult(state);
    bool moved = false;
    while (movePiece(state, state->x + 1, state->y)) { moved = true; }
    if (moved) {
        state->was_last_rotation = false;
        state->spin_type = SpinType::NONE;
    }
    return moved;
}
bool softDrop(State* state)
{
    clearLastPlacementResult(state);
    bool moved = movePiece(state, state->x, state->y + 1);
    if (moved) {
        state->was_last_rotation = false;
        state->spin_type = SpinType::NONE;
    }
    return moved;
}
bool softDropToFloor(State* state)
{
    clearLastPlacementResult(state);
    bool moved = false;
    while (movePiece(state, state->x, state->y + 1)) { moved = true; }
    if (moved) {
        state->was_last_rotation = false;
        state->spin_type = SpinType::NONE;
    }
    return moved;
}
bool hardDrop(State* state)
{
    bool moved = false;
    while (movePiece(state, state->x, state->y + 1)) { moved = true; }
    if (moved) { state->was_last_rotation = false; }
    state->spin_type = getSpinType(state); // TODO: remove redundant check
    processPiecePlacement(state);
    PieceType next_piece = fetchNextPiece(state);
    if (newCurrentPiece(state, next_piece)) {
        state->has_held = false;
        return true;
    }
    return false;
}
bool rotateCounterclockwise(State* state)
{
    clearLastPlacementResult(state);
    bool moved = rotatePiece(state, Rotation::CCW);
    if (moved) {
        state->was_last_rotation = true;
        state->spin_type = getSpinType(state);
    }
    return moved;
}
bool rotateClockwise(State* state)
{
    clearLastPlacementResult(state);
    bool moved = rotatePiece(state, Rotation::CW);
    if (moved) {
        state->was_last_rotation = true;
        state->spin_type = getSpinType(state);
    }
    return moved;
}
bool rotate180(State* state)
{
    clearLastPlacementResult(state);
    bool moved = rotatePiece(state, Rotation::HALF);
    if (moved) {
        state->was_last_rotation = true;
        state->spin_type = getSpinType(state);
    }
    return moved;
}
bool hold(State* state)
{
    clearLastPlacementResult(state);
    if (state->has_held) { return false; }
    // reset state for new piece (TODO: remove redundancy with newCurrentPiece)
    state->was_last_rotation = false;
    state->spin_type = SpinType::NONE;
    // hold current piece
    PieceType new_piece = state->hold != PieceType::NONE ? state->hold : fetchNextPiece(state);
    state->hold = state->current;
    state->has_held = true;
    // setup new current piece
    newCurrentPiece(state, new_piece);
    return true;
}
bool noop(State* state)
{
    clearLastPlacementResult(state);
    return true;
}

bool addGarbage(State* state, std::uint8_t lines, std::uint8_t delay)
{
    if (lines == 0) { return false; }
    for (int i = 0; i < GARBAGE_QUEUE_SIZE; ++i) {
        if (state->garbage_queue[i] == 0) {
            assert(state->garbage_delay[i] == 0);
            state->garbage_queue[i] = lines;
            state->garbage_delay[i] = delay;
            return true;
        }
    }
    // TODO: garbage queue full, maybe append to the last entry?
    return false;
}

void toString(State* state, char* buf, std::size_t size)
{
    // coordinates below: x is in units of 2 chars (each cell = 2 characters wide)
    constexpr int STRING_BOARD_WIDTH = 23;
    constexpr int STRING_BOARD_HEIGHT = 22;
    constexpr int STRING_BOARD_LEFT = 7;
    constexpr int STRING_BOARD_TOP = 0;
    constexpr int STRING_HOLD_X = 1;
    constexpr int STRING_HOLD_Y = 1;
    constexpr int STRING_NEXT_X = 18;
    constexpr int STRING_NEXT_Y = 1;
    constexpr int STRING_NEXT_SPACING = 3;
    // garbage coordinates: x is in units of 1 char
    constexpr int STRING_GARBAGE_TOP = 0;
    constexpr int STRING_GARBAGE_BOTTOM = 20;
    constexpr int STRING_GARBAGE_LEFT = 12;
    struct StringLayout {
        char board[STRING_BOARD_HEIGHT][STRING_BOARD_WIDTH * 2 + 1]; // +1 for newline
    };
    struct StringCell {
        char left, right;
    };
    constexpr char initial_board[] =
        "XXXXXXXXXXXX X                    XXXXXXXXXXXX\n"
        "XX        XX X                    XX        XX\n"
        "XX        XX X                    XX        XX\n"
        "XXXXXXXXXXXX X                    XXXXXXXXXXXX\n"
        "          XX X                    XX        XX\n"
        "          XX X                    XX        XX\n"
        "          XX X                    XXXXXXXXXXXX\n"
        "          XX X                    XX        XX\n"
        "          XX X                    XX        XX\n"
        "          XX X                    XXXXXXXXXXXX\n"
        "          XX X                    XX        XX\n"
        "          XX X                    XX        XX\n"
        "          XX X                    XXXXXXXXXXXX\n"
        "          XX X                    XX        XX\n"
        "          XX X                    XX        XX\n"
        "          XX X                    XXXXXXXXXXXX\n"
        "          XX X                    XX          \n"
        "          XX X                    XX          \n"
        "          XX X                    XX          \n"
        "          XX X                    XX          \n"
        "          XX X                    XX          \n"
        "          XXXXXXXXXXXXXXXXXXXXXXXXXX          \0";
    // symbols for representing pending garbage with different delays (0-9, A-Z, then *)
    static constexpr char pending_garbage_symbols[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ*";
    static constexpr int pending_garbage_symbol_size = sizeof(pending_garbage_symbols) - 1; // exclude null terminator
    // some pieces are shifted half a cell to the right in the string representation to look better (since the string cells are twice as wide as the board cells)
    static constexpr bool half_shift[] = {
        // Z,    L,     O,    S,     I,    J,    T
        true, true, false, true, false, true, true};
    // helper functions for drawing the board and pieces into the string layout
    static constexpr auto drawStringCell = [](StringLayout& sl, int string_x, int string_y, StringCell string_cell, bool half_shift = false) {
        if (string_x >= 0 && string_x < STRING_BOARD_WIDTH && string_y >= 0 && string_y < STRING_BOARD_HEIGHT) {
            auto target_cell = &sl.board[string_y][string_x * 2 + half_shift];
            target_cell[0] = string_cell.left;
            target_cell[1] = string_cell.right;
        }
    };
    static constexpr auto drawPiece = [](StringLayout& sl, int string_x, int string_y, PieceType piece_type, std::uint8_t orientation, bool half_shift = false, StringCell string_cell = {'[', ']'}) {
        auto& piece = ops::getPiece(piece_type, orientation);
        // put 4x4 piece shape into the string layout
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                if (ops::getCell(piece, j, i) == Cell::EMPTY) { continue; }
                drawStringCell(sl, string_x + j, string_y + i, string_cell, half_shift);
            }
        }
    };
    static constexpr auto drawHold = [](StringLayout& sl, PieceType hold) {
        if (hold == PieceType::NONE) { return; }
        drawPiece(sl, STRING_HOLD_X, STRING_HOLD_Y, hold, 0, half_shift[static_cast<std::underlying_type_t<PieceType>>(hold)]);
    };
    static constexpr auto drawNext = [](StringLayout& sl, const PieceType* next) {
        for (int i = 0; i < 5; ++i) {
            if (next[i] == PieceType::NONE) { break; }
            drawPiece(sl, STRING_NEXT_X, STRING_NEXT_Y + i * STRING_NEXT_SPACING, next[i], 0, half_shift[static_cast<std::underlying_type_t<PieceType>>(next[i])]);
        }
    };
    static constexpr auto drawPendingGarbageQueue = [](StringLayout& sl, std::uint8_t garbage_queue[], std::uint8_t garbage_delay[]) {
        int string_y = STRING_GARBAGE_BOTTOM;
        for (int i = 0; i < GARBAGE_QUEUE_SIZE; ++i) {
            if (garbage_queue[i] == 0 || string_y < STRING_GARBAGE_TOP) { break; }
            int length = garbage_queue[i];
            int delay = garbage_delay[i];
            char symbol = delay <= pending_garbage_symbol_size ? pending_garbage_symbols[delay] : pending_garbage_symbols[pending_garbage_symbol_size - 1];
            for (; string_y >= STRING_GARBAGE_TOP && length > 0; --string_y, --length) {
                sl.board[string_y][STRING_GARBAGE_LEFT] = symbol;
            }
        }
    };
    // sanity check for buffer size
    if (size < sizeof(StringLayout)) { return; }
    // view buf as StringLayout
    StringLayout* sl = reinterpret_cast<StringLayout*>(buf);
    // calculate shadow position
    auto& piece = ops::getPiece(state->current, state->orientation);
    int shadow_y = state->y;
    while (ops::canPlacePiece(state->board, piece, state->x, shadow_y + 1)) {
        shadow_y++;
    }
    // copy the initial board layout
    memcpy(sl->board, initial_board, sizeof(sl->board));
    // draw state to buf
    for (int y = BOARD_TOP - 1; y <= BOARD_BOTTOM; ++y) {
        for (int x = BOARD_LEFT; x <= BOARD_RIGHT; ++x) {
            Cell cell = ops::getCell(state->board, x, y);
            StringCell string_cell;
            switch (cell) {
                case Cell::EMPTY: string_cell = {' ', ' '}; break;
                case Cell::BLOCK: string_cell = {'[', ']'}; break;
                case Cell::GARBAGE: string_cell = {'#', '#'}; break;
                default:
                    string_cell = {'-', '-'};
                    assert(false);
                    break;
            }
            int string_x = x - BOARD_LEFT + STRING_BOARD_LEFT;
            int string_y = y - (BOARD_TOP - 1) + STRING_BOARD_TOP;
            drawStringCell(*sl, string_x, string_y, string_cell);
        }
    }
    // draw shadow piece
    int shadow_string_x = state->x - BOARD_LEFT + STRING_BOARD_LEFT;
    int shadow_string_y = shadow_y - (BOARD_TOP - 1) + STRING_BOARD_TOP;
    drawPiece(*sl, shadow_string_x, shadow_string_y, state->current, state->orientation, false, {':', ':'});
    // draw current piece
    int current_string_x = state->x - BOARD_LEFT + STRING_BOARD_LEFT;
    int current_string_y = state->y - (BOARD_TOP - 1) + STRING_BOARD_TOP;
    drawPiece(*sl, current_string_x, current_string_y, state->current, state->orientation);
    // draw hold and next pieces
    drawHold(*sl, state->hold);
    drawNext(*sl, state->next);
    // draw pending garbage queue
    drawPendingGarbageQueue(*sl, state->garbage_queue, state->garbage_delay);
}

void placeCurrentPiece(State* state) { ops::placePiece(state->board, ops::getPiece(state->current, state->orientation), state->x, state->y); }
void removeCurrentPiece(State* state) { ops::removePiece(state->board, ops::getPiece(state->current, state->orientation), state->x, state->y); }
bool canPlaceCurrentPiece(State* state) { return ops::canPlacePiece(state->board, ops::getPiece(state->current, state->orientation), state->x, state->y); }

} // namespace minizero::env::moderntetris::engine
