#pragma once
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace minizero::env::moderntetris::engine {

constexpr int BOARD_HEIGHT = 32; // total rows
constexpr int BOARD_WIDTH = 16;  // total columns (2-bit cells per 32-bit row)

constexpr int BOARD_TOP = 9;     // first visible row
constexpr int BOARD_BOTTOM = 28; // last visible row
constexpr int BOARD_LEFT = 3;    // first playfield column
constexpr int BOARD_RIGHT = 12;  // last playfield column

constexpr int BOARD_FLOOR = BOARD_HEIGHT - BOARD_BOTTOM - 1; // floor wall thickness

constexpr int PIECE_SPAWN_X = BOARD_LEFT + 3;
constexpr int PIECE_SPAWN_Y = BOARD_TOP - 1;

constexpr int GARBAGE_QUEUE_SIZE = 20;

using Row = std::uint32_t;
template <std::size_t N>
struct Rows {
    enum { SIZE = N };
    Row data[N];
};
using Board = Rows<BOARD_HEIGHT>;
using Piece = Rows<4>;

/**
 * Cell encoding (2 bits per cell):
 * bit1 bit0
 *    0    0 EMPTY   - unoccupied
 *    0    1 BLOCK   - occupied (placed piece)
 *    1    0 (reserved)
 *    1    1 GARBAGE - occupied (garbage line)
 * bit0 = "occupied" flag. All collision/line-clear logic tests bit0 only.
 * GARBAGE is intentionally a superset of BLOCK (bit0 set + bit1 garbage flag).
 */
enum class Cell : Row {
    EMPTY = 0b00'000000000000000000000000000000u,
    BLOCK = 0b01'000000000000000000000000000000u,
    GARBAGE = 0b11'000000000000000000000000000000u,
};
constexpr Row CELL_MASK = 0b11'000000000000000000000000000000u;

/**
 * Creates a 32-bit row representation from a C-style null-terminated string.
 * The string must be at most 16 characters long (excluding '\0').
 * Each character represents a cell: 'B' (BLOCK), 'G' (GARBAGE),
 * or any other character (treated as EMPTY).
 * The input string is aligned to the most significant bits of the 32-bit row.
 * If the string is shorter than 16 characters, remaining bits are treated as EMPTY.
 */
template <int N, typename = std::enable_if_t<(N <= 17)>>
constexpr Row mkrow(const char (&s)[N])
{
    Row bits = 0;
    for (int i = 0; s[i]; ++i) {
        switch (s[i]) {
            case 'B': bits |= (static_cast<Row>(Cell::BLOCK) >> (i << 1)); break;
            case 'G': bits |= (static_cast<Row>(Cell::GARBAGE) >> (i << 1)); break;
            default: bits |= (static_cast<Row>(Cell::EMPTY) >> (i << 1)); break;
        }
    }
    return bits;
}

// Row patterns
constexpr Row ROW_EMPTY = mkrow("BBB..........BBB");   // walls + empty row
constexpr Row ROW_FULL = mkrow("BBBBBBBBBBBBBBBB");    // walls + full row
constexpr Row ROW_GARBAGE = mkrow("BBBGGGGGGGGGGBBB"); // walls + full garbage row

enum class PieceType : std::int8_t {
    NONE = -1,
    Z = 0,
    L,
    O,
    S,
    I,
    J,
    T,
    SIZE
};

constexpr Piece pieces[static_cast<std::underlying_type_t<PieceType>>(PieceType::SIZE)][4] = {
    {{mkrow("BB  "),
      mkrow(" BB "),
      mkrow("    "),
      mkrow("    ")},
     {mkrow("  B "),
      mkrow(" BB "),
      mkrow(" B  "),
      mkrow("    ")},
     {mkrow("    "),
      mkrow("BB  "),
      mkrow(" BB "),
      mkrow("    ")},
     {mkrow(" B  "),
      mkrow("BB  "),
      mkrow("B   "),
      mkrow("    ")}},
    {{mkrow("  B "),
      mkrow("BBB "),
      mkrow("    "),
      mkrow("    ")},
     {mkrow(" B  "),
      mkrow(" B  "),
      mkrow(" BB "),
      mkrow("    ")},
     {mkrow("    "),
      mkrow("BBB "),
      mkrow("B   "),
      mkrow("    ")},
     {mkrow("BB  "),
      mkrow(" B  "),
      mkrow(" B  "),
      mkrow("    ")}},
    {{mkrow(" BB "),
      mkrow(" BB "),
      mkrow("    "),
      mkrow("    ")},
     {mkrow(" BB "),
      mkrow(" BB "),
      mkrow("    "),
      mkrow("    ")},
     {mkrow(" BB "),
      mkrow(" BB "),
      mkrow("    "),
      mkrow("    ")},
     {mkrow(" BB "),
      mkrow(" BB "),
      mkrow("    "),
      mkrow("    ")}},
    {{mkrow(" BB "),
      mkrow("BB  "),
      mkrow("    "),
      mkrow("    ")},
     {mkrow(" B  "),
      mkrow(" BB "),
      mkrow("  B "),
      mkrow("    ")},
     {mkrow("    "),
      mkrow(" BB "),
      mkrow("BB  "),
      mkrow("    ")},
     {mkrow("B   "),
      mkrow("BB  "),
      mkrow(" B  "),
      mkrow("    ")}},
    {{mkrow("    "),
      mkrow("BBBB"),
      mkrow("    "),
      mkrow("    ")},
     {mkrow("  B "),
      mkrow("  B "),
      mkrow("  B "),
      mkrow("  B ")},
     {mkrow("    "),
      mkrow("    "),
      mkrow("BBBB"),
      mkrow("    ")},
     {mkrow(" B  "),
      mkrow(" B  "),
      mkrow(" B  "),
      mkrow(" B  ")}},
    {{mkrow("B   "),
      mkrow("BBB "),
      mkrow("    "),
      mkrow("    ")},
     {mkrow(" BB "),
      mkrow(" B  "),
      mkrow(" B  "),
      mkrow("    ")},
     {mkrow("    "),
      mkrow("BBB "),
      mkrow("  B "),
      mkrow("    ")},
     {mkrow(" B  "),
      mkrow(" B  "),
      mkrow("BB  "),
      mkrow("    ")}},
    {{mkrow(" B  "),
      mkrow("BBB "),
      mkrow("    "),
      mkrow("    ")},
     {mkrow(" B  "),
      mkrow(" BB "),
      mkrow(" B  "),
      mkrow("    ")},
     {mkrow("    "),
      mkrow("BBB "),
      mkrow(" B  "),
      mkrow("    ")},
     {mkrow(" B  "),
      mkrow("BB  "),
      mkrow(" B  "),
      mkrow("    ")}}};

enum class SpinType : std::uint8_t {
    NONE,
    SPIN,
    SPIN_MINI
};

struct State {
    Board board;
    std::uint8_t /* bool */ is_alive;
    PieceType next[14];
    PieceType hold;
    std::uint8_t /* bool */ has_held;
    PieceType current;
    std::uint8_t orientation;
    std::int8_t x, y;
    std::uint32_t seed;
    std::int8_t srs_index;
    std::uint32_t piece_count;
    // TODO: remove was_last_rotation and use spin_type only
    std::uint8_t /* bool */ was_last_rotation; // Indicates if the last successful action was a rotation
    SpinType spin_type;                        // Type of spin (NONE, SPIN, SPIN_MINI) for the last piece placement
    std::uint8_t /* bool */ perfect_clear;     // Last piece placement resulted in a perfect clear
    std::int32_t back_to_back_count;
    std::int32_t combo_count;
    std::uint16_t lines_cleared;       // Lines cleared by the last piece placement
    std::uint16_t attack;              // Attack generated by the last piece placement
    std::uint16_t lines_sent;          // Lines sent to opponent from the last piece placement
    std::uint32_t total_lines_cleared; // Total lines cleared
    std::uint32_t total_attack;        // Total attack accumulated
    std::uint32_t total_lines_sent;    // Total lines sent to opponent
    std::uint32_t garbage_seed;
    std::uint8_t garbage_queue[GARBAGE_QUEUE_SIZE];
    std::uint8_t garbage_delay[GARBAGE_QUEUE_SIZE];
    // Configurations (TODO: pack configurations into a struct?)
    std::uint8_t max_garbage_spawn = 6;              // Maximum garbage lines that can be placed at once
    std::uint8_t /* bool */ garbage_blocking = true; // If true, clears temporarily block garbage placement
};

// Super Rotation System (https://harddrop.com/wiki/SRS)
struct SRSKickData {
    struct Kick {
        std::int8_t x, y;
    };
    const Kick* kicks;
    const int length;
};

enum class Rotation : std::uint8_t {
    CW,
    CCW,
    HALF, // 180
    SIZE
};

extern const SRSKickData srs_table[static_cast<std::underlying_type_t<PieceType>>(PieceType::SIZE)][4][static_cast<std::underlying_type_t<Rotation>>(Rotation::SIZE)];

namespace ops {

    inline constexpr Row shift(Row row, int dx)
    {
        return (dx >= 0) ? (row >> (dx << 1)) : (row << (-dx << 1));
    }
    template <std::size_t N>
    inline constexpr Rows<N> shift(const Rows<N>& rows, int dx)
    {
        Rows<N> result = {};
        if (dx >= 0) {
            const int x_offset = dx << 1;
            for (std::size_t i = 0; i < N; ++i) { result.data[i] = rows.data[i] >> x_offset; }
        } else {
            const int x_offset = -dx << 1;
            for (std::size_t i = 0; i < N; ++i) { result.data[i] = rows.data[i] << x_offset; }
        }
        return result;
    }

    inline constexpr void setCell(Row& row, int x, Cell cell)
    {
        row = (row & ~shift(CELL_MASK, x)) | shift(static_cast<Row>(cell), x);
    }
    template <std::size_t N>
    inline constexpr void setCell(Rows<N>& rows, int x, int y, Cell cell) { setCell(rows.data[y], x, cell); }
    inline constexpr Cell getCell(Row row, int x)
    {
        return static_cast<Cell>(shift(row, -x) & CELL_MASK);
    }
    template <std::size_t N>
    inline constexpr Cell getCell(const Rows<N>& rows, int x, int y) { return getCell(rows.data[y], x); }

    template <std::size_t N>
    inline constexpr void placeRows(Board& board, const Rows<N>& rows, int x, int y)
    {
        for (std::size_t i = 0; i < N; ++i) { board.data[static_cast<std::size_t>(y) + i] |= shift(rows.data[i], x); }
    }
    // Precondition: only remove rows that were previously placed on empty cells via placeRows.
    template <std::size_t N>
    inline constexpr void removeRows(Board& board, const Rows<N>& rows, int x, int y)
    {
        for (std::size_t i = 0; i < N; ++i) { board.data[static_cast<std::size_t>(y) + i] &= ~shift(rows.data[i], x); }
    }
    template <std::size_t N>
    inline constexpr bool canPlaceRows(const Board& board, const Rows<N>& rows, int x, int y)
    {
        if (y < 0 || static_cast<std::size_t>(y) + N > Board::SIZE) { return false; }
        for (std::size_t i = 0; i < N; ++i) {
            if ((board.data[static_cast<std::size_t>(y) + i] & shift(rows.data[i], x)) != 0) { return false; }
        }
        return true;
    }

    inline constexpr const Piece& getPiece(PieceType type, std::uint8_t orientation) { return pieces[static_cast<std::underlying_type_t<PieceType>>(type)][orientation]; }
    inline constexpr void placePiece(Board& board, const Piece& piece, int x, int y) { placeRows(board, piece, x, y); }
    inline constexpr void removePiece(Board& board, const Piece& piece, int x, int y) { removeRows(board, piece, x, y); }
    inline constexpr bool canPlacePiece(const Board& board, const Piece& piece, int x, int y) { return canPlaceRows(board, piece, x, y); }

} // namespace ops

void setSeed(State* state, std::uint32_t seed, std::uint32_t garbage_seed);

void reset(State* state);

bool moveLeft(State* state);
bool moveRight(State* state);
bool moveLeftToWall(State* state);
bool moveRightToWall(State* state);
bool softDrop(State* state);
bool softDropToFloor(State* state);
bool hardDrop(State* state);
bool rotateCounterclockwise(State* state);
bool rotateClockwise(State* state);
bool rotate180(State* state);
bool hold(State* state);
bool noop(State* state);

bool addGarbage(State* state, std::uint8_t lines, std::uint8_t delay);

void toString(State* state, char* buf, std::size_t size);

void placeCurrentPiece(State* state);
void removeCurrentPiece(State* state);
bool canPlaceCurrentPiece(State* state);

} // namespace minizero::env::moderntetris::engine
