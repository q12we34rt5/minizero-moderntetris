#include "placement_search.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

namespace minizero::env::moderntetris::engine {
// namespace {

constexpr int X_OFFSET = 10;
constexpr int Y_OFFSET = 10;
constexpr int X_SIZE = 32;
constexpr int Y_SIZE = 64;
constexpr int ORIENTATION_SIZE = 4;
constexpr int WAS_ROTATION_SIZE = 2;
constexpr int SRS_MIN = -1;
constexpr int SRS_SIZE = 7;
constexpr int SPIN_TYPE_SIZE = 3;
constexpr std::size_t MAX_STRUCTURAL_STATES = static_cast<std::size_t>(X_SIZE) * Y_SIZE * ORIENTATION_SIZE * WAS_ROTATION_SIZE * SRS_SIZE;

struct SearchState {
    std::int8_t x;
    std::int8_t y;
    std::int8_t orientation;
    bool was_last_rotation;
    std::int8_t srs_index;
};

struct VisitedNode {
    bool visited;
    PlacementAction action;
    std::int8_t parent_x;
    std::int8_t parent_y;
    std::int8_t parent_orientation;
    bool parent_was_last_rotation;
    std::int8_t parent_srs_index;
};

inline SearchState canonicalize(SearchState state)
{
    if (!state.was_last_rotation) {
        state.srs_index = -1;
    }
    return state;
}

inline int xIndex(int x)
{
    const int index = x + X_OFFSET;
    assert(index >= 0 && index < X_SIZE);
    return index;
}

inline int yIndex(int y)
{
    const int index = y + Y_OFFSET;
    assert(index >= 0 && index < Y_SIZE);
    return index;
}

inline int orientationIndex(int orientation)
{
    assert(orientation >= 0 && orientation < ORIENTATION_SIZE);
    return orientation;
}

inline int srsIndex(int srs_index)
{
    const int index = srs_index - SRS_MIN;
    assert(index >= 0 && index < SRS_SIZE);
    return index;
}

inline int spinTypeIndex(SpinType spin_type)
{
    const int index = static_cast<int>(spin_type);
    assert(index >= 0 && index < SPIN_TYPE_SIZE);
    return index;
}

struct VisitedArray {
    VisitedNode data[X_SIZE][Y_SIZE][ORIENTATION_SIZE][WAS_ROTATION_SIZE][SRS_SIZE];

    VisitedArray()
    {
        std::memset(data, 0, sizeof(data));
    }

    bool get(const SearchState& state) const
    {
        return data[xIndex(state.x)][yIndex(state.y)][orientationIndex(state.orientation)][state.was_last_rotation][srsIndex(state.srs_index)].visited;
    }

    void set(const SearchState& state, PlacementAction action, const SearchState& parent)
    {
        auto& node = data[xIndex(state.x)][yIndex(state.y)][orientationIndex(state.orientation)][state.was_last_rotation][srsIndex(state.srs_index)];
        node.visited = true;
        node.action = action;
        node.parent_x = parent.x;
        node.parent_y = parent.y;
        node.parent_orientation = parent.orientation;
        node.parent_was_last_rotation = parent.was_last_rotation;
        node.parent_srs_index = parent.srs_index;
    }

    void clear(const SearchState& state)
    {
        data[xIndex(state.x)][yIndex(state.y)][orientationIndex(state.orientation)][state.was_last_rotation][srsIndex(state.srs_index)].visited = false;
    }

    void getParent(const SearchState& state, PlacementAction& action, SearchState& parent) const
    {
        const auto& node = data[xIndex(state.x)][yIndex(state.y)][orientationIndex(state.orientation)][state.was_last_rotation][srsIndex(state.srs_index)];
        action = node.action;
        parent.x = node.parent_x;
        parent.y = node.parent_y;
        parent.orientation = node.parent_orientation;
        parent.was_last_rotation = node.parent_was_last_rotation;
        parent.srs_index = node.parent_srs_index;
    }
};

// Scan the visible playfield rows top-down and return the first y with any
// non-empty cell (ignoring wall cells). Returns BOARD_BOTTOM + 1 if the entire
// playfield is empty. Used by findPlacements to skip the free-fall region:
// when the piece is strictly above the stack the BFS state space is huge but
// structurally redundant (every in-air y has the same successor topology), so
// we start BFS at the lowest y where the stack starts to matter and prepend
// N single-row soft-drops to reproduced paths.
inline int findHighestStackY(const Board& board)
{
    for (int y = BOARD_TOP; y <= BOARD_BOTTOM; ++y) {
        if ((board.data[y] & ~ROW_EMPTY) != 0u) { return y; }
    }
    return BOARD_BOTTOM + 1;
}

struct PlacementArray {
    bool data[X_SIZE][Y_SIZE][ORIENTATION_SIZE][SPIN_TYPE_SIZE];

    PlacementArray()
    {
        std::memset(data, 0, sizeof(data));
    }

    bool markIfNew(int x, int y, int orientation, SpinType spin_type)
    {
        bool& seen = data[xIndex(x)][yIndex(y)][orientationIndex(orientation)][spinTypeIndex(spin_type)];
        if (seen) {
            return false;
        }
        seen = true;
        return true;
    }

    void clear(int x, int y, int orientation, SpinType spin_type)
    {
        data[xIndex(x)][yIndex(y)][orientationIndex(orientation)][spinTypeIndex(spin_type)] = false;
    }
};

inline State materializeState(const State& base_state, const SearchState& structural_state)
{
    State state = base_state;
    state.x = structural_state.x;
    state.y = structural_state.y;
    state.orientation = static_cast<std::uint8_t>(structural_state.orientation);
    state.was_last_rotation = structural_state.was_last_rotation;
    state.srs_index = structural_state.srs_index;
    return state;
}

inline bool canPlacePiece(const State& base_state, int x, int y, int orientation)
{
    const auto& piece = ops::getPiece(base_state.current, static_cast<std::uint8_t>(orientation));
    return ops::canPlacePiece(base_state.board, piece, x, y);
}

inline bool moveHorizontally(SearchState& state, const State& base_state, int dx)
{
    const int next_x = state.x + dx;
    if (!canPlacePiece(base_state, next_x, state.y, state.orientation)) {
        return false;
    }
    state.x = static_cast<std::int8_t>(next_x);
    state.was_last_rotation = false;
    state.srs_index = -1;
    return true;
}

inline bool moveHorizontallyToWall(SearchState& state, const State& base_state, int dx)
{
    const auto& piece = ops::getPiece(base_state.current, static_cast<std::uint8_t>(state.orientation));
    int x = state.x;
    bool moved = false;
    while (ops::canPlacePiece(base_state.board, piece, x + dx, state.y)) {
        x += dx;
        moved = true;
    }
    if (!moved) {
        return false;
    }
    state.x = static_cast<std::int8_t>(x);
    state.was_last_rotation = false;
    state.srs_index = -1;
    return true;
}

inline bool moveDown(SearchState& state, const State& base_state)
{
    const int next_y = state.y + 1;
    if (!canPlacePiece(base_state, state.x, next_y, state.orientation)) {
        return false;
    }
    state.y = static_cast<std::int8_t>(next_y);
    state.was_last_rotation = false;
    state.srs_index = -1;
    return true;
}

inline bool moveDownToFloor(SearchState& state, const State& base_state)
{
    const auto& piece = ops::getPiece(base_state.current, static_cast<std::uint8_t>(state.orientation));
    int y = state.y;
    bool moved = false;
    while (ops::canPlacePiece(base_state.board, piece, state.x, y + 1)) {
        ++y;
        moved = true;
    }
    if (!moved) {
        return false;
    }
    state.y = static_cast<std::int8_t>(y);
    state.was_last_rotation = false;
    state.srs_index = -1;
    return true;
}

inline bool rotate(SearchState& state, const State& base_state, Rotation rotation)
{
    constexpr std::uint8_t orientation_delta[] = {1, 3, 2};
    const auto rotation_index = static_cast<std::underlying_type_t<Rotation>>(rotation);
    const int next_orientation = (state.orientation + orientation_delta[rotation_index]) % ORIENTATION_SIZE;
    const auto& piece = ops::getPiece(base_state.current, static_cast<std::uint8_t>(next_orientation));
    const auto& [kicks, length] = srs_table[static_cast<std::underlying_type_t<PieceType>>(base_state.current)][state.orientation][rotation_index];

    for (int i = 0; i < length; ++i) {
        const int next_x = state.x + kicks[i].x;
        const int next_y = state.y - kicks[i].y;
        if (ops::canPlacePiece(base_state.board, piece, next_x, next_y)) {
            state.x = static_cast<std::int8_t>(next_x);
            state.y = static_cast<std::int8_t>(next_y);
            state.orientation = static_cast<std::int8_t>(next_orientation);
            state.was_last_rotation = true;
            state.srs_index = static_cast<std::int8_t>(i);
            return true;
        }
    }
    return false;
}

std::vector<PlacementAction> buildPath(const VisitedArray& visited, SearchState state)
{
    std::vector<PlacementAction> path;
    while (true) {
        PlacementAction action;
        SearchState parent{};
        visited.getParent(state, action, parent);
        if (action == PlacementAction::NONE) {
            break;
        }
        path.push_back(action);
        state = parent;
    }
    std::reverse(path.begin(), path.end());
    return path;
}

// } // namespace

std::vector<PlacementSearchResult> findPlacements(const State& initial_state)
{
    std::vector<PlacementSearchResult> results;

    if (initial_state.current == PieceType::NONE || !initial_state.is_alive) {
        return results;
    }

    // Thread-local scratch reused across calls. On hot paths (MCTS self-play)
    // this function is called thousands of times per second per thread; a
    // per-call ~1.5MB allocation + ~920KB memset was serializing threads on
    // the malloc arena. Instead we pay O(reachable states) per call by
    // clearing only the slots we actually touched.
    thread_local VisitedArray visited;
    thread_local PlacementArray seen_placements;
    thread_local std::vector<SearchState> queue;

    queue.clear();
    results.reserve(256);

    // Drop the BFS root by dy_safe rows. Piece bbox is 4 tall; we want the
    // bottom (y+3) to stay strictly above any occupied cell, so target_y
    // ensures y+3 < min_stack_y. dy_safe == 0 when the stack is already close
    // to spawn, which preserves the original behaviour. Verified against the
    // unoptimised BFS over 23k random-walk states: 0 mismatches.
    const int min_stack_y = findHighestStackY(initial_state.board);
    const int target_y = min_stack_y - 4;
    const int dy_safe = std::max(0, target_y - static_cast<int>(initial_state.y));

    SearchState root{
        initial_state.x,
        static_cast<std::int8_t>(initial_state.y + dy_safe),
        static_cast<std::int8_t>(initial_state.orientation),
        // dy_safe > 0 means we synthesise soft-drops before the real BFS path;
        // soft-drop resets was_last_rotation / srs_index (see moveDown).
        (dy_safe > 0) ? false : (initial_state.was_last_rotation != 0),
        initial_state.srs_index,
    };
    root = canonicalize(root);

    visited.set(root, PlacementAction::NONE, SearchState{});
    queue.push_back(root);

    for (std::size_t head = 0; head < queue.size(); ++head) {
        const SearchState current = queue[head];

        SearchState lock_state = current;
        moveDownToFloor(lock_state, initial_state);

        State final_state = materializeState(initial_state, current);
        hardDrop(&final_state);

        if (seen_placements.markIfNew(lock_state.x, lock_state.y, lock_state.orientation, final_state.spin_type)) {
            PlacementSearchResult result;
            result.lock_x = lock_state.x;
            result.lock_y = lock_state.y;
            result.orientation = static_cast<std::uint8_t>(lock_state.orientation);
            result.spin_type = final_state.spin_type;
            result.final_state = final_state;
            // Prepend dy_safe soft-drops so the engine replay reproduces the
            // exact piece position the BFS started from. Each SOFT_DROP here
            // is guaranteed to succeed in the engine because we chose dy_safe
            // such that the piece bbox never enters an occupied row.
            auto bfs_path = buildPath(visited, current);
            result.path.reserve(dy_safe + bfs_path.size());
            result.path.assign(dy_safe, PlacementAction::SOFT_DROP);
            result.path.insert(result.path.end(), bfs_path.begin(), bfs_path.end());
            results.push_back(std::move(result));
        }

        const auto tryEnqueue = [&](SearchState next_state, PlacementAction action) {
            next_state = canonicalize(next_state);
            if (!visited.get(next_state)) {
                visited.set(next_state, action, current);
                queue.push_back(next_state);
            }
        };

        {
            SearchState next = current;
            if (moveHorizontally(next, initial_state, -1)) { tryEnqueue(next, PlacementAction::LEFT); }
        }
        {
            SearchState next = current;
            if (moveHorizontally(next, initial_state, 1)) { tryEnqueue(next, PlacementAction::RIGHT); }
        }
        {
            SearchState next = current;
            if (moveHorizontallyToWall(next, initial_state, -1)) { tryEnqueue(next, PlacementAction::LEFT_WALL); }
        }
        {
            SearchState next = current;
            if (moveHorizontallyToWall(next, initial_state, 1)) { tryEnqueue(next, PlacementAction::RIGHT_WALL); }
        }
        {
            SearchState next = current;
            if (moveDown(next, initial_state)) { tryEnqueue(next, PlacementAction::SOFT_DROP); }
        }
        {
            SearchState next = current;
            if (moveDownToFloor(next, initial_state)) { tryEnqueue(next, PlacementAction::SOFT_DROP_FLOOR); }
        }
        {
            SearchState next = current;
            if (rotate(next, initial_state, Rotation::CW)) { tryEnqueue(next, PlacementAction::ROTATE_CW); }
        }
        {
            SearchState next = current;
            if (rotate(next, initial_state, Rotation::CCW)) { tryEnqueue(next, PlacementAction::ROTATE_CCW); }
        }
        {
            SearchState next = current;
            if (rotate(next, initial_state, Rotation::HALF)) { tryEnqueue(next, PlacementAction::ROTATE_180); }
        }
    }

    // Undo only the slots we touched so the scratch stays clean for the next
    // call without memset'ing the full arrays. queue contains every visited
    // state; results contains every seen_placements slot that was set true
    // (duplicate lock-slot attempts return false in markIfNew, and the
    // original setter is already accounted for in results).
    for (const auto& s : queue) {
        visited.clear(s);
    }
    for (const auto& r : results) {
        seen_placements.clear(r.lock_x, r.lock_y, r.orientation, r.spin_type);
    }
    queue.clear();

    return results;
}

} // namespace minizero::env::moderntetris::engine
