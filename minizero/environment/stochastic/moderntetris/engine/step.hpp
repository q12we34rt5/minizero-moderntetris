#pragma once
#include "tetris.hpp"
#include <cstdint>

#include <cassert>

namespace minizero::env::moderntetris::engine::step {

enum class Action : std::uint8_t {
    MOVE_LEFT,
    MOVE_RIGHT,
    SOFT_DROP,
    HARD_DROP,
    ROTATE_CW,
    ROTATE_CCW,
    ROTATE_180,
    HOLD,
    // extended
    MOVE_LEFT_TO_WALL,
    MOVE_RIGHT_TO_WALL,
    SOFT_DROP_TO_FLOOR,
    // no-op
    NOOP,
    // sentinel
    SIZE
};

struct Info {
    Action action_id;                         // which action the agent chose
    std::uint8_t /* bool */ action_success;   // whether the action executed successfully
    std::uint8_t /* bool */ forced_hard_drop; // whether lifetime expired and a hard drop was forced
};

struct Config {
    std::int32_t piece_life = 20;          // steps before forced hard drop
    std::uint8_t /* bool */ auto_drop = 1; // simulate gravity each step (bool)
};

struct Context {
    State state;
    std::int32_t lifetime; // remaining steps for current piece
    Config config;
};

inline void setConfig(Context* ctx, const Config& config)
{
    ctx->config = config;
}

inline void setSeed(Context* ctx, std::uint32_t seed, std::uint32_t garbage_seed)
{
    setSeed(&ctx->state, seed, garbage_seed);
}

inline void reset(Context* ctx)
{
    reset(&ctx->state);
    ctx->lifetime = ctx->config.piece_life;
}

inline Info step(Context* ctx, Action action)
{
    Info info{
        .action_id = action,
        .action_success = false,
        .forced_hard_drop = false};
    // game already over
    if (!ctx->state.is_alive) {
        return info;
    }
    // perform action
    bool success = false;
    bool lifetime_reset = false;
    switch (action) {
        case Action::MOVE_LEFT:
            success = moveLeft(&ctx->state);
            break;
        case Action::MOVE_RIGHT:
            success = moveRight(&ctx->state);
            break;
        case Action::MOVE_LEFT_TO_WALL:
            success = moveLeftToWall(&ctx->state);
            break;
        case Action::MOVE_RIGHT_TO_WALL:
            success = moveRightToWall(&ctx->state);
            break;
        case Action::SOFT_DROP:
            success = softDrop(&ctx->state);
            break;
        case Action::SOFT_DROP_TO_FLOOR:
            success = softDropToFloor(&ctx->state);
            break;
        case Action::HARD_DROP:
            success = hardDrop(&ctx->state);
            lifetime_reset = true;
            break;
        case Action::ROTATE_CW:
            success = rotateClockwise(&ctx->state);
            break;
        case Action::ROTATE_CCW:
            success = rotateCounterclockwise(&ctx->state);
            break;
        case Action::ROTATE_180:
            success = rotate180(&ctx->state);
            break;
        case Action::HOLD:
            success = hold(&ctx->state);
            if (success) { lifetime_reset = true; }
            break;
        case Action::NOOP:
            success = noop(&ctx->state);
            break;
        default:
            assert(false && "Invalid action");
            break;
    }
    info.action_success = success ? 1 : 0;
    // simulate gravity
    if (ctx->config.auto_drop && action != Action::HARD_DROP && ctx->state.is_alive) {
        softDrop(&ctx->state);
    }
    // lifetime management
    if (lifetime_reset) {
        ctx->lifetime = ctx->config.piece_life;
    } else {
        ctx->lifetime--;
        if (ctx->lifetime <= 0 && ctx->state.is_alive) {
            ctx->lifetime = ctx->config.piece_life;
            // force hard drop when lifetime expires
            hardDrop(&ctx->state);
            info.forced_hard_drop = true;
        }
    }
    return info;
}

} // namespace minizero::env::moderntetris::engine::step
