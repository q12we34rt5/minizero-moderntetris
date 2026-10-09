#pragma once

namespace minizero::console {

// Versus matches between two players on the moderntetris placement env: each
// player is a trained model (its own cfg, weights and search) or an external bot
// process speaking the fusion_stdio JSON protocol. Both boards advance one piece
// per tick; after both lock, each side's lines sent (after countering its own
// queue) are queued on the other board. The first to top out loses; at the step
// cap the side that sent more garbage wins. See arena.cpp for the spec file.
class Arena {
public:
    void run();
};

} // namespace minizero::console
