// Checks nn_placement_afterstate_before_garbage: with it on, the action
// descriptors (incl. afterstate) must not depend on garbage_seed, and the legal
// action set must be the same as with it off.
#include "configuration.h"
#include "moderntetris_placement.h"
#include "random.h"
#include <iostream>
using namespace minizero;
using namespace minizero::env::moderntetris_placement;

static bool sameDescs(const std::vector<PlacementActionDescriptor>& a, const std::vector<PlacementActionDescriptor>& b)
{
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].action_id != b[i].action_id || a[i].lines_cleared != b[i].lines_cleared || a[i].afterstate != b[i].afterstate) return false;
    }
    return true;
}

int main()
{
    utils::Random::seed(7);
    config::nn_placement_use_afterstate_feature = true;
    config::env_modern_tetris_garbage_probability = 0.3f;
    int with_garbage = 0, leak_off = 0, leak_on = 0, action_mismatch = 0;
    for (int game = 0; game < 200; ++game) {
        ModernTetrisPlacementEnv env;
        env.reset(5000 + game);
        for (int step = 0; step < 80 && !env.isTerminal(); ++step) {
            const auto& s = env.getEngineState();
            int pending = 0;
            for (int i = 0; i < engine::GARBAGE_QUEUE_SIZE; ++i) pending += s.garbage_queue[i];
            if (pending > 0) {
                ++with_garbage;
                engine::step::Context other_ctx = env.getEngineContext();
                other_ctx.state.garbage_seed ^= 0x5bd1e995U;
                for (bool on : {false, true}) {
                    config::nn_placement_afterstate_before_garbage = on;
                    ModernTetrisPlacementEnv a = env;
                    a.shrinkPlacementCache();
                    ModernTetrisPlacementEnv b;
                    b.setState(other_ctx);
                    const bool same = sameDescs(a.getActionDescriptors(), b.getActionDescriptors());
                    if (!same) (on ? leak_on : leak_off)++;
                }
                config::nn_placement_afterstate_before_garbage = false;
                ModernTetrisPlacementEnv off = env;
                off.shrinkPlacementCache();
                auto la = off.getLegalActions();
                config::nn_placement_afterstate_before_garbage = true;
                ModernTetrisPlacementEnv on = env;
                on.shrinkPlacementCache();
                auto lb = on.getLegalActions();
                bool same_actions = la.size() == lb.size();
                for (size_t i = 0; same_actions && i < la.size(); ++i) same_actions = la[i].getActionID() == lb[i].getActionID();
                if (!same_actions) ++action_mismatch;
            }
            config::nn_placement_afterstate_before_garbage = false;
            env.shrinkPlacementCache();
            auto legal = env.getLegalActions();
            env.act(legal[utils::Random::randInt() % legal.size()]);
        }
    }
    std::cout << "states with pending garbage: " << with_garbage
              << "\n  descriptors depend on garbage_seed -- option off: " << leak_off << ", option on: " << leak_on
              << "\n  legal action set differs off vs on: " << action_mismatch << "\n";
    return (leak_on != 0 || action_mismatch != 0) ? 1 : 0;
}
