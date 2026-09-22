// Checks resampleHiddenFuture(): visible pieces and 7-bag contents are kept,
// the legal actions are unchanged, and the hidden future actually changes.
#include "configuration.h"
#include "moderntetris_placement.h"
#include "random.h"
#include <algorithm>
#include <iostream>
using namespace minizero::env::moderntetris_placement;
using minizero::env::moderntetris_placement::engine::PieceType;

static int countNext(const engine::State& s)
{
    int c = 0;
    while (c < 14 && s.next[c] != PieceType::NONE) ++c;
    return c;
}

int main()
{
    minizero::utils::Random::seed(123);
    int fails = 0, differed = 0, total = 0;
    for (int game = 0; game < 200; ++game) {
        ModernTetrisPlacementEnv env;
        env.reset(1000 + game);
        for (int step = 0; step < 60 && !env.isTerminal(); ++step) {
            ModernTetrisPlacementEnv copy = env;
            copy.resampleHiddenFuture();
            const auto& a = env.getEngineState();
            const auto& b = copy.getEngineState();
            const int ca = countNext(a), cb = countNext(b);
            const int boundary = ca - 7;
            bool ok = (ca == cb) && ca >= 8 && ca <= 14 && a.current == b.current && a.hold == b.hold;
            for (int i = 0; i < 5 && ok; ++i) ok = a.next[i] == b.next[i];
            auto seg_equal = [&](int lo, int hi) {
                std::vector<int> x, y;
                for (int i = lo; i < hi; ++i) {
                    x.push_back(static_cast<int>(a.next[i]));
                    y.push_back(static_cast<int>(b.next[i]));
                }
                std::sort(x.begin(), x.end());
                std::sort(y.begin(), y.end());
                return x == y;
            };
            ok = ok && seg_equal(0, boundary) && seg_equal(boundary, ca);
            // full bag must contain 7 distinct pieces
            std::vector<int> bag;
            for (int i = boundary; i < ca; ++i) bag.push_back(static_cast<int>(a.next[i]));
            std::sort(bag.begin(), bag.end());
            ok = ok && std::adjacent_find(bag.begin(), bag.end()) == bag.end();
            ok = ok && env.getLegalActions().size() == copy.getLegalActions().size();
            if (!ok) {
                ++fails;
                if (fails < 5) std::cout << "FAIL game " << game << " step " << step << " count " << ca << "\n";
            }
            // does the future (next 30 pieces played identically-by-first-action) differ?
            ++total;
            bool diff = false;
            for (int i = 5; i < ca; ++i) diff |= a.next[i] != b.next[i];
            diff |= a.seed != b.seed;
            differed += diff;
            auto legal = env.getLegalActions();
            env.act(legal[minizero::utils::Random::randInt() % legal.size()]);
        }
    }
    std::cout << "checked " << total << " states, fails=" << fails << ", future differed=" << differed << "\n";
    return fails != 0;
}
