// Checks getChanceOutcomes()/actWithOutcome(): the outcomes' probabilities sum
// to 1, acting with an outcome reveals exactly its pieces, garbage holes and
// arriving garbage, distinct outcomes lead to distinct observations, and the
// outcome frequencies of the env's own random play match the probabilities.
#include "configuration.h"
#include "moderntetris_placement.h"
#include "random.h"
#include <cmath>
#include <iostream>
#include <map>
#include <set>
#include <tuple>
using namespace minizero;
using namespace minizero::env::moderntetris_placement;
using minizero::env::moderntetris_placement::engine::PieceType;

static int pendingGarbage(const engine::State& s)
{
    int total = 0;
    for (int i = 0; i < engine::GARBAGE_QUEUE_SIZE; ++i) total += s.garbage_queue[i];
    return total;
}

int main()
{
    utils::Random::seed(7);
    config::env_modern_tetris_garbage_probability = 0.3f;
    config::env_modern_tetris_garbage_delay = 2; // garbage lands often
    const int preview = config::env_modern_tetris_num_preview_piece;
    int fails = 0, checked = 0, with_holes = 0, with_two_pieces = 0;
    double max_tv = 0;
    for (int game = 0; game < 60; ++game) {
        ModernTetrisPlacementEnv env;
        env.reset(500 + game);
        for (int step = 0; step < 80 && !env.isTerminal(); ++step) {
            auto legal = env.getLegalActions();
            const auto action = legal[utils::Random::randInt() % legal.size()];
            const auto outcomes = env.getChanceOutcomes(action);
            double sum = 0;
            for (const auto& o : outcomes) sum += o.probability;
            if (outcomes.empty() || std::abs(sum - 1.0) > 1e-4) {
                ++fails;
                std::cout << "FAIL sum " << sum << " (" << outcomes.size() << " outcomes)\n";
            }
            std::set<std::uint64_t> keys;
            for (const auto& o : outcomes) {
                ModernTetrisPlacementEnv copy = env;
                const int pending_before = pendingGarbage(copy.getEngineState());
                if (!copy.actWithOutcome(action, o)) {
                    ++fails;
                    continue;
                }
                const auto& s = copy.getEngineState();
                bool ok = true;
                // revealed pieces sit at the end of the preview window
                for (int t = 0; t < o.num_pieces; ++t) ok &= s.next[preview - o.num_pieces + t] == o.pieces[t];
                if (s.is_alive && o.garbage_lines > 0) ok &= pendingGarbage(s) >= o.garbage_lines;
                if (o.num_holes == 0 && s.is_alive) ok &= pendingGarbage(s) - o.garbage_lines <= pending_before;
                if (!ok) {
                    ++fails;
                    if (fails < 10) std::cout << "FAIL outcome mismatch game " << game << " step " << step << "\n";
                }
                keys.insert(copy.getObservationKey());
                with_holes += o.num_holes > 0;
                with_two_pieces += o.num_pieces == 2;
                ++checked;
            }
            if (keys.size() != outcomes.size()) {
                ++fails;
                std::cout << "FAIL " << outcomes.size() << " outcomes but " << keys.size() << " distinct observations\n";
            }
            // frequencies of the env's own random play vs the listed probabilities
            // (only where no garbage lands: resampleHiddenFuture keeps this lock's hole draws)
            if (step % 10 == 5 && outcomes[0].num_holes == 0) {
                std::map<std::uint64_t, double> expected, seen;
                for (const auto& o : outcomes) {
                    ModernTetrisPlacementEnv copy = env;
                    copy.actWithOutcome(action, o);
                    expected[copy.getObservationKey()] += o.probability;
                }
                const int n = 4000;
                for (int i = 0; i < n; ++i) {
                    ModernTetrisPlacementEnv copy = env;
                    copy.resampleHiddenFuture(true, true, false);
                    copy.act(action);
                    seen[copy.getObservationKey()] += 1.0 / n;
                }
                double tv = 0;
                for (auto& [k, p] : expected) tv += std::abs(p - seen[k]);
                for (auto& [k, p] : seen)
                    if (!expected.count(k)) tv += p;
                max_tv = std::max(max_tv, tv / 2);
            }
            env.act(action);
        }
    }
    std::cout << "checked " << checked << " outcomes (" << with_holes << " with landing garbage, " << with_two_pieces
              << " revealing two pieces), max total-variation vs random play " << max_tv << ", fails " << fails << "\n";
    return fails == 0 ? 0 : 1;
}
