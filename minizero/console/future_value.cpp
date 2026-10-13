#include "future_value.h"
#include <iostream>

#if MODERNTETRIS_PLACEMENT
#include "configuration.h"
#include "create_actor.h"
#include "create_network.h"
#include "moderntetris_placement.h"
#include "placement_transformer_network.h"
#include "random.h"
#include "zero_actor.h"
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <torch/cuda.h>
#include <vector>
#endif

namespace minizero::console {

#if MODERNTETRIS_PLACEMENT

using Env = env::moderntetris_placement::ModernTetrisPlacementEnv;
using EnvLoader = env::moderntetris_placement::ModernTetrisPlacementEnvLoader;

namespace {

// Parameters come from the environment so that no config key is added (a new
// key would make older binaries reject every cfg generated afterwards):
//   FV_SGF      game records (self-play or eval .sgf, one game per line)   required
//   FV_OUT      output TSV                                                 required
//   FV_K        hidden futures drawn per position                          32
//   FV_GAMES    games used from FV_SGF                                     64
//   FV_STRIDE   one position every FV_STRIDE moves (from FV_STRIDE/2)      20
//   FV_TAIL     skip positions this close to the end of a record           12
//   FV_WAVE     searches run together (one network batch per step)        1024
//   FV_THREADS  CPU threads for the searches                               12
//   FV_SEED     base seed of the futures                                   1
// The search runs with the training cfg plus -conf_str; pass at least
// nn_file_name=<pt>:actor_mcts_resample_hidden_future=false so the search sees
// the drawn future, and actor_use_gumbel_noise=false so the futures are the only
// source of spread.
std::string envString(const char* name, const std::string& fallback)
{
    const char* v = std::getenv(name);
    return (v && *v) ? std::string(v) : fallback;
}

int envInt(const char* name, int fallback) { return std::stoi(envString(name, std::to_string(fallback))); }

// Runs fn(i) for i in [0, n) on `threads` threads; each worker gets its own Random seed.
void parallelFor(int n, int threads, const std::function<void(int)>& fn)
{
    static std::atomic<unsigned> seed_counter{std::random_device{}()};
    threads = std::max(1, std::min(threads, n));
    std::vector<std::thread> workers;
    for (int t = 0; t < threads; ++t) {
        const unsigned seed = seed_counter.fetch_add(0x9e3779b9U);
        workers.emplace_back([=, &fn] {
            utils::Random::seed(static_cast<int>(seed));
            for (int i = t; i < n; i += threads) { fn(i); }
        });
    }
    for (auto& w : workers) { w.join(); }
}

struct Position {
    int game = 0;
    int move = 0;
    float recorded_value = 0.0f; // V[] of the record: the value of the search that was played
    Env env;
};

struct Job {
    int position = 0;
    int sample = 0; // 0: the record's real future; 1..K: drawn futures
    Env env;
    float root_value = 0.0f;
    float nn_value = 0.0f;
};

std::vector<Position> loadPositions(const std::string& sgf, int games, int stride, int tail)
{
    std::vector<Position> positions;
    std::ifstream in(sgf);
    std::string line;
    int game = 0;
    while (game < games && std::getline(in, line)) {
        if (line.rfind("(;GM", 0) != 0) { continue; }
        EnvLoader loader;
        if (!loader.loadFromString(line)) { continue; }
        const auto& actions = loader.getActionPairs();
        Env env;
        env.reset(loader.getSeed());
        for (int move = 0; move + tail < static_cast<int>(actions.size()); ++move) {
            if (move >= stride / 2 && (move - stride / 2) % stride == 0 && !env.isTerminal()) {
                Position p;
                p.game = game;
                p.move = move;
                const auto it = actions[move].second.find("V");
                p.recorded_value = it == actions[move].second.end() ? 0.0f : std::stof(it->second);
                p.env = env;
                p.env.shrinkPlacementCache();
                positions.push_back(std::move(p));
            }
            env.act(actions[move].first);
        }
        ++game;
    }
    return positions;
}

} // namespace

void FutureValue::run()
{
    const std::string sgf = envString("FV_SGF", "");
    const std::string out_file = envString("FV_OUT", "");
    if (sgf.empty() || out_file.empty()) {
        std::cerr << "future_value: set FV_SGF and FV_OUT (see future_value.cpp)" << std::endl;
        return;
    }
    const int samples = envInt("FV_K", 32);
    const int wave_size = envInt("FV_WAVE", 1024);
    const int threads = envInt("FV_THREADS", 12);
    const int base_seed = envInt("FV_SEED", 1);
    if (config::actor_mcts_resample_hidden_future) {
        std::cerr << "future_value: actor_mcts_resample_hidden_future is on, so every search would draw its own future; "
                     "pass actor_mcts_resample_hidden_future=false"
                  << std::endl;
    }

    std::vector<Position> positions = loadPositions(sgf, envInt("FV_GAMES", 64), envInt("FV_STRIDE", 20), envInt("FV_TAIL", 12));
    std::cerr << "future_value: " << positions.size() << " positions from " << sgf << ", " << samples << " futures each" << std::endl;

    // the jobs: per position the real future, then K drawn ones (drawn here, on one thread, with fixed seeds)
    std::vector<Job> jobs;
    jobs.reserve(positions.size() * (samples + 1));
    for (int p = 0; p < static_cast<int>(positions.size()); ++p) {
        for (int k = 0; k <= samples; ++k) {
            Job job;
            job.position = p;
            job.sample = k;
            job.env = positions[p].env;
            if (k > 0) {
                utils::Random::seed(base_seed * 1000003 + p * 1009 + k);
                job.env.resampleHiddenFuture(true, true, true);
            }
            jobs.push_back(std::move(job));
        }
    }

    const int gpu = torch::cuda::device_count() > 0 ? 0 : -1;
    auto net = std::static_pointer_cast<network::PlacementTransformerNetwork>(network::createNetwork(config::nn_file_name, gpu));
    const uint64_t tree_size = static_cast<uint64_t>(config::actor_num_simulation + 1) *
                               (net->getActionSize() + std::max(0, config::actor_mcts_chance_worlds) + (config::actor_mcts_chance_nodes ? actor::kMaxChanceChildren : 0));
    std::vector<std::shared_ptr<actor::BaseActor>> actors;
    for (int i = 0; i < std::min<int>(wave_size, jobs.size()); ++i) { actors.push_back(actor::createActor(tree_size, net)); }

    for (std::size_t start = 0; start < jobs.size(); start += actors.size()) {
        const int n = static_cast<int>(std::min(actors.size(), jobs.size() - start));
        for (int i = 0; i < n; ++i) { actors[i]->getEnvironment() = jobs[start + i].env; }
        parallelFor(n, threads, [&](int i) { actors[i]->resetSearch(); });
        while (true) {
            std::vector<int> pending;
            for (int i = 0; i < n; ++i) {
                if (!actors[i]->isSearchDone()) { pending.push_back(i); }
            }
            if (pending.empty()) { break; }
            parallelFor(static_cast<int>(pending.size()), threads, [&](int k) { actors[pending[k]]->beforeNNEvaluation(); });
            const auto outputs = net->forward();
            parallelFor(static_cast<int>(pending.size()), threads, [&](int k) {
                auto& a = actors[pending[k]];
                a->afterNNEvaluation(outputs[a->getNNEvaluationBatchIndex()]);
            });
        }
        for (int i = 0; i < n; ++i) {
            auto mcts = std::static_pointer_cast<actor::ZeroActor>(actors[i])->getMCTS();
            jobs[start + i].root_value = mcts->getRootNode()->getMean();
            jobs[start + i].nn_value = mcts->getRootNode()->getValue();
            jobs[start + i].env = Env(); // free the copy
        }
        std::cerr << "future_value: " << start + n << "/" << jobs.size() << " searches" << std::endl;
    }

    std::ofstream out(out_file);
    out << "game\tmove\trecorded_value\tnn_value\treal_future_value\tdrawn_future_values\n";
    for (int p = 0; p < static_cast<int>(positions.size()); ++p) {
        const Job* row = &jobs[p * (samples + 1)];
        out << positions[p].game << "\t" << positions[p].move << "\t" << positions[p].recorded_value << "\t" << row[0].nn_value << "\t" << row[0].root_value << "\t";
        for (int k = 1; k <= samples; ++k) { out << (k > 1 ? "," : "") << row[k].root_value; }
        out << "\n";
    }
    std::cerr << "future_value: wrote " << out_file << std::endl;
}

#else

void FutureValue::run()
{
    std::cerr << "future_value mode is only built for moderntetris_placement" << std::endl;
}

#endif

} // namespace minizero::console
