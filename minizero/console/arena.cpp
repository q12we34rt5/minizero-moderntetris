#include "arena.h"
#include "configuration.h"
#include <iostream>

#if MODERNTETRIS_PLACEMENT
#include "create_actor.h"
#include "create_network.h"
#include "moderntetris_placement.h"
#include "placement_transformer_network.h"
#include "random.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>
#endif

namespace minizero::console {

#if MODERNTETRIS_PLACEMENT

// Spec file (config key arena_spec), one key=value per line, '#' starts a comment:
//   games=128          matches to play (match k uses piece/garbage seed seed*1000003+k
//                      on both boards, so both players get the same pieces)
//   parallel=64        matches in flight at once
//   max_steps=500      pieces per player before the match is decided on garbage sent
//   garbage_delay=4    locks before received garbage lands
//   threads=16         CPU threads for the search bookkeeping
//   seed=1
//   out=experiments/arena/<name>    results.jsonl and summary.txt go here
//   a.name=...  a.type=model|external
//   a.cfg=<training cfg>  a.model=<weights .pt>  a.conf=<extra conf_str>  a.gpu=0   (model)
//   a.cmd=<shell command>  a.procs=8                                                (external)
//   b.* the same for the second player
// Every model player searches with its own cfg (reward, simulations, ...); the
// arena forces random garbage off, the step cap, the honest search and the fixed
// terminal value on top of it.

using namespace minizero::env::moderntetris_placement;

struct ArenaPlayerSpec {
    std::string name, type = "model", cfg, model, conf, cmd;
    int gpu = 0;
    int procs = 8;
};

struct ArenaSpec {
    int games = 128, parallel = 64, max_steps = 500, garbage_delay = 4, threads = 16;
    unsigned seed = 1;
    std::string out = "arena-out";
    ArenaPlayerSpec player[2];
};

static ArenaSpec parseSpec(const std::string& path)
{
    std::ifstream fin(path);
    if (!fin) { throw std::runtime_error{"arena: cannot open spec " + path}; }
    ArenaSpec spec;
    std::string line;
    while (std::getline(fin, line)) {
        line = line.substr(0, line.find('#'));
        const std::size_t eq = line.find('=');
        if (eq == std::string::npos) { continue; }
        auto trim = [](std::string s) {
            s.erase(0, s.find_first_not_of(" \t"));
            s.erase(s.find_last_not_of(" \t\r") + 1);
            return s;
        };
        const std::string key = trim(line.substr(0, eq)), value = trim(line.substr(eq + 1));
        if (key == "games") {
            spec.games = std::stoi(value);
        } else if (key == "parallel") {
            spec.parallel = std::stoi(value);
        } else if (key == "max_steps") {
            spec.max_steps = std::stoi(value);
        } else if (key == "garbage_delay") {
            spec.garbage_delay = std::stoi(value);
        } else if (key == "threads") {
            spec.threads = std::stoi(value);
        } else if (key == "seed") {
            spec.seed = static_cast<unsigned>(std::stoul(value));
        } else if (key == "out") {
            spec.out = value;
        } else if (key.size() > 2 && (key[0] == 'a' || key[0] == 'b') && key[1] == '.') {
            ArenaPlayerSpec& p = spec.player[key[0] == 'a' ? 0 : 1];
            const std::string field = key.substr(2);
            if (field == "name") {
                p.name = value;
            } else if (field == "type") {
                p.type = value;
            } else if (field == "cfg") {
                p.cfg = value;
            } else if (field == "model") {
                p.model = value;
            } else if (field == "conf") {
                p.conf = value;
            } else if (field == "gpu") {
                p.gpu = std::stoi(value);
            } else if (field == "cmd") {
                p.cmd = value;
            } else if (field == "procs") {
                p.procs = std::stoi(value);
            } else {
                throw std::runtime_error{"arena: unknown player key " + key};
            }
        } else {
            throw std::runtime_error{"arena: unknown key " + key};
        }
    }
    for (ArenaPlayerSpec& p : spec.player) {
        if (p.type != "model" && p.type != "external") { throw std::runtime_error{"arena: player type must be model or external"}; }
        if (p.type == "model" && (p.cfg.empty() || p.model.empty())) { throw std::runtime_error{"arena: a model player needs cfg and model"}; }
        if (p.type == "external" && p.cmd.empty()) { throw std::runtime_error{"arena: an external player needs cmd"}; }
        if (p.name.empty()) { p.name = p.type == "model" ? p.model : p.cmd; }
    }
    if (spec.player[0].type != "model" && spec.player[1].type != "model") { throw std::runtime_error{"arena: at least one player must be a model (its cfg sets the rules)"}; }
    return spec;
}

// Runs fn(i) for i in [0, n) on `threads` threads. Random is thread_local and a
// fresh thread starts from the same default seed, so every worker is reseeded.
static void parallelFor(int n, int threads, const std::function<void(int)>& fn)
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

// A bot process speaking one JSON line in, one JSON line out (fusion_stdio).
class ArenaExternalBot {
public:
    explicit ArenaExternalBot(const std::string& cmd)
    {
        int to_child[2], from_child[2];
        if (pipe(to_child) != 0 || pipe(from_child) != 0) { throw std::runtime_error{"arena: pipe failed"}; }
        pid_ = fork();
        if (pid_ == 0) {
            dup2(to_child[0], 0);
            dup2(from_child[1], 1);
            close(to_child[1]);
            close(from_child[0]);
            execl("/bin/sh", "sh", "-c", cmd.c_str(), static_cast<char*>(nullptr));
            _exit(127);
        }
        close(to_child[0]);
        close(from_child[1]);
        out_ = fdopen(to_child[1], "w");
        in_ = fdopen(from_child[0], "r");
    }
    ~ArenaExternalBot()
    {
        if (out_) { fclose(out_); }
        if (in_) { fclose(in_); }
        if (pid_ > 0) { kill(pid_, SIGTERM); }
    }
    std::string request(const std::string& line)
    {
        std::fputs((line + "\n").c_str(), out_);
        std::fflush(out_);
        std::string reply;
        int c;
        while ((c = std::fgetc(in_)) != EOF && c != '\n') { reply.push_back(static_cast<char>(c)); }
        return reply;
    }

private:
    pid_t pid_ = -1;
    FILE* out_ = nullptr;
    FILE* in_ = nullptr;
};

static char pieceLetter(engine::PieceType t)
{
    switch (t) {
        case engine::PieceType::Z: return 'Z';
        case engine::PieceType::L: return 'L';
        case engine::PieceType::O: return 'O';
        case engine::PieceType::S: return 'S';
        case engine::PieceType::I: return 'I';
        case engine::PieceType::J: return 'J';
        case engine::PieceType::T: return 'T';
        default: return '?';
    }
}

// The external protocol: x from the left wall, y from the bottom row, both from 0.
static std::pair<int, int> toExternal(int x, int y) { return {x - engine::BOARD_LEFT, engine::BOARD_BOTTOM - y}; }

static std::string stateJson(const Environment& env, int match_id)
{
    const engine::State& s = env.getEngineState();
    const int preview = std::clamp(config::env_modern_tetris_num_preview_piece, 0, 14);
    std::ostringstream oss;
    oss << "{\"id\":" << match_id << ",\"board\":[";
    int last = 0;
    std::vector<int> rows;
    for (int y = engine::BOARD_BOTTOM; y >= 0; --y) {
        int bits = 0;
        for (int x = engine::BOARD_LEFT; x <= engine::BOARD_RIGHT; ++x) {
            if (engine::ops::getCell(s.board, x, y) != engine::Cell::EMPTY) { bits |= 1 << (x - engine::BOARD_LEFT); }
        }
        rows.push_back(bits);
        if (bits) { last = static_cast<int>(rows.size()); }
    }
    for (int i = 0; i < last; ++i) { oss << (i ? "," : "") << rows[i]; }
    oss << "],\"current\":\"" << pieceLetter(s.current) << "\",\"hold\":";
    if (s.hold == engine::PieceType::NONE) {
        oss << "null";
    } else {
        oss << "\"" << pieceLetter(s.hold) << "\"";
    }
    oss << ",\"has_held\":" << (s.has_held ? "true" : "false") << ",\"queue\":[";
    for (int i = 0; i < preview && s.next[i] != engine::PieceType::NONE; ++i) { oss << (i ? "," : "") << "\"" << pieceLetter(s.next[i]) << "\""; }
    // ours count from -1 (no chain / no combo); the protocol counts from 0
    oss << "],\"b2b\":" << std::max(0, s.back_to_back_count + 1) << ",\"combo\":" << std::max(0, s.combo_count + 1) << ",\"pending_garbage\":[";
    bool first = true;
    for (int i = 0; i < engine::GARBAGE_QUEUE_SIZE; ++i) {
        if (s.garbage_queue[i] == 0) { continue; }
        oss << (first ? "" : ",") << "{\"rows\":" << static_cast<int>(s.garbage_queue[i]) << ",\"delay\":" << static_cast<int>(s.garbage_delay[i]) << "}";
        first = false;
    }
    oss << "]}";
    return oss.str();
}

// Board cells a placement action fills, in external coordinates.
static std::set<std::pair<int, int>> actionCells(const Environment& env, int action_id)
{
    const engine::State& s = env.getEngineState();
    const UnpackedPlacement up = unpackPlacementId(action_id);
    engine::PieceType type = s.current;
    if (up.use_hold) { type = s.hold != engine::PieceType::NONE ? s.hold : s.next[0]; }
    const auto& piece = engine::ops::getPiece(type, static_cast<std::uint8_t>(up.orientation));
    std::set<std::pair<int, int>> cells;
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            if (engine::ops::getCell(piece, c, r) != engine::Cell::EMPTY) { cells.insert(toExternal(up.lock_x + c, up.lock_y + r)); }
        }
    }
    return cells;
}

struct ArenaExternalMove {
    bool ok = false;
    Action action;
    bool spin_mismatch = false;
};

static ArenaExternalMove parseExternalMove(const Environment& env, const std::string& reply)
{
    ArenaExternalMove move;
    if (reply.find("\"error\"") != std::string::npos) { return move; }
    const std::size_t cells_at = reply.find("\"cells\"");
    if (cells_at == std::string::npos) { return move; }
    const std::size_t end = reply.find("]]", cells_at);
    const std::string cells_text = reply.substr(cells_at, end == std::string::npos ? std::string::npos : end - cells_at + 2);
    static const std::regex cell_re("\\[\\s*(-?\\d+)\\s*,\\s*(-?\\d+)\\s*\\]");
    std::set<std::pair<int, int>> want;
    for (auto it = std::sregex_iterator(cells_text.begin(), cells_text.end(), cell_re); it != std::sregex_iterator(); ++it) {
        want.insert({std::stoi((*it)[1].str()), std::stoi((*it)[2].str())});
    }
    const bool hold_used = reply.find("\"hold_used\":true") != std::string::npos;
    int spin = 0;
    if (reply.find("\"spin\":\"full\"") != std::string::npos) { spin = static_cast<int>(engine::SpinType::SPIN); }
    if (reply.find("\"spin\":\"mini\"") != std::string::npos) { spin = static_cast<int>(engine::SpinType::SPIN_MINI); }
    for (const Action& a : env.getLegalActions()) {
        const UnpackedPlacement up = unpackPlacementId(a.getActionID());
        if (up.use_hold != hold_used || actionCells(env, a.getActionID()) != want) { continue; }
        if (!move.ok || up.spin_type == spin) {
            move.ok = true;
            move.action = a;
            move.spin_mismatch = up.spin_type != spin;
        }
    }
    return move;
}

struct ArenaMatch {
    unsigned seed = 0;
    bool done = false;
    int winner = -1; // 0 / 1, 2 = draw
    int steps = 0;
    std::int64_t sent[2] = {0, 0};
    std::int64_t attack[2] = {0, 0};
    std::shared_ptr<actor::BaseActor> actor[2];
    Environment external_env[2];
    Action chosen[2];
    Environment& env(int p) { return actor[p] ? actor[p]->getEnvironment() : external_env[p]; }
};

class ArenaRunner {
public:
    explicit ArenaRunner(const ArenaSpec& spec) : spec_(spec)
    {
        for (int p = 0; p < 2; ++p) {
            if (spec_.player[p].type == "model") { rules_cfg_ = spec_.player[p].cfg; }
        }
        for (int p = 0; p < 2; ++p) {
            const ArenaPlayerSpec& ps = spec_.player[p];
            if (ps.type == "model") {
                loadConfig(p);
                network_[p] = network::createNetwork(ps.model, ps.gpu);
                tree_size_[p] = static_cast<uint64_t>(config::actor_num_simulation + 1) *
                                (network_[p]->getActionSize() + std::max(0, config::actor_mcts_chance_worlds) + (config::actor_mcts_chance_nodes ? actor::kMaxChanceChildren : 0));
            } else {
                for (int i = 0; i < ps.procs; ++i) { bots_[p].push_back(std::make_unique<ArenaExternalBot>(ps.cmd)); }
            }
        }
    }

    void run()
    {
        std::filesystem::create_directories(spec_.out);
        std::ofstream results(spec_.out + "/results.jsonl");
        std::vector<ArenaMatch> finished;
        for (int start = 0; start < spec_.games; start += spec_.parallel) {
            const int n = std::min(spec_.parallel, spec_.games - start);
            std::vector<ArenaMatch> wave(n);
            for (int i = 0; i < n; ++i) { wave[i].seed = spec_.seed * 1000003U + static_cast<unsigned>(start + i); }
            playWave(wave);
            for (ArenaMatch& m : wave) {
                results << "{\"seed\":" << m.seed << ",\"winner\":" << m.winner << ",\"steps\":" << m.steps
                        << ",\"sent\":[" << m.sent[0] << "," << m.sent[1] << "],\"attack\":[" << m.attack[0] << "," << m.attack[1] << "]}\n";
                finished.push_back(std::move(m));
            }
            results.flush();
            std::cerr << "arena: " << finished.size() << "/" << spec_.games << " matches" << std::endl;
        }
        writeSummary(finished);
    }

private:
    void loadConfig(int p) const
    {
        const ArenaPlayerSpec& ps = spec_.player[p];
        config::ConfigureLoader cl;
        config::setConfiguration(cl);
        cl.loadFromFile(ps.type == "model" ? ps.cfg : rules_cfg_);
        std::string conf = "env_modern_tetris_garbage_probability=0:env_modern_tetris_max_episode_steps=" + std::to_string(spec_.max_steps) +
                           ":actor_mcts_resample_hidden_future=true:actor_mcts_terminal_value_eval_score=false:program_quiet=true";
        if (ps.type == "model") { conf += ":nn_file_name=" + ps.model; }
        if (!ps.conf.empty()) { conf += ":" + ps.conf; }
        cl.loadFromString(conf);
    }

    void playWave(std::vector<ArenaMatch>& wave)
    {
        const int n = static_cast<int>(wave.size());
        for (int p = 0; p < 2; ++p) {
            loadConfig(p);
            for (ArenaMatch& m : wave) {
                if (network_[p]) {
                    m.actor[p] = actor::createActor(tree_size_[p], network_[p]);
                    m.actor[p]->getEnvironment().reset(static_cast<int>(m.seed));
                    m.actor[p]->resetSearch();
                } else {
                    m.external_env[p].reset(static_cast<int>(m.seed));
                }
            }
        }
        while (true) {
            std::vector<int> active;
            for (int i = 0; i < n; ++i) {
                if (!wave[i].done) { active.push_back(i); }
            }
            if (active.empty()) { break; }
            for (int p = 0; p < 2; ++p) {
                loadConfig(p);
                if (network_[p]) {
                    searchModel(wave, active, p);
                } else {
                    askExternal(wave, active, p);
                }
            }
            for (int i : active) { step(wave[i]); }
        }
    }

    void searchModel(std::vector<ArenaMatch>& wave, const std::vector<int>& active, int p)
    {
        auto net = std::static_pointer_cast<network::PlacementTransformerNetwork>(network_[p]);
        parallelFor(static_cast<int>(active.size()), spec_.threads, [&](int k) { wave[active[k]].actor[p]->resetSearch(); });
        while (true) {
            std::vector<int> pending;
            for (int i : active) {
                if (!wave[i].actor[p]->isSearchDone()) { pending.push_back(i); }
            }
            if (pending.empty()) { break; }
            parallelFor(static_cast<int>(pending.size()), spec_.threads, [&](int k) { wave[pending[k]].actor[p]->beforeNNEvaluation(); });
            const auto outputs = net->forward();
            parallelFor(static_cast<int>(pending.size()), spec_.threads, [&](int k) {
                auto& actor = wave[pending[k]].actor[p];
                actor->afterNNEvaluation(outputs[actor->getNNEvaluationBatchIndex()]);
            });
        }
        for (int i : active) { wave[i].chosen[p] = wave[i].actor[p]->getSearchAction(); }
    }

    void askExternal(std::vector<ArenaMatch>& wave, const std::vector<int>& active, int p)
    {
        const int procs = static_cast<int>(bots_[p].size());
        parallelFor(procs, procs, [&](int b) {
            for (std::size_t k = b; k < active.size(); k += procs) {
                ArenaMatch& m = wave[active[k]];
                const std::string reply = bots_[p][b]->request(stateJson(m.env(p), active[k]));
                ArenaExternalMove move = parseExternalMove(m.env(p), reply);
                if (!move.ok) {
                    ++external_fallbacks_;
                    if (external_fallbacks_ <= 5) { std::cerr << "arena: external move not legal here, falling back: " << reply << std::endl; }
                    move.action = m.env(p).getLegalActions().front();
                }
                if (move.spin_mismatch) { ++spin_mismatches_; }
                m.chosen[p] = move.action;
            }
        });
    }

    void step(ArenaMatch& m)
    {
        for (int p = 0; p < 2; ++p) {
            if (m.actor[p]) {
                m.actor[p]->act(m.chosen[p]);
            } else {
                m.external_env[p].act(m.chosen[p]);
            }
        }
        const engine::State& s0 = m.env(0).getEngineState();
        const engine::State& s1 = m.env(1).getEngineState();
        const int sent0 = s0.lines_sent, sent1 = s1.lines_sent;
        m.sent[0] += sent0;
        m.sent[1] += sent1;
        m.attack[0] += s0.attack;
        m.attack[1] += s1.attack;
        m.env(0).receiveGarbage(sent1, spec_.garbage_delay);
        m.env(1).receiveGarbage(sent0, spec_.garbage_delay);
        ++m.steps;
        const bool dead0 = !m.env(0).getEngineState().is_alive, dead1 = !m.env(1).getEngineState().is_alive;
        if (dead0 || dead1) {
            m.winner = (dead0 && dead1) ? 2 : (dead0 ? 1 : 0);
            m.done = true;
        } else if (m.steps >= spec_.max_steps) {
            m.winner = m.sent[0] > m.sent[1] ? 0 : (m.sent[1] > m.sent[0] ? 1 : 2);
            m.done = true;
        }
        if (m.done) {
            for (auto& a : m.actor) { a.reset(); } // free the search trees
        }
    }

    void writeSummary(const std::vector<ArenaMatch>& matches) const
    {
        int wins[3] = {0, 0, 0}, capped = 0;
        double steps = 0, sent[2] = {0, 0}, attack[2] = {0, 0};
        for (const ArenaMatch& m : matches) {
            ++wins[m.winner];
            capped += m.steps >= spec_.max_steps;
            steps += m.steps;
            for (int p = 0; p < 2; ++p) {
                sent[p] += m.sent[p];
                attack[p] += m.attack[p];
            }
        }
        const double n = std::max<std::size_t>(1, matches.size());
        const double score = (wins[0] + 0.5 * wins[2]) / n;
        const double se = std::sqrt(score * (1 - score) / n);
        std::ostringstream oss;
        oss.setf(std::ios::fixed);
        oss.precision(3);
        oss << "A: " << spec_.player[0].name << "\nB: " << spec_.player[1].name << "\n"
            << "matches " << matches.size() << ": A wins " << wins[0] << ", B wins " << wins[1] << ", draws " << wins[2]
            << " (" << capped << " reached the " << spec_.max_steps << "-piece cap)\n"
            << "A score " << score << " +- " << 1.96 * se << " (95%)\n"
            << "pieces per player " << steps / n << "; attack per piece A " << attack[0] / std::max(1.0, steps) << ", B " << attack[1] / std::max(1.0, steps)
            << "; lines sent per match A " << sent[0] / n << ", B " << sent[1] / n << "\n"
            << "external fallbacks " << external_fallbacks_ << ", spin mismatches " << spin_mismatches_ << "\n";
        std::ofstream(spec_.out + "/summary.txt") << oss.str();
        std::cout << oss.str();
    }

    ArenaSpec spec_;
    std::string rules_cfg_;
    std::shared_ptr<network::Network> network_[2];
    uint64_t tree_size_[2] = {0, 0};
    std::vector<std::unique_ptr<ArenaExternalBot>> bots_[2];
    std::atomic<std::int64_t> external_fallbacks_{0};
    std::atomic<std::int64_t> spin_mismatches_{0};
};

void Arena::run()
{
    if (config::arena_spec.empty()) { throw std::runtime_error{"arena: set arena_spec to a spec file"}; }
    ArenaRunner runner(parseSpec(config::arena_spec));
    runner.run();
}

#else

void Arena::run()
{
    std::cerr << "arena mode is only built for moderntetris_placement" << std::endl;
}

#endif

} // namespace minizero::console
