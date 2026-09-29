// Replays moderntetris_placement self-play records through the env and reports
// what the agent actually scored with: clear types, spins, and how much of the
// attack reward came from the combo and back-to-back knobs.
//
// The sgf only stores action ids, so every statistic here comes from replaying
// the game: reset to the record's seed, then act() each placement and read the
// post-lock engine state. Rewards use the same RewardConfig as training, so the
// decomposition is the reward the agent was actually trained on -- not the raw
// engine attack.
//
// Usage: placement_sgf_stats <conf_file> <sgf_file>...
#include "configuration.h"
#include "environment.h"
#include "moderntetris_placement.h"
#include "reward_common.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <regex>
#include <string>
#include <vector>

namespace tetris = minizero::env::moderntetris;
namespace placement = minizero::env::moderntetris_placement;
namespace engine = tetris::engine;

namespace {

struct Stats {
    int games = 0;
    int placements = 0;
    int deaths = 0;
    // clears, indexed by lines cleared (1..4); index 0 counts placements that cleared nothing
    int normal[5] = {};
    int tspin[4] = {};      // full T-spin by lines (1..3)
    int tspin_mini[3] = {}; // mini T-spin by lines (1..2)
    int allspin = 0;        // non-T spin clear
    int perfect_clear = 0;
    int clears_in_combo = 0; // clears that landed with a combo already running
    int max_combo = 0;
    int max_b2b = 0;
    // Player-facing measures, from the raw engine state rather than the shaped reward
    std::int64_t raw_attack = 0; // engine attack, summed (APP = raw_attack / placements)
    int b2b_clears = 0;          // clears that keep / start a back-to-back chain (tetris, any spin)
    int breaking_clears = 0;     // clears that break it (single/double/triple without a spin)
    std::vector<int> b2b_chains; // length of every finished chain, in qualifying clears
    int t_pieces = 0;            // placements that locked a T
    double height_sum = 0;       // stack height before each placement
    double height_clear_sum = 0; // ... before each clearing placement
    double height_combo_sum = 0; // ... before each clear that continued a combo
    int combo_clears = 0;
    double attack_reward = 0; // depth-weighted attack bucket (no survival/death term)
    double combo_part = 0;    // the part of it produced by attack_combo_weight
    double b2b_part = 0;      // the part produced by attack_b2b
    double total_reward = 0;  // full per-lock reward, as trained (incl. death penalty)
};

// Reward configs that isolate one attack component: everything else is zeroed,
// so computeLockBaseReward returns just that component's depth-weighted value.
struct RewardConfigs {
    tetris::reward::RewardConfig attack, combo_only, b2b_only, full;
};

RewardConfigs makeRewardConfigs()
{
    RewardConfigs r;
    r.full = tetris::reward::RewardConfig::fromGlobals();
    r.attack = r.full;
    r.attack.survival_bonus = 0;
    r.attack.death_penalty = 0;
    r.attack.height_weight = 0;
    r.attack.hole_weight = 0;

    r.combo_only = r.attack;
    r.combo_only.attack_single = 0;
    r.combo_only.attack_double = 0;
    r.combo_only.attack_triple = 0;
    r.combo_only.attack_tetris = 0;
    r.combo_only.attack_tspin_single = 0;
    r.combo_only.attack_tspin_double = 0;
    r.combo_only.attack_tspin_triple = 0;
    r.combo_only.attack_tspin_mini_single = 0;
    r.combo_only.attack_tspin_mini_double = 0;
    r.combo_only.attack_allspin = 0;
    r.combo_only.attack_pc = 0;
    r.combo_only.attack_b2b = 0;

    r.b2b_only = r.combo_only;
    r.b2b_only.attack_combo_weight = 0;
    r.b2b_only.attack_b2b = r.attack.attack_b2b;
    return r;
}

// Height of the visible stack: rows from the bottom up to the highest occupied row.
int stackHeight(const engine::State& state)
{
    for (int y = engine::BOARD_TOP; y <= engine::BOARD_BOTTOM; ++y) {
        if (state.board.data[y] != engine::ROW_EMPTY) { return engine::BOARD_BOTTOM - y + 1; }
    }
    return 0;
}

// The piece this placement locks: the current piece, or what a hold would swap in.
engine::PieceType lockedPiece(const engine::State& pre, bool use_hold)
{
    if (!use_hold) { return pre.current; }
    return pre.hold != engine::PieceType::NONE ? pre.hold : pre.next[0];
}

void recordPlacement(Stats& s, const engine::State& pre, const engine::State& post,
                     engine::PieceType piece, int locked_y, const RewardConfigs& cfgs)
{
    const int lines = static_cast<int>(post.lines_cleared);
    const bool is_t = piece == engine::PieceType::T;
    const bool died = !post.is_alive;
    if (died) { s.deaths++; }

    if (lines == 0) {
        s.normal[0]++;
    } else if (post.perfect_clear) {
        s.perfect_clear++;
    } else if (post.spin_type == engine::SpinType::SPIN && is_t) {
        s.tspin[std::min(lines, 3)]++;
    } else if (post.spin_type == engine::SpinType::SPIN_MINI && is_t) {
        s.tspin_mini[std::min(lines, 2)]++;
    } else if (post.spin_type != engine::SpinType::NONE) {
        s.allspin++;
    } else {
        s.normal[std::min(lines, 4)]++;
    }

    if (lines > 0 && pre.combo_count > 0) { s.clears_in_combo++; }

    const int height = stackHeight(pre);
    s.height_sum += height;
    if (is_t) { s.t_pieces++; }
    s.raw_attack += post.attack;
    if (lines > 0) {
        s.height_clear_sum += height;
        if (pre.combo_count >= 0) { // a combo was already running, so this clear extends it
            s.height_combo_sum += height;
            s.combo_clears++;
        }
        // back_to_back_count: -1 with no chain, 0 on its first qualifying clear, +1 per
        // further one; a non-qualifying clear drops it back to -1.
        if (post.back_to_back_count >= 0) {
            s.b2b_clears++;
        } else {
            s.breaking_clears++;
            if (pre.back_to_back_count >= 0) { s.b2b_chains.push_back(pre.back_to_back_count + 1); }
        }
    }
    s.max_combo = std::max(s.max_combo, static_cast<int>(post.combo_count));
    s.max_b2b = std::max(s.max_b2b, static_cast<int>(post.back_to_back_count));

    s.attack_reward += tetris::reward::computeLockBaseReward(post, piece, locked_y, false, cfgs.attack);
    s.combo_part += tetris::reward::computeLockBaseReward(post, piece, locked_y, false, cfgs.combo_only);
    s.b2b_part += tetris::reward::computeLockBaseReward(post, piece, locked_y, false, cfgs.b2b_only);
    s.total_reward += tetris::reward::computeLockBaseReward(post, piece, locked_y, died, cfgs.full);
}

void replayGame(const std::string& record, Stats& s, const RewardConfigs& cfgs)
{
    static const std::regex seed_re("SD\\[(-?\\d+)\\]");
    static const std::regex action_re(";B\\[(\\d+)\\]");

    std::smatch m;
    if (!std::regex_search(record, m, seed_re)) { return; }
    placement::ModernTetrisPlacementEnv env;
    env.reset(std::stoi(m[1].str()));
    s.games++;

    for (auto it = std::sregex_iterator(record.begin(), record.end(), action_re);
         it != std::sregex_iterator(); ++it) {
        const int action_id = std::stoi((*it)[1].str());
        const engine::State pre = env.getEngineState();
        const placement::UnpackedPlacement unpacked = placement::unpackPlacementId(action_id);
        const engine::PieceType piece = lockedPiece(pre, unpacked.use_hold);
        placement::ModernTetrisPlacementAction action(action_id, minizero::env::Player::kPlayer1);
        if (!env.act(action)) {
            std::cerr << "illegal action " << action_id << " at placement " << s.placements << "\n";
            return;
        }
        s.placements++;
        recordPlacement(s, pre, env.getEngineState(), piece, unpacked.lock_y, cfgs);
    }
    const int running = env.getEngineState().back_to_back_count;
    if (running >= 0) { s.b2b_chains.push_back(running + 1); }
}

void report(const std::string& label, const Stats& s)
{
    if (s.games == 0) {
        std::cout << label << ": no games\n";
        return;
    }
    const double per_game = 1.0 / s.games;
    const int clears = s.normal[1] + s.normal[2] + s.normal[3] + s.normal[4] + s.allspin + s.perfect_clear +
                       s.tspin[1] + s.tspin[2] + s.tspin[3] + s.tspin_mini[1] + s.tspin_mini[2];
    const double attack = s.attack_reward != 0 ? s.attack_reward : 1;
    std::printf("%s\n", label.c_str());
    std::printf("  games %d, placements %d (%.1f/game), deaths %d, total reward %.1f/game\n",
                s.games, s.placements, s.placements * per_game, s.deaths, s.total_reward * per_game);
    std::printf("  clears %.1f/game: single %.1f  double %.1f  triple %.1f  tetris %.1f\n",
                clears * per_game, s.normal[1] * per_game, s.normal[2] * per_game,
                s.normal[3] * per_game, s.normal[4] * per_game);
    std::printf("         t-spin %.2f (S %.2f D %.2f T %.2f)  t-spin mini %.2f  all-spin %.2f  perfect clear %.2f\n",
                (s.tspin[1] + s.tspin[2] + s.tspin[3]) * per_game, s.tspin[1] * per_game,
                s.tspin[2] * per_game, s.tspin[3] * per_game,
                (s.tspin_mini[1] + s.tspin_mini[2]) * per_game, s.allspin * per_game,
                s.perfect_clear * per_game);
    std::printf("  attack reward %.1f/game: from combo %.1f (%.1f%%), from b2b %.1f (%.1f%%)\n",
                s.attack_reward * per_game, s.combo_part * per_game, 100 * s.combo_part / attack,
                s.b2b_part * per_game, 100 * s.b2b_part / attack);
    std::printf("  clears while a combo was running: %.1f%%, max combo %d, max b2b %d\n",
                clears > 0 ? 100.0 * s.clears_in_combo / clears : 0.0, s.max_combo, s.max_b2b);

    // player-facing measures (raw engine attack, not the shaped reward)
    double chain_sum = 0;
    int chain_max = 0;
    for (int c : s.b2b_chains) {
        chain_sum += c;
        chain_max = std::max(chain_max, c);
    }
    const int tspins = s.tspin[1] + s.tspin[2] + s.tspin[3];
    std::printf("  APP %.3f (raw attack %.1f/game over %.1f pieces)\n",
                s.placements > 0 ? static_cast<double>(s.raw_attack) / s.placements : 0.0,
                s.raw_attack * per_game, s.placements * per_game);
    std::printf("  b2b: %.1f%% of clears keep it (%d kept / %d broke); chains %zu, mean %.2f, longest %d\n",
                (s.b2b_clears + s.breaking_clears) > 0 ? 100.0 * s.b2b_clears / (s.b2b_clears + s.breaking_clears) : 0.0,
                s.b2b_clears, s.breaking_clears, s.b2b_chains.size(),
                s.b2b_chains.empty() ? 0.0 : chain_sum / s.b2b_chains.size(), chain_max);
    std::printf("  t-spin use: %.1f%% of T pieces (full), %.1f%% incl. mini; %.1f T pieces/game\n",
                s.t_pieces > 0 ? 100.0 * tspins / s.t_pieces : 0.0,
                s.t_pieces > 0 ? 100.0 * (tspins + s.tspin_mini[1] + s.tspin_mini[2]) / s.t_pieces : 0.0,
                s.t_pieces * per_game);
    std::printf("  stack height: %.2f on average, %.2f at clears, %.2f at combo-extending clears (%d)\n",
                s.placements > 0 ? s.height_sum / s.placements : 0.0,
                clears > 0 ? s.height_clear_sum / clears : 0.0,
                s.combo_clears > 0 ? s.height_combo_sum / s.combo_clears : 0.0, s.combo_clears);
}

} // namespace

int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " <conf_file> <sgf_file>...\n";
        return 1;
    }
    minizero::env::setUpEnv();
    minizero::config::ConfigureLoader cl;
    minizero::config::setConfiguration(cl);
    if (!cl.loadFromFile(argv[1])) {
        std::cerr << "failed to load " << argv[1] << "\n";
        return 1;
    }
    const RewardConfigs cfgs = makeRewardConfigs();

    for (int i = 2; i < argc; ++i) {
        std::ifstream fin(argv[i]);
        if (!fin) {
            std::cerr << "failed to open " << argv[i] << "\n";
            return 1;
        }
        Stats s;
        std::string line;
        while (std::getline(fin, line)) {
            if (line.compare(0, 4, "(;GM") == 0) { replayGame(line, s, cfgs); }
        }
        report(argv[i], s);
    }
    return 0;
}
