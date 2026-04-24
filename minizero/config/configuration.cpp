#include "configuration.h"
#include <string>

namespace minizero::config {

// program parameters
int program_seed = 0;
bool program_auto_seed = false;
bool program_quiet = false;

// actor parameters
int actor_num_simulation = 50;
float actor_mcts_puct_base = 19652;
float actor_mcts_puct_init = 1.25;
float actor_mcts_reward_discount = 1.0f;
int actor_mcts_think_batch_size = 1;
float actor_mcts_think_time_limit = 0;
bool actor_mcts_value_rescale = false;
char actor_mcts_value_flipping_player = 'W';
bool actor_select_action_by_count = false;
bool actor_select_action_by_softmax_count = true;
float actor_select_action_softmax_temperature = 1.0f;
bool actor_select_action_softmax_temperature_decay = false;
bool actor_use_random_rotation_features = true;
bool actor_use_dirichlet_noise = true;
float actor_dirichlet_noise_alpha = 0.03f;
float actor_dirichlet_noise_epsilon = 0.25f;
bool actor_use_gumbel = false;
bool actor_use_gumbel_noise = false;
int actor_gumbel_sample_size = 16;
float actor_gumbel_sigma_visit_c = 50;
float actor_gumbel_sigma_scale_c = 1;
float actor_resign_threshold = -0.9f;

// zero parameters
int zero_num_threads = 4;
int zero_num_parallel_games = 32;
int zero_server_port = 9999;
std::string zero_training_directory = "";
int zero_num_games_per_iteration = 2000;
int zero_start_iteration = 0;
int zero_end_iteration = 100;
int zero_replay_buffer = 20;
float zero_disable_resign_ratio = 0.1;
int zero_actor_intermediate_sequence_length = 0;
std::string zero_actor_ignored_command = "reset_actors";
bool zero_server_accept_different_model_games = true;

// learner parameters
bool learner_use_per = false;
float learner_per_alpha = 1.0f;
float learner_per_init_beta = 1.0f;
bool learner_per_beta_anneal = true;
int learner_training_step = 500;
int learner_training_display_step = 100;
int learner_batch_size = 1024;
int learner_muzero_unrolling_step = 5;
int learner_n_step_return = 0;
std::string learner_optimizer = "SGD";
float learner_learning_rate = 0.02;
float learner_momentum = 0.9;
float learner_weight_decay = 0.0001;
float learner_value_loss_scale = 1.0f;
int learner_num_thread = 8;
int learner_placement_cache_interval = 1;

// network parameters
std::string nn_file_name = "";
int nn_num_blocks = 1;
int nn_num_hidden_channels = 256;
int nn_num_value_hidden_channels = 256;
std::string nn_type_name = "alphazero";
int nn_placement_d_model = 192;
int nn_placement_n_layers = 4;
int nn_placement_n_heads = 6;
int nn_placement_mlp_ratio = 4;
float nn_placement_dropout = 0.1f;

// environment parameters
int env_board_size = 0;
std::string env_atari_rom_dir = "/opt/atari57/";
std::string env_atari_name = "ms_pacman";
bool env_conhex_use_swap_rule = true;
float env_go_komi = 7.5;
std::string env_go_ko_rule = "positional";
std::string env_gomoku_rule = "standard";
bool env_gomoku_exactly_five_stones = true;
bool env_havannah_use_swap_rule = true;
bool env_hex_use_swap_rule = true;
bool env_killallgo_use_seki = false;
int env_rubiks_scramble_rotate = 5;
int env_surakarta_no_capture_plies = 50;
int env_modern_tetris_piece_lifetime = 20;
bool env_modern_tetris_auto_drop = false;
int env_modern_tetris_num_preview_piece = 5;
int env_modern_tetris_history_length = 4;
int env_modern_tetris_max_episode_steps = 5000;
// Per-lock reward shaping. Defaults are a "moderate" starting point:
// soft death penalty, small survival bonus, baseline lines_sent attack, plus
// potential-based shaping on board height and holes. Spin / clear / combo
// multipliers are zero by default because lines_sent already includes them.
float env_modern_tetris_reward_survival_bonus = 0.05f;
float env_modern_tetris_reward_death_penalty = 2.0f;
float env_modern_tetris_reward_lines_sent_weight = 1.0f;
float env_modern_tetris_reward_clear_1 = 0.0f;
float env_modern_tetris_reward_clear_2 = 0.0f;
float env_modern_tetris_reward_clear_3 = 0.0f;
float env_modern_tetris_reward_clear_4 = 0.0f;
float env_modern_tetris_reward_tspin_bonus = 0.0f;
float env_modern_tetris_reward_tspin_mini_bonus = 0.0f;
float env_modern_tetris_reward_all_spin_bonus = 0.0f;
float env_modern_tetris_reward_clear_depth_weight_bottom = 1.0f;
float env_modern_tetris_reward_clear_depth_weight_top = 1.0f;
float env_modern_tetris_reward_b2b_bonus = 0.0f;
float env_modern_tetris_reward_combo_bonus = 0.0f;
float env_modern_tetris_reward_perfect_clear_bonus = 5.0f;
float env_modern_tetris_reward_height_weight = 0.02f;
float env_modern_tetris_reward_hole_weight = 0.1f;
int env_tetris_block_puzzle_num_holding_block = 3;
int env_tetris_block_puzzle_num_preview_holding_block = 0;

void setConfiguration(ConfigureLoader& cl)
{
    // program parameters
    cl.addParameter("program_seed", program_seed, "assign a program seed", "Program");
    cl.addParameter("program_auto_seed", program_auto_seed, "true for assigning a random seed automatically", "Program");
    cl.addParameter("program_quiet", program_quiet, "true for silencing the error message", "Program");

    // actor parameters
    cl.addParameter("actor_num_simulation", actor_num_simulation, "simulation number of MCTS", "Actor");
    cl.addParameter("actor_mcts_puct_base", actor_mcts_puct_base, "hyperparameter for puct_bias in the PUCT formula of MCTS, determining the level of exploration", "Actor"); // ref: AZ, Sec. Methods
    cl.addParameter("actor_mcts_puct_init", actor_mcts_puct_init, "hyperparameter for puct_bias in the PUCT formula of MCTS", "Actor");                                       // ref: AZ, Sec. Methods
    cl.addParameter("actor_mcts_reward_discount", actor_mcts_reward_discount, "discount factor for calculating Q values", "Actor");                                           // ref: MZ, Sec. Methods
    cl.addParameter("actor_mcts_value_rescale", actor_mcts_value_rescale, "true for games whose rewards are not bounded in [-1, 1], e.g., Atari games", "Actor");             // ref: MZ
    cl.addParameter("actor_mcts_think_batch_size", actor_mcts_think_batch_size, "the MCTS selection batch size; only works when running console", "Actor");
    cl.addParameter("actor_mcts_think_time_limit", actor_mcts_think_time_limit, "the MCTS time limit in seconds, 0 represents disabling time limit (only uses actor_num_simulation); only works when running console", "Actor");
    cl.addParameter("actor_select_action_by_count", actor_select_action_by_count, "true for selecting the action by the maximum MCTS count; should not be true together with actor_select_action_by_softmax_count", "Actor");
    cl.addParameter("actor_select_action_by_softmax_count", actor_select_action_by_softmax_count, "true for selecting the action by the propotion of MCTS count; should not be true together with actor_select_action_by_count", "Actor");
    cl.addParameter("actor_select_action_softmax_temperature", actor_select_action_softmax_temperature, "the softmax temperature when using actor_select_action_by_softmax_count", "Actor");
    cl.addParameter("actor_select_action_softmax_temperature_decay", actor_select_action_softmax_temperature_decay, "true for decaying the temperature based on training iteration; set 1, 0.5, and 0.25 for 0%-50%, 50%-75%, and 75%-100% of total iterations, respectively", "Actor"); // ref: MZ
    cl.addParameter("actor_use_random_rotation_features", actor_use_random_rotation_features, "true for randomly rotating input features; only supports in alphazero", "Actor");
    cl.addParameter("actor_use_dirichlet_noise", actor_use_dirichlet_noise, "true for adding dirchlet noise to the policy", "Actor");                                          // ref: AZ, Sec. Methods
    cl.addParameter("actor_dirichlet_noise_alpha", actor_dirichlet_noise_alpha, "hyperparameter for dirchlet noise, usually (1 / sqrt(number of actions))", "Actor");          // ref: AZ, Sec. Methods
    cl.addParameter("actor_dirichlet_noise_epsilon", actor_dirichlet_noise_epsilon, "hyperparameter for dirchlet noise", "Actor");                                             // ref: AZ, Sec. Methods
    cl.addParameter("actor_use_gumbel", actor_use_gumbel, "true for enabling Gumbel Zero", "Actor");                                                                           // ref: GZ, Sec. 3
    cl.addParameter("actor_use_gumbel_noise", actor_use_gumbel_noise, "true for adding Gumbel noise to the policy", "Actor");                                                  // ref: GZ, Sec. 3
    cl.addParameter("actor_gumbel_sample_size", actor_gumbel_sample_size, "hyperparameter for Gumbel Zero; the number of sampled actions", "Actor");                           // ref: GZ, Sec. 3
    cl.addParameter("actor_gumbel_sigma_visit_c", actor_gumbel_sigma_visit_c, "hyperparameter for the monotonically increasing transformation sigma in Gumbel Zero", "Actor"); // ref: GZ, Sec. 3.4
    cl.addParameter("actor_gumbel_sigma_scale_c", actor_gumbel_sigma_scale_c, "hyperparameter for the monotonically increasing transformation sigma in Gumbel Zero", "Actor"); // ref: GZ, Sec. 3.4
    cl.addParameter("actor_resign_threshold", actor_resign_threshold, "the threshold determining when to resign in the actor", "Actor");                                       // ref: AG, Sec. Methods

    // zero parameters
    cl.addParameter("zero_num_threads", zero_num_threads, "the number of threads that the zero server uses for zero training", "Zero");
    cl.addParameter("zero_num_parallel_games", zero_num_parallel_games, "the number of games to be run in parallel for zero training", "Zero");
    cl.addParameter("zero_server_port", zero_server_port, "the port number to host the server; workers should connect to this port number", "Zero");
    cl.addParameter("zero_training_directory", zero_training_directory, "the output directory name for storing training results", "Zero");
    cl.addParameter("zero_num_games_per_iteration", zero_num_games_per_iteration, "the nunmber of games to play in each iteration", "Zero");
    cl.addParameter("zero_start_iteration", zero_start_iteration, "the first iteration of training; usually 1 unless continuing with previous training", "Zero");
    cl.addParameter("zero_end_iteration", zero_end_iteration, "the last iteration of training", "Zero");
    cl.addParameter("zero_replay_buffer", zero_replay_buffer, "hyperparameter for replay buffer; replay buffer stores (zero_replay_buffer x zero_num_games_per_iteration) games/sequences", "Zero");
    cl.addParameter("zero_disable_resign_ratio", zero_disable_resign_ratio, "the probability to keep playing when the winrate is below actor_resign_threshold", "Zero");                                                       // ref: AZ, Sec. Methods
    cl.addParameter("zero_actor_intermediate_sequence_length", zero_actor_intermediate_sequence_length, "the max sequence length when running self-play; usually 0 (unlimited) for board games, 200 for atari games", "Zero"); // ref: MZ
    cl.addParameter("zero_actor_ignored_command", zero_actor_ignored_command, "the commands to ignore by the actor; format: command1 command2 ...", "Zero");
    cl.addParameter("zero_server_accept_different_model_games", zero_server_accept_different_model_games, "true for accepting self-play games generated by out-of-date model", "Zero");

    // learner parameters
    cl.addParameter("learner_use_per", learner_use_per, "true for enabling Prioritized Experience Replay", "Learner");                                                              // ref: PER
    cl.addParameter("learner_per_alpha", learner_per_alpha, "hyperparameter for PER that controlls the probability of sampling transition", "Learner");                             // ref: PER, Sec. 3.3
    cl.addParameter("learner_per_init_beta", learner_per_init_beta, "hyperparameter for PER that sets the initial beta value of linearly annealing", "Learner");                    // ref: PER, Sec. 3.4
    cl.addParameter("learner_per_beta_anneal", learner_per_beta_anneal, "hyperparameter for PER that enables linearly anneal beta based on current training iteration", "Learner"); // ref: PER, Sec. 3.4
    cl.addParameter("learner_training_step", learner_training_step, "the number of training steps for updating the model in each iteration", "Learner");
    cl.addParameter("learner_training_display_step", learner_training_display_step, "the training step interval to display training information", "Learner");
    cl.addParameter("learner_batch_size", learner_batch_size, "the batch size for training", "Learner");
    cl.addParameter("learner_muzero_unrolling_step", learner_muzero_unrolling_step, "the number of steps to unroll for muzero training", "Learner");                                // ref: MZ
    cl.addParameter("learner_n_step_return", learner_n_step_return, "the number of steps to calculate the n-step value; usually 0 for board games, 10 for atari games", "Learner"); // ref: MZ
    cl.addParameter("learner_optimizer", learner_optimizer, "the type of optimizer, support SGD, Adam, AdamW", "Learner");
    cl.addParameter("learner_learning_rate", learner_learning_rate, "hyperparameter for learning rate; usually 0.02 for sgd, 0.001 for adam and adamw", "Learner");
    cl.addParameter("learner_momentum", learner_momentum, "hyperparameter for momentum; only for sgd", "Learner");
    cl.addParameter("learner_weight_decay", learner_weight_decay, "hyperparameter for weight decay; usually 0.0001 for sgd, 0 for adam, 0.01 for adamw", "Learner");
    cl.addParameter("learner_value_loss_scale", learner_value_loss_scale, "hyperparameter for scaling of the value loss", "Learner");
    cl.addParameter("learner_num_thread", learner_num_thread, "the number of threads for training", "Learner");
    cl.addParameter("learner_placement_cache_interval", learner_placement_cache_interval, "moderntetris_placement: store a replay snapshot every K positions (1 = every step; higher = less memory, more per-sample replay)", "Learner");

    // network parameters
    cl.addParameter("nn_file_name", nn_file_name, "the file name of model weights", "Network");
    cl.addParameter("nn_num_blocks", nn_num_blocks, "hyperparameter for the model; the number of the residual blocks", "Network");                                                  // ref: AGZ
    cl.addParameter("nn_num_hidden_channels", nn_num_hidden_channels, "hyperparameter for the model; the size of the hidden channels in residual blocks", "Network");               // ref: AGZ
    cl.addParameter("nn_num_value_hidden_channels", nn_num_value_hidden_channels, "hyperparameter for the model; the size of the hidden channels in the value network", "Network"); // ref: AGZ
    cl.addParameter("nn_type_name", nn_type_name, "the type of training algorithm and network: alphazero/muzero/placement_transformer", "Network");
    cl.addParameter("nn_placement_d_model", nn_placement_d_model, "placement transformer: hidden dim", "Network");
    cl.addParameter("nn_placement_n_layers", nn_placement_n_layers, "placement transformer: number of encoder layers", "Network");
    cl.addParameter("nn_placement_n_heads", nn_placement_n_heads, "placement transformer: number of attention heads", "Network");
    cl.addParameter("nn_placement_mlp_ratio", nn_placement_mlp_ratio, "placement transformer: FFN expansion ratio", "Network");
    cl.addParameter("nn_placement_dropout", nn_placement_dropout, "placement transformer: dropout", "Network");

    // environment parameters
    cl.addParameter("env_board_size", env_board_size, "the size of board", "Environment");
#if ATARI
    cl.addParameter("env_atari_rom_dir", env_atari_rom_dir, "the file path of the atari rom", "Environment");
    cl.addParameter("env_atari_name", env_atari_name, "the atari game to play; supported 57 atari games:\n"
                                                      "#\talien amidar assault asterix asteroids atlantis bank_heist battle_zone beam_rider berzerk\n"
                                                      "#\tbowling boxing breakout centipede chopper_command crazy_climber defender demon_attack double_dunk enduro\n"
                                                      "#\tfishing_derby freeway frostbite gopher gravitar hero ice_hockey jamesbond kangaroo krull\n"
                                                      "#\tkung_fu_master montezuma_revenge ms_pacman name_this_game phoenix pitfall pong private_eye qbert riverraid\n"
                                                      "#\troad_runner robotank seaquest skiing solaris space_invaders star_gunner surround tennis time_pilot\n"
                                                      "#\ttutankham up_n_down venture video_pinball wizard_of_wor yars_revenge zaxxon",
                    "Environment");
#elif CONHEX
    cl.addParameter("env_conhex_use_swap_rule", env_conhex_use_swap_rule, "the swap rule in ConHex", "Environment");
#elif GO
    cl.addParameter("env_go_komi", env_go_komi, "the komi in Go", "Environment");
    cl.addParameter("env_go_ko_rule", env_go_ko_rule, "the ko rules in Go: positional (only consider stones), situational (consider stones and the turn)", "Environment");
#elif GOMOKU
    cl.addParameter("env_gomoku_rule", env_gomoku_rule, "the opening rule in Gomoku: standard (standard Gomoku rule), outer_open (restricted first Black move)", "Environment");
    cl.addParameter("env_gomoku_exactly_five_stones", env_gomoku_exactly_five_stones, "true for standard Gomoku; false for freestyle Gomoku (allow winning with more than five stones, i.e., an overline)", "Environment");
#elif HAVANNAH
    cl.addParameter("env_havannah_use_swap_rule", env_havannah_use_swap_rule, "true for enabling swap rule in Havannah", "Environment");
#elif HEX
    cl.addParameter("env_hex_use_swap_rule", env_hex_use_swap_rule, "the swap rule in Hex", "Environment");
#elif KILLALLGO
    cl.addParameter("env_killallgo_ko_rule", env_go_ko_rule, "the ko rules in Killall-Go: positional (only consider stones), situational (consider stones and the turn)", "Environment");
    cl.addParameter("env_killallgo_use_seki", env_killallgo_use_seki, "true for enabling seki", "Environment");
#elif MODERNTETRIS
#define MODERNTETRIS_CONFIG_REGISTRATION 1
#elif MODERNTETRIS_PLACEMENT
#define MODERNTETRIS_CONFIG_REGISTRATION 1
#elif RUBIKS
    cl.addParameter("env_rubiks_scramble_rotate", env_rubiks_scramble_rotate, "the number random rotations from the initial state of a rubik's cube", "Enviroment");
#elif SURAKARTA
    cl.addParameter("env_surakarta_no_capture_plies", env_surakarta_no_capture_plies, "game is over if playing this plies without capture", "Environment");
#elif TETRISBLOCKPUZZLE
    cl.addParameter("env_tetris_block_puzzle_num_holding_block", env_tetris_block_puzzle_num_holding_block, "number of holding block", "Environment");
    cl.addParameter("env_tetris_block_puzzle_num_preview_holding_block", env_tetris_block_puzzle_num_preview_holding_block, "number of preview holding block", "Environment");
#endif

#ifdef MODERNTETRIS_CONFIG_REGISTRATION
    cl.addParameter("env_modern_tetris_piece_lifetime", env_modern_tetris_piece_lifetime, "the number of actions before the current piece is forced to hard-drop", "Environment");
    cl.addParameter("env_modern_tetris_auto_drop", env_modern_tetris_auto_drop, "true for enabling one-row gravity after each action", "Environment");
    cl.addParameter("env_modern_tetris_num_preview_piece", env_modern_tetris_num_preview_piece, "the number of preview pieces exposed in input features", "Environment");
    cl.addParameter("env_modern_tetris_history_length", env_modern_tetris_history_length, "the number of previous active-piece occupancies kept in the input features", "Environment");
    cl.addParameter("env_modern_tetris_max_episode_steps", env_modern_tetris_max_episode_steps, "the maximum number of agent actions in one episode", "Environment");
    cl.addParameter("env_modern_tetris_reward_survival_bonus", env_modern_tetris_reward_survival_bonus, "reward shaping: bonus added for each lock that does not kill", "Environment");
    cl.addParameter("env_modern_tetris_reward_death_penalty", env_modern_tetris_reward_death_penalty, "reward shaping: positive value subtracted when the lock kills the piece", "Environment");
    cl.addParameter("env_modern_tetris_reward_lines_sent_weight", env_modern_tetris_reward_lines_sent_weight, "reward shaping: multiplier for engine-computed lines_sent (the canonical attack value)", "Environment");
    cl.addParameter("env_modern_tetris_reward_clear_1", env_modern_tetris_reward_clear_1, "reward shaping: extra bonus for a single-line clear (on top of lines_sent)", "Environment");
    cl.addParameter("env_modern_tetris_reward_clear_2", env_modern_tetris_reward_clear_2, "reward shaping: extra bonus for a double clear", "Environment");
    cl.addParameter("env_modern_tetris_reward_clear_3", env_modern_tetris_reward_clear_3, "reward shaping: extra bonus for a triple clear", "Environment");
    cl.addParameter("env_modern_tetris_reward_clear_4", env_modern_tetris_reward_clear_4, "reward shaping: extra bonus for a tetris clear", "Environment");
    cl.addParameter("env_modern_tetris_reward_tspin_bonus", env_modern_tetris_reward_tspin_bonus, "reward shaping: extra bonus when a line clear comes with a T-spin", "Environment");
    cl.addParameter("env_modern_tetris_reward_tspin_mini_bonus", env_modern_tetris_reward_tspin_mini_bonus, "reward shaping: extra bonus when a line clear comes with a T-spin mini", "Environment");
    cl.addParameter("env_modern_tetris_reward_all_spin_bonus", env_modern_tetris_reward_all_spin_bonus, "reward shaping: extra bonus when a line clear comes with an all-spin (non-T piece spin)", "Environment");
    cl.addParameter("env_modern_tetris_reward_clear_depth_weight_bottom", env_modern_tetris_reward_clear_depth_weight_bottom, "reward shaping: multiplier applied to all clear-related rewards (lines_sent, clear_N, spin, b2b, combo, perfect_clear) when the piece locks at the bottom visible row; linearly interpolated with clear_depth_weight_top by pre-lock y", "Environment");
    cl.addParameter("env_modern_tetris_reward_clear_depth_weight_top", env_modern_tetris_reward_clear_depth_weight_top, "reward shaping: multiplier applied to all clear-related rewards when the piece locks at the top visible row; linearly interpolated with clear_depth_weight_bottom by pre-lock y", "Environment");
    cl.addParameter("env_modern_tetris_reward_b2b_bonus", env_modern_tetris_reward_b2b_bonus, "reward shaping: flat bonus when a back-to-back streak is active on clear", "Environment");
    cl.addParameter("env_modern_tetris_reward_combo_bonus", env_modern_tetris_reward_combo_bonus, "reward shaping: bonus multiplied by combo_count on clear", "Environment");
    cl.addParameter("env_modern_tetris_reward_perfect_clear_bonus", env_modern_tetris_reward_perfect_clear_bonus, "reward shaping: bonus when the lock results in a perfect clear", "Environment");
    cl.addParameter("env_modern_tetris_reward_height_weight", env_modern_tetris_reward_height_weight, "potential shaping: weight for max column height (lower = better)", "Environment");
    cl.addParameter("env_modern_tetris_reward_hole_weight", env_modern_tetris_reward_hole_weight, "potential shaping: weight for hole count (lower = better)", "Environment");
#endif

    // references
    // [AZ] A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play
    // [MZ] Mastering Atari, Go, chess and shogi by planning with a learned model
    // [GZ] Policy improvement by planning with Gumbel
    // [AG] Mastering the game of Go with deep neural networks and tree search
    // [AGZ] Mastering the game of Go without human knowledge
    // [PER] Prioritized Experience Replay
}

} // namespace minizero::config
