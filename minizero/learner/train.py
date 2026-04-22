#!/usr/bin/env python

import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from minizero.network.py.create_network import create_network
from tools.analysis import analysis


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs, flush=True)


class MinizeroDadaLoader:
    def __init__(self, conf_file_name):
        self.data_loader = py.DataLoader(conf_file_name)
        self.data_loader.initialize()
        self.data_list = []

        # allocate memory
        self.sampled_index = np.zeros(py.get_batch_size() * 2, dtype=np.int32)
        self.features = np.zeros(py.get_batch_size() * py.get_nn_num_input_channels() * py.get_nn_input_channel_height() * py.get_nn_input_channel_width(), dtype=np.float32)
        self.loss_scale = np.zeros(py.get_batch_size(), dtype=np.float32)
        self.value_accumulator = np.ones(1) if py.get_nn_discrete_value_size() == 1 else np.arange(-int(py.get_nn_discrete_value_size() / 2), int(py.get_nn_discrete_value_size() / 2) + 1)
        if py.get_nn_type_name() == "placement_transformer":
            self._init_placement_buffers()
            self.action_features = None
            self.reward = None
        elif py.get_nn_type_name() == "alphazero":
            self.action_features = None
            self.policy = np.zeros(py.get_batch_size() * py.get_nn_action_size(), dtype=np.float32)
            self.value = np.zeros(py.get_batch_size() * py.get_nn_discrete_value_size(), dtype=np.float32)
            self.reward = None
        else:
            self.action_features = np.zeros(py.get_batch_size() * py.get_muzero_unrolling_step() * py.get_nn_num_action_feature_channels()
                                            * py.get_nn_hidden_channel_height() * py.get_nn_hidden_channel_width(), dtype=np.float32)
            self.policy = np.zeros(py.get_batch_size() * (py.get_muzero_unrolling_step() + 1) * py.get_nn_action_size(), dtype=np.float32)
            self.value = np.zeros(py.get_batch_size() * (py.get_muzero_unrolling_step() + 1) * py.get_nn_discrete_value_size(), dtype=np.float32)
            self.reward = np.zeros(py.get_batch_size() * py.get_muzero_unrolling_step() * py.get_nn_discrete_value_size(), dtype=np.float32)

    def _init_placement_buffers(self):
        B = py.get_batch_size()
        N = py.DataLoader.placement_n_max()
        prev = py.get_env_modern_tetris_num_preview_piece()
        H = py.get_nn_input_channel_height()
        W = py.get_nn_input_channel_width()
        self._pl_board = np.zeros(B * 1 * H * W, dtype=np.float32)
        self.policy = np.zeros(B * N, dtype=np.float32)
        self.value = np.zeros(B * py.get_nn_discrete_value_size(), dtype=np.float32)
        self._pl_current = np.zeros(B, dtype=np.int64)
        self._pl_hold = np.zeros(B, dtype=np.int64)
        self._pl_has_held = np.zeros(B, dtype=np.float32)
        self._pl_preview = np.zeros(B * prev, dtype=np.int64)
        self._pl_was_rotation = np.zeros(B, dtype=np.float32)
        self._pl_srs = np.zeros(B, dtype=np.int64)
        self._pl_lifetime = np.zeros(B, dtype=np.float32)
        self._pl_combo = np.zeros(B, dtype=np.float32)
        self._pl_b2b = np.zeros(B, dtype=np.float32)
        self._pl_garbage = np.zeros(B, dtype=np.float32)
        self._pl_a_use_hold = np.zeros(B * N, dtype=np.int64)
        self._pl_a_lock_x = np.zeros(B * N, dtype=np.int64)
        self._pl_a_lock_y = np.zeros(B * N, dtype=np.int64)
        self._pl_a_orient = np.zeros(B * N, dtype=np.int64)
        self._pl_a_spin = np.zeros(B * N, dtype=np.int64)
        self._pl_a_piece = np.zeros(B * N, dtype=np.int64)
        self._pl_a_lines = np.zeros(B * N, dtype=np.int64)
        self._pl_a_mask = np.ones(B * N, dtype=np.uint8)  # default all padded
        self._pl_n_max = N
        self._pl_preview_size = prev

    def _sample_placement(self, device):
        B = py.get_batch_size()
        N = self._pl_n_max
        prev = self._pl_preview_size
        # Reset mask to padded before each sample (C++ side writes valid positions).
        self._pl_a_mask.fill(1)
        self.data_loader.sample_data_placement(
            self._pl_board, self.policy, self.value, self.loss_scale, self.sampled_index,
            self._pl_current, self._pl_hold, self._pl_has_held, self._pl_preview,
            self._pl_was_rotation, self._pl_srs, self._pl_lifetime, self._pl_combo,
            self._pl_b2b, self._pl_garbage,
            self._pl_a_use_hold, self._pl_a_lock_x, self._pl_a_lock_y,
            self._pl_a_orient, self._pl_a_spin, self._pl_a_piece, self._pl_a_lines,
            self._pl_a_mask, N, prev)

        # Trim N down to the real max valid placements in this batch. Attention
        # memory scales with seq_len^2 so this can easily reduce GPU mem ~10x
        # (real max ~40-80 vs N_max=256).
        a_mask_2d = self._pl_a_mask.reshape(B, N)  # 0 = valid, 1 = padded
        valid_counts = (a_mask_2d == 0).sum(axis=1)
        Nr = max(1, int(valid_counts.max()))

        def t2(flat):
            # slice is non-contiguous; copy to satisfy torch.from_numpy
            return torch.from_numpy(flat.reshape(B, N)[:, :Nr].copy()).to(device)

        def t(a, shape):
            return torch.from_numpy(a.reshape(shape)).to(device)

        batch = {
            "board": t(self._pl_board, (B, 1, py.get_nn_input_channel_height(), py.get_nn_input_channel_width())),
            "current": t(self._pl_current, (B,)),
            "hold": t(self._pl_hold, (B,)),
            "has_held": t(self._pl_has_held, (B,)),
            "preview": t(self._pl_preview, (B, prev)),
            "was_rotation": t(self._pl_was_rotation, (B,)),
            "srs": t(self._pl_srs, (B,)),
            "lifetime": t(self._pl_lifetime, (B,)),
            "combo": t(self._pl_combo, (B,)),
            "b2b": t(self._pl_b2b, (B,)),
            "garbage": t(self._pl_garbage, (B,)),
            "a_use_hold": t2(self._pl_a_use_hold),
            "a_lock_x": t2(self._pl_a_lock_x),
            "a_lock_y": t2(self._pl_a_lock_y),
            "a_orient": t2(self._pl_a_orient),
            "a_spin": t2(self._pl_a_spin),
            "a_piece": t2(self._pl_a_piece),
            "a_lines": t2(self._pl_a_lines),
            "a_mask": t2(self._pl_a_mask),
        }
        policy = torch.FloatTensor(self.policy.reshape(B, N)[:, :Nr].copy()).to(device)
        value = torch.FloatTensor(self.value).view(B, py.get_nn_discrete_value_size()).to(device)
        loss_scale = torch.FloatTensor(self.loss_scale / max(1e-8, float(np.amax(self.loss_scale)))).to(device)
        sampled_index = self.sampled_index
        return batch, policy, value, loss_scale, sampled_index

    def load_data(self, training_dir, start_iter, end_iter):
        for i in range(start_iter, end_iter + 1):
            file_name = f"{training_dir}/sgf/{i}.sgf"
            if file_name in self.data_list:
                continue
            self.data_loader.load_data_from_file(file_name)
            self.data_list.append(file_name)
            if len(self.data_list) > py.get_zero_replay_buffer():
                self.data_list.pop(0)

    def sample_data(self, device='cpu'):
        self.data_loader.sample_data(self.features, self.action_features, self.policy, self.value, self.reward, self.loss_scale, self.sampled_index)
        features = torch.FloatTensor(self.features).view(py.get_batch_size(), py.get_nn_num_input_channels(), py.get_nn_input_channel_height(), py.get_nn_input_channel_width()).to(device)
        action_features = None if self.action_features is None else torch.FloatTensor(self.action_features).view(py.get_batch_size(),
                                                                                                                 -1,
                                                                                                                 py.get_nn_num_action_feature_channels(),
                                                                                                                 py.get_nn_hidden_channel_height(),
                                                                                                                 py.get_nn_hidden_channel_width()).to(device)
        policy = torch.FloatTensor(self.policy).view(py.get_batch_size(), -1, py.get_nn_action_size()).to(device)
        value = torch.FloatTensor(self.value).view(py.get_batch_size(), -1, py.get_nn_discrete_value_size()).to(device)
        reward = None if self.reward is None else torch.FloatTensor(self.reward).view(py.get_batch_size(), -1, py.get_nn_discrete_value_size()).to(device)
        loss_scale = torch.FloatTensor(self.loss_scale / np.amax(self.loss_scale)).to(device)
        sampled_index = self.sampled_index

        return features, action_features, policy, value, reward, loss_scale, sampled_index

    def update_priority(self, sampled_index, batch_values):
        batch_values = (batch_values * self.value_accumulator).sum(axis=1)
        self.data_loader.update_priority(sampled_index, batch_values)


class Model:
    def __init__(self):
        self.training_step = 0
        self.network = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.optimizer = None
        self.scheduler = None

    def load_model(self, training_dir, model_file):
        self.training_step = 0
        placement_cfg = None
        if py.get_nn_type_name() == "placement_transformer":
            placement_cfg = {
                "preview_size": py.get_env_modern_tetris_num_preview_piece(),
                "d_model": py.get_nn_placement_d_model(),
                "n_heads": py.get_nn_placement_n_heads(),
                "n_layers": py.get_nn_placement_n_layers(),
                "mlp_ratio": py.get_nn_placement_mlp_ratio(),
                "dropout": py.get_nn_placement_dropout(),
            }
        self.network = create_network(py.get_game_name(),
                                      py.get_nn_num_input_channels(),
                                      py.get_nn_input_channel_height(),
                                      py.get_nn_input_channel_width(),
                                      py.get_nn_num_hidden_channels(),
                                      py.get_nn_hidden_channel_height(),
                                      py.get_nn_hidden_channel_width(),
                                      py.get_nn_num_action_feature_channels(),
                                      py.get_nn_num_blocks(),
                                      py.get_nn_action_size(),
                                      py.get_nn_num_value_hidden_channels(),
                                      py.get_nn_discrete_value_size(),
                                      py.get_nn_type_name(),
                                      placement_cfg)
        self.network.to(self.device)
        if py.get_optimizer().lower() == "adam":
            self.optimizer = optim.Adam(self.network.parameters(),
                                        lr=py.get_learning_rate(),
                                        weight_decay=py.get_weight_decay())
        elif py.get_optimizer().lower() == "adamw":
            self.optimizer = optim.AdamW(self.network.parameters(),
                                         lr=py.get_learning_rate(),
                                         weight_decay=py.get_weight_decay())
        else:
            self.optimizer = optim.SGD(self.network.parameters(),
                                       lr=py.get_learning_rate(),
                                       momentum=py.get_momentum(),
                                       weight_decay=py.get_weight_decay())
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000000, gamma=0.1)

        if model_file:
            snapshot = torch.load(f"{training_dir}/model/{model_file}", map_location=torch.device('cpu'))
            self.training_step = snapshot['training_step']
            self.network.load_state_dict(snapshot['network'])
            self.optimizer.load_state_dict(snapshot['optimizer'])
            self.optimizer.param_groups[0]["lr"] = py.get_learning_rate()
            self.scheduler.load_state_dict(snapshot['scheduler'])

        # for multi-gpu
        self.network = nn.DataParallel(self.network)

    def save_model(self, training_dir):
        snapshot = {'training_step': self.training_step,
                    'network': self.network.module.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict()}
        torch.save(snapshot, f"{training_dir}/model/weight_iter_{self.training_step}.pkl")
        torch.jit.script(self.network.module).save(f"{training_dir}/model/weight_iter_{self.training_step}.pt")


def calculate_loss(network_output, label_policy, label_value, label_reward, loss_scale):
    # policy
    if py.use_gumbel():
        loss_policy = (nn.functional.kl_div(nn.functional.log_softmax(network_output["policy_logit"], dim=1), label_policy, reduction='none').sum(dim=1) * loss_scale).mean()
    else:
        loss_policy = -((label_policy * nn.functional.log_softmax(network_output["policy_logit"], dim=1)).sum(dim=1) * loss_scale).mean()

    # value
    if py.get_nn_discrete_value_size() == 1:
        loss_value = (nn.functional.mse_loss(network_output["value"], label_value, reduction='none') * loss_scale).mean()
    else:
        loss_value = -((label_value * nn.functional.log_softmax(network_output["value_logit"], dim=1)).sum(dim=1) * loss_scale).mean()

    # reward
    loss_reward = 0
    if label_reward is not None and "reward_logit" in network_output:
        loss_reward = -((label_reward * nn.functional.log_softmax(network_output["reward_logit"], dim=1)).sum(dim=1) * loss_scale).mean()

    return loss_policy, loss_value, loss_reward


def add_training_info(training_info, key, value):
    if key not in training_info:
        training_info[key] = 0
    training_info[key] += value


def calculate_accuracy(output, label, batch_size):
    max_output = np.argmax(output.to('cpu').detach().numpy(), axis=1)
    max_label = np.argmax(label.to('cpu').detach().numpy(), axis=1)
    return (max_output == max_label).sum() / batch_size


def train(model, training_dir, data_loader, start_iter, end_iter):
    if start_iter == -1:
        model.save_model(training_dir)
        return

    # load data
    data_loader.load_data(training_dir, start_iter, end_iter)

    training_info = {}
    for i in range(1, py.get_training_step() + 1):
        model.optimizer.zero_grad()

        if py.get_nn_type_name() == "placement_transformer":
            # Linear LR warmup — transformers trained from scratch often get stuck at
            # uniform policy when given full LR immediately. Ramp from 0 to base_lr
            # over the first `warmup_steps` optimizer steps.
            warmup_steps = 2000
            base_lr = py.get_learning_rate()
            if model.training_step < warmup_steps:
                scaled = base_lr * float(model.training_step + 1) / float(warmup_steps)
                for g in model.optimizer.param_groups:
                    g["lr"] = scaled
            elif model.optimizer.param_groups[0]["lr"] != base_lr:
                for g in model.optimizer.param_groups:
                    g["lr"] = base_lr

            batch, label_policy, label_value, loss_scale, _ = data_loader._sample_placement(model.device)
            network_output = model.network(batch["board"], batch["current"], batch["hold"], batch["has_held"],
                                           batch["preview"], batch["was_rotation"], batch["srs"], batch["lifetime"],
                                           batch["combo"], batch["b2b"], batch["garbage"],
                                           batch["a_use_hold"], batch["a_lock_x"], batch["a_lock_y"],
                                           batch["a_orient"], batch["a_spin"], batch["a_piece"],
                                           batch["a_lines"], batch["a_mask"])
            # Policy loss on valid (unmasked) positions only, via KL from label -> predicted log-softmax.
            logits = network_output["policy_logit"]  # [B, N], padded = -1e9
            log_probs = nn.functional.log_softmax(logits, dim=1)
            # label_policy is already normalized across unmasked positions; zero on padded.
            loss_policy = -((label_policy * log_probs).sum(dim=1) * loss_scale).mean()
            loss_value = -((label_value * nn.functional.log_softmax(network_output["value_logit"], dim=1)).sum(dim=1) * loss_scale).mean()
            loss = loss_policy + py.get_value_loss_scale() * loss_value
            add_training_info(training_info, 'loss_policy', loss_policy.item())
            add_training_info(training_info, 'loss_value', loss_value.item())
            loss.backward()
            model.optimizer.step()
            model.scheduler.step()
            model.training_step += 1
            if model.training_step != 0 and model.training_step % py.get_training_display_step() == 0:
                eprint("[{}] nn step {}, lr: {}.".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), model.training_step, round(model.optimizer.param_groups[0]["lr"], 6)))
                for loss_key in training_info:
                    eprint("\t{}: {}".format(loss_key, round(training_info[loss_key] / py.get_training_display_step(), 5)))
                training_info = {}
            continue

        features, action_features, label_policy, label_value, label_reward, loss_scale, sampled_index = data_loader.sample_data(model.device)

        if py.get_nn_type_name() == "alphazero":
            network_output = model.network(features)
            loss_policy, loss_value, _ = calculate_loss(network_output, label_policy[:, 0], label_value[:, 0], None, loss_scale)
            loss = loss_policy + py.get_value_loss_scale() * loss_value

            # record training info
            add_training_info(training_info, 'loss_policy', loss_policy.item())
            add_training_info(training_info, 'accuracy_policy', calculate_accuracy(network_output["policy_logit"], label_policy[:, 0], py.get_batch_size()))
            add_training_info(training_info, 'loss_value', loss_value.item())
        elif py.get_nn_type_name() == "muzero":
            network_output = model.network(features)
            batch_values = network_output['value'].to('cpu').detach().numpy()
            loss_step_policy, loss_step_value, loss_step_reward = calculate_loss(network_output, label_policy[:, 0], label_value[:, 0], None, loss_scale)
            add_training_info(training_info, 'loss_policy_0', loss_step_policy.item())
            add_training_info(training_info, 'accuracy_policy_0', calculate_accuracy(network_output["policy_logit"], label_policy[:, 0], py.get_batch_size()))
            add_training_info(training_info, 'loss_value_0', loss_step_value.item())
            loss_policy = loss_step_policy
            loss_value = loss_step_value
            loss_reward = loss_step_reward
            for i in range(py.get_muzero_unrolling_step()):
                network_output = model.network(network_output["hidden_state"], action_features[:, i])
                batch_values = np.concatenate((batch_values, network_output['value'].to('cpu').detach().numpy()), axis=0)
                loss_step_policy, loss_step_value, loss_step_reward = calculate_loss(network_output, label_policy[:, i + 1], label_value[:, i + 1], label_reward[:, i], loss_scale)
                add_training_info(training_info, f'loss_policy_{i+1}', loss_step_policy.item() / py.get_muzero_unrolling_step())
                add_training_info(training_info, f'accuracy_policy_{i+1}', calculate_accuracy(network_output["policy_logit"], label_policy[:, i + 1], py.get_batch_size()))
                add_training_info(training_info, f'loss_value_{i+1}', loss_step_value.item() / py.get_muzero_unrolling_step())
                if "reward_logit" in network_output:
                    add_training_info(training_info, f'loss_reward_{i+1}', loss_step_reward.item() / py.get_muzero_unrolling_step())
                loss_policy += loss_step_policy / py.get_muzero_unrolling_step()
                loss_value += loss_step_value / py.get_muzero_unrolling_step()
                loss_reward += loss_step_reward / py.get_muzero_unrolling_step()
                network_output["hidden_state"].register_hook(lambda grad: grad / 2)
            if py.use_per():
                data_loader.update_priority(sampled_index, batch_values)
            loss = loss_policy + py.get_value_loss_scale() * loss_value + loss_reward

            add_training_info(training_info, 'loss_policy', loss_policy.item())
            add_training_info(training_info, 'loss_value', loss_value.item())
            if "reward_logit" in network_output:
                add_training_info(training_info, 'loss_reward', loss_reward.item())

        loss.backward()
        model.optimizer.step()
        model.scheduler.step()

        model.training_step += 1
        if model.training_step != 0 and model.training_step % py.get_training_display_step() == 0:
            eprint("[{}] nn step {}, lr: {}.".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), model.training_step, round(model.optimizer.param_groups[0]["lr"], 6)))
            for loss in training_info:
                eprint("\t{}: {}".format(loss, round(training_info[loss] / py.get_training_display_step(), 5)))
            training_info = {}

    model.save_model(training_dir)
    print("Optimization_Done", model.training_step, flush=True)
    eprint("Optimization_Done", model.training_step)
    analysis(training_dir, "analysis")


if __name__ == '__main__':
    if len(sys.argv) == 4:
        game_type = sys.argv[1]
        training_dir = sys.argv[2]
        conf_file_name = sys.argv[3]

        # import pybind library
        _temps = __import__(f'build.{game_type}', globals(), locals(), ['minizero_py'], 0)
        py = _temps.minizero_py
    else:
        eprint("python train.py game_type training_dir conf_file")
        exit(0)

    py.load_config_file(conf_file_name)
    data_loader = MinizeroDadaLoader(conf_file_name)
    model = Model()

    while True:
        try:
            command = input()
            command_prefix = command.split()[0]
            if command == "keep_alive":
                continue

            eprint("[{}] [command] {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), command))
            if command_prefix == "update_config":
                conf_str = command.split(" ", 1)[-1]
                if not py.load_config_string(conf_str):
                    eprint("Failed to load configuration string.")
                    exit(0)
            elif command_prefix == "train":
                _, model_file, start_iter, end_iter = command.split()
                model_file = model_file.replace('"', '')

                # skip loading model if the model is loaded
                if model.network is None:
                    model.load_model(training_dir, model_file)

                train(model, training_dir, data_loader, int(start_iter), int(end_iter))
            elif command_prefix == "quit":
                exit(0)

        except (KeyboardInterrupt, EOFError) as e:
            break
