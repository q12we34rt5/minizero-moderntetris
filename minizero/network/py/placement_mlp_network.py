import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

try:
    from .placement_transformer_network import LOCK_X_OFFSET, LOCK_Y_OFFSET, lock_grid_size
except ImportError:  # allow running this file directly for the smoke test
    from placement_transformer_network import LOCK_X_OFFSET, LOCK_Y_OFFSET, lock_grid_size


# Design philosophy (per-action MLP, "simple projection -> deep MLP"):
#
#   state features  -> ONE linear projection -> state_h  [B, d]
#   action features -> ONE linear projection -> action_h [B, N, d]
#   logit[i]        = DeepMLP( cat[state_h, action_h[i]] )      (per action)
#   value           = ValueMLP( state_h )
#
# The front end is deliberately dumb: every categorical field is one-hot encoded
# and projected by a single Linear (one-hot @ Linear == embedding lookup), and
# scalars are passed raw. ALL the learning of state<->action interaction lives in
# the deep policy MLP. This mirrors the transformer's split (cheap token
# embeddings + heavy backbone) and keeps the ablation clean: the only thing that
# differs from PlacementTransformerNetwork is the backbone (independent per-action
# MLP vs. self-attention over state + all actions).
#
# Inputs (the 18 forward tensors) and outputs are identical to
# PlacementTransformerNetwork, so the C++ wrapper / actor / training all reuse the
# same path unchanged (get_type_name() also returns "placement_transformer").


def _make_mlp(in_dim: int, hidden: int, out_dim: int,
              n_hidden_layers: int, dropout: float) -> nn.Sequential:
    """LayerNorm -> [Linear -> GELU -> Dropout] * n_hidden_layers -> Linear."""
    layers: List[nn.Module] = [nn.LayerNorm(in_dim)]
    d = in_dim
    for _ in range(n_hidden_layers):
        layers.append(nn.Linear(d, hidden))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class ConvBoardEncoder(nn.Module):
    """[B, C, H, W] -> [B, conv_hidden*H*W] via a small conv stack (keeps spatial
    structure), then flattened. Used by the mlp_conv backbone."""

    def __init__(self, board_channels: int, conv_hidden_channels: int,
                 num_conv_layers: int):
        super().__init__()
        layers: List[nn.Module] = []
        in_ch = board_channels
        for _ in range(num_conv_layers):
            layers.append(nn.Conv2d(in_ch, conv_hidden_channels, kernel_size=3, padding=1))
            layers.append(nn.GELU())
            in_ch = conv_hidden_channels
        layers.append(nn.Flatten())
        self.net = nn.Sequential(*layers)

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        return self.net(board)


class PlacementMLPNetwork(nn.Module):
    """Per-action MLP placement network with simple linear front-end projections
    and a deep policy MLP. board_encoder selects how the board enters the state
    feature vector: "flat" (raw flatten -> part of the single state projection,
    no spatial prior) or "conv" (small conv stack first)."""

    def __init__(self,
                 game_name: str = "moderntetris_placement",
                 board_channels: int = 1,
                 board_height: int = 20,
                 board_width: int = 10,
                 num_piece_types: int = 7,
                 preview_size: int = 5,
                 d_model: int = 192,
                 board_encoder: str = "flat",
                 hidden_ratio: int = 4,
                 num_mlp_layers: int = 3,
                 num_value_mlp_layers: int = 2,
                 dropout: float = 0.1,
                 conv_hidden_channels: int = 32,
                 num_conv_layers: int = 3,
                 num_value_hidden_channels: int = 256,
                 discrete_value_size: int = 601,
                 afterstate_feature_size: int = 0):
        super().__init__()
        assert board_encoder in ("flat", "conv"), board_encoder
        self.game_name = game_name
        self.board_channels = board_channels
        self.board_height = board_height
        self.board_width = board_width
        self.num_piece_types = num_piece_types
        self.num_piece_types_with_none = num_piece_types + 1  # +1 NONE sentinel (index 7)
        self.srs_size = 7  # srs_index shifted to 0..6
        self.preview_size = preview_size
        self.d_model = d_model
        self.board_encoder_name = board_encoder
        self.discrete_value_size = discrete_value_size
        # Lock anchors fall outside the visible board; the one-hot axes are offset
        # and widened to cover the real range (see LOCK_*_OFFSET in
        # placement_transformer_network.py). Same treatment in both backbones.
        self.lock_y_size, self.lock_x_size = lock_grid_size(board_height, board_width)
        self.lock_x_offset = LOCK_X_OFFSET
        self.lock_y_offset = LOCK_Y_OFFSET

        # --- board entry into the state feature vector ---
        if board_encoder == "flat":
            self.board_encode: nn.Module = nn.Flatten()          # raw, no params
            board_feat_dim = board_channels * board_height * board_width
        else:
            self.board_encode = ConvBoardEncoder(board_channels, conv_hidden_channels,
                                                 num_conv_layers)
            board_feat_dim = conv_hidden_channels * board_height * board_width

        # --- state side: assemble raw features, ONE linear projection -> [B, d] ---
        # one-hots: current(8) + hold(8) + preview(8*P) + srs(7); scalars: 5
        # (has_held, was_rotation, combo, back_to_back, pending_garbage)
        state_feat_dim = (board_feat_dim
                          + self.num_piece_types_with_none * (2 + preview_size)
                          + self.srs_size
                          + 5)
        self.state_proj = nn.Linear(state_feat_dim, d_model)

        # --- action side: one-hot all fields, ONE linear projection -> [B, N, d] ---
        # use_hold(2) + lock_x(W+2) + lock_y(H+9) + orient(4) + spin(3) + piece(7) + lines(5)
        action_feat_dim = (2 + self.lock_x_size + self.lock_y_size + 4 + 3
                           + num_piece_types + 5)
        self.action_proj = nn.Linear(action_feat_dim, d_model)
        # Optional afterstate summary, added onto the action embedding. Held in a
        # ModuleList for the same reason as in the transformer backbone: the
        # disabled variant then has no extra parameters and TorchScript still
        # compiles the forward. See ActionTokenEmbed in
        # placement_transformer_network.py.
        self.afterstate_proj = nn.ModuleList()
        if afterstate_feature_size > 0:
            self.afterstate_proj.append(nn.Sequential(
                nn.Linear(afterstate_feature_size, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            ))

        # --- deep policy MLP over [state_h, action_h] -> per-action logit ---
        hidden = d_model * hidden_ratio
        self.policy_mlp = _make_mlp(d_model * 2, hidden, 1, num_mlp_layers, dropout)

        # --- value MLP over state_h ---
        if discrete_value_size == 1:
            self.value_mlp = _make_mlp(d_model, num_value_hidden_channels, 1,
                                       num_value_mlp_layers, dropout)
            self.value_act = nn.Tanh()
        else:
            self.value_mlp = _make_mlp(d_model, num_value_hidden_channels,
                                       discrete_value_size, num_value_mlp_layers, dropout)
            self.value_act = nn.Identity()

    @torch.jit.export
    def get_type_name(self) -> str:
        # Same runtime type as the transformer so the C++ wrapper / actor / console
        # / data loader treat it identically; the concrete backbone is recorded by
        # the nn_placement_backbone config.
        return "placement_transformer"

    @torch.jit.export
    def get_game_name(self) -> str:
        return self.game_name

    @torch.jit.export
    def get_discrete_value_size(self) -> int:
        return self.discrete_value_size

    @torch.jit.export
    def get_d_model(self) -> int:
        return self.d_model

    def _state_h(self,
                 board: torch.Tensor,
                 current_piece: torch.Tensor,
                 hold_piece: torch.Tensor,
                 has_held: torch.Tensor,
                 preview: torch.Tensor,
                 was_rotation: torch.Tensor,
                 srs_index: torch.Tensor,
                 combo: torch.Tensor,
                 back_to_back: torch.Tensor,
                 pending_garbage: torch.Tensor) -> torch.Tensor:
        board_feat = self.board_encode(board)                                         # [B, board_feat_dim]
        npn = self.num_piece_types_with_none
        current_oh = F.one_hot(current_piece, npn).to(board_feat.dtype)               # [B, 8]
        hold_oh = F.one_hot(hold_piece, npn).to(board_feat.dtype)                     # [B, 8]
        preview_oh = F.one_hot(preview, npn).to(board_feat.dtype).flatten(1)          # [B, 8*P]
        srs_oh = F.one_hot(srs_index, self.srs_size).to(board_feat.dtype)            # [B, 7]
        scalars = torch.stack([has_held, was_rotation, combo, back_to_back,
                               pending_garbage], dim=1)                               # [B, 5]
        state_feat = torch.cat([board_feat, current_oh, hold_oh, preview_oh,
                                srs_oh, scalars], dim=1)
        return self.state_proj(state_feat)                                           # [B, d]

    def _action_h(self,
                  action_use_hold: torch.Tensor,
                  action_lock_x: torch.Tensor,
                  action_lock_y: torch.Tensor,
                  action_orientation: torch.Tensor,
                  action_spin_type: torch.Tensor,
                  action_piece_type: torch.Tensor,
                  action_lines_cleared: torch.Tensor,
                  action_afterstate: Optional[torch.Tensor] = None) -> torch.Tensor:
        # all [B, N] long, except action_afterstate: [B, N, F] float
        dt = self.action_proj.weight.dtype
        # Shift the bounding-box anchor into the widened one-hot range: lock_x can be
        # as low as -2 (left-wall hugging shapes) and lock_y as low as -9 (hidden
        # spawn buffer). The clamps are a safety net for padded slots / unexpected
        # descriptors; real placements never trigger them. Same as
        # PlacementTransformerNetwork.ActionTokenEmbed.
        lock_x = (action_lock_x + self.lock_x_offset).clamp(0, self.lock_x_size - 1)
        lock_y = (action_lock_y + self.lock_y_offset).clamp(0, self.lock_y_size - 1)
        feat = torch.cat([
            F.one_hot(action_use_hold, 2).to(dt),
            F.one_hot(lock_x, self.lock_x_size).to(dt),
            F.one_hot(lock_y, self.lock_y_size).to(dt),
            F.one_hot(action_orientation, 4).to(dt),
            F.one_hot(action_spin_type, 3).to(dt),
            F.one_hot(action_piece_type, self.num_piece_types).to(dt),
            F.one_hot(action_lines_cleared, 5).to(dt),
        ], dim=-1)                                                                    # [B, N, action_feat_dim]
        h = self.action_proj(feat)                                                   # [B, N, d]
        if action_afterstate is not None:
            a = action_afterstate
            for proj in self.afterstate_proj:
                h = h + proj(a)
        return h

    def _value_from_state(self, state_h: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.discrete_value_size == 1:
            value = self.value_act(self.value_mlp(state_h)).squeeze(-1)              # [B]
            return {"value": value}
        else:
            value_logit = self.value_mlp(state_h)                                    # [B, V]
            value = torch.softmax(value_logit, dim=1)
            return {"value_logit": value_logit, "value": value}

    def forward(self,
                board: torch.Tensor,
                current_piece: torch.Tensor,
                hold_piece: torch.Tensor,
                has_held: torch.Tensor,
                preview: torch.Tensor,
                was_rotation: torch.Tensor,
                srs_index: torch.Tensor,
                combo: torch.Tensor,
                back_to_back: torch.Tensor,
                pending_garbage: torch.Tensor,
                action_use_hold: torch.Tensor,
                action_lock_x: torch.Tensor,
                action_lock_y: torch.Tensor,
                action_orientation: torch.Tensor,
                action_spin_type: torch.Tensor,
                action_piece_type: torch.Tensor,
                action_lines_cleared: torch.Tensor,
                action_mask: torch.Tensor,
                action_afterstate: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Same shapes/contract as PlacementTransformerNetwork.forward."""
        B = board.shape[0]
        state_h = self._state_h(board, current_piece, hold_piece, has_held, preview,
                                was_rotation, srs_index, combo, back_to_back,
                                pending_garbage)                                      # [B, d]
        action_h = self._action_h(action_use_hold, action_lock_x, action_lock_y,
                                  action_orientation, action_spin_type,
                                  action_piece_type, action_lines_cleared,
                                  action_afterstate)                                  # [B, N, d]
        N = action_h.shape[1]
        ctx = state_h.unsqueeze(1).expand(B, N, self.d_model)                         # [B, N, d]
        pa_in = torch.cat([ctx, action_h], dim=-1)                                    # [B, N, 2d]
        policy_logit = self.policy_mlp(pa_in).squeeze(-1)                             # [B, N]
        policy_logit = policy_logit.masked_fill(action_mask.to(torch.bool), -1e9)
        policy = torch.softmax(policy_logit, dim=1)

        out: Dict[str, torch.Tensor] = {"policy_logit": policy_logit, "policy": policy}
        for k, v in self._value_from_state(state_h).items():
            out[k] = v
        return out


def _smoke_test():
    torch.manual_seed(0)
    B = 4
    N_max = 30
    preview_size = 5
    for board_encoder in ("flat", "conv"):
        net = PlacementMLPNetwork(preview_size=preview_size, board_encoder=board_encoder,
                                  d_model=64, hidden_ratio=2, num_mlp_layers=3)
        net.eval()  # disable dropout so eager and scripted forwards match
        board = torch.zeros(B, 1, 20, 10)
        current = torch.randint(0, 7, (B,))
        hold = torch.randint(0, 8, (B,))
        has_held = torch.randint(0, 2, (B,)).float()
        preview = torch.randint(0, 7, (B, preview_size))
        was_rotation = torch.randint(0, 2, (B,)).float()
        srs_index = torch.randint(0, 7, (B,))
        combo = torch.rand(B)
        b2b = torch.randint(0, 2, (B,)).float()
        garbage = torch.rand(B)
        a_use_hold = torch.randint(0, 2, (B, N_max))
        # full real descriptor range: lock_x in [-2, 8], lock_y in [-9, 18]
        # (anchors outside the visible board; see LOCK_*_OFFSET)
        a_lock_x = torch.randint(-2, 9, (B, N_max))
        a_lock_y = torch.randint(-9, 19, (B, N_max))
        a_orient = torch.randint(0, 4, (B, N_max))
        a_spin = torch.randint(0, 3, (B, N_max))
        a_piece = torch.randint(0, 7, (B, N_max))
        a_clear = torch.randint(0, 5, (B, N_max))
        Ns = torch.tensor([10, 22, 15, 5])
        action_mask = torch.zeros(B, N_max, dtype=torch.bool)
        for i, n in enumerate(Ns.tolist()):
            action_mask[i, n:] = True

        args = (board, current, hold, has_held, preview,
                was_rotation, srs_index, combo, b2b, garbage,
                a_use_hold, a_lock_x, a_lock_y, a_orient, a_spin, a_piece, a_clear,
                action_mask)
        out = net(*args)
        assert out["policy"].shape == (B, N_max)
        assert out["value"].shape == (B, net.discrete_value_size)
        row_sums = out["policy"].sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(B), atol=1e-4), row_sums
        for i, n in enumerate(Ns.tolist()):
            assert out["policy"][i, n:].abs().max().item() < 1e-6
        scripted = torch.jit.script(net)
        out2 = scripted(*args)
        assert torch.allclose(out["policy"], out2["policy"], atol=1e-5)
        print(f"[smoke] {board_encoder}: forward + TorchScript OK, "
              f"policy {tuple(out['policy'].shape)} value {tuple(out['value'].shape)}")


if __name__ == "__main__":
    _smoke_test()
