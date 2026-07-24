import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


# --- Lock-position coordinate range -----------------------------------------
#
# PlacementActionDescriptor.lock_x/lock_y are the piece's 4x4 bounding-box ANCHOR
# in playfield coords (engine x/y minus BOARD_LEFT=3 / BOARD_TOP=9), NOT an
# occupied cell. Because a shape's left column / top row can be empty, and because
# the engine board keeps 9 hidden buffer rows above the visible board, the anchor
# legitimately falls outside [0, W-1] x [0, H-1].
#
# Enumerating engine/tetris.hpp `pieces[]` against the wall bounds, plus
# placement_search.hpp canonicalizePlacement's (lock_y+1 / lock_x-1) shifts:
#
#     lock_x in [-2, 8]    (min: vertical I hugging the left wall)
#     lock_y in [-9, 18]   (min: piece locked in the hidden spawn buffer)
#
# So the position table is OFFSET and WIDENED rather than clamped -- clamping to
# [0, W-1] x [0, H-1] would alias lock_x in {-2,-1,0} onto one embedding and
# collapse all 9 buffer rows onto visible row 0, making high placements and
# left-wall placements indistinguishable to the network.
#
# Index = lock + OFFSET; table spans OFFSET + board_dim entries (1 spare each).
LOCK_X_OFFSET = 2  # covers lock_x in [-2, board_width  - 1 + 2]
LOCK_Y_OFFSET = 9  # == engine::BOARD_TOP (hidden buffer height)


def lock_grid_size(board_height: int, board_width: int):
    """(grid_h, grid_w) of the lock-position table, indexed by lock + offset."""
    return board_height + LOCK_Y_OFFSET, board_width + LOCK_X_OFFSET


class BoardPatchEmbed(nn.Module):
    """[B, C, H, W] -> [B, n_patches, d_model] with learnable 2D pos embed."""

    def __init__(self, board_channels: int, board_height: int, board_width: int,
                 patch_size: int, d_model: int):
        super().__init__()
        assert board_height % patch_size == 0 and board_width % patch_size == 0, \
            f"board {board_height}x{board_width} must divide evenly by patch {patch_size}"
        self.patch_rows = board_height // patch_size
        self.patch_cols = board_width // patch_size
        self.n_patches = self.patch_rows * self.patch_cols
        self.proj = nn.Conv2d(board_channels, d_model,
                              kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        self.type_embed = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.type_embed, std=0.02)

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        # board: [B, C, H, W]
        x = self.proj(board)                 # [B, d, pr, pc]
        x = x.flatten(2).transpose(1, 2)     # [B, n_patches, d]
        x = x + self.pos_embed + self.type_embed
        return x


class PieceQueueEmbed(nn.Module):
    """7 slots: current, hold, preview_1..preview_5 -> [B, 7, d_model]."""

    def __init__(self, preview_size: int, d_model: int):
        super().__init__()
        self.num_piece_types_with_none = 8  # 7 real + 1 NONE sentinel (index=7)
        self.num_roles = 3                  # 0=current, 1=hold, 2=preview
        self.preview_size = preview_size
        self.queue_len = 2 + preview_size
        self.piece_embed = nn.Embedding(self.num_piece_types_with_none, d_model)
        self.slot_pos_embed = nn.Parameter(torch.zeros(1, self.queue_len, d_model))
        self.role_embed = nn.Embedding(self.num_roles, d_model)
        self.has_held_embed = nn.Parameter(torch.zeros(1, 1, d_model))  # added only to hold slot when has_held
        self.type_embed = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.slot_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.has_held_embed, std=0.02)
        nn.init.trunc_normal_(self.type_embed, std=0.02)
        # Precompute role indices for 7 slots: [0, 1, 2, 2, 2, 2, 2]
        role_ids = torch.zeros(self.queue_len, dtype=torch.long)
        role_ids[0] = 0
        role_ids[1] = 1
        role_ids[2:] = 2
        self.register_buffer("role_ids", role_ids)

    def forward(self, current: torch.Tensor, hold: torch.Tensor,
                has_held: torch.Tensor, preview: torch.Tensor) -> torch.Tensor:
        # current: [B] long, hold: [B] long (NONE encoded as 7),
        # has_held: [B] float 0/1, preview: [B, preview_size] long
        B = current.shape[0]
        queue = torch.cat([current.unsqueeze(1), hold.unsqueeze(1), preview], dim=1)  # [B, 7]
        x = self.piece_embed(queue)                                                   # [B, 7, d]
        roles = self.role_embed(self.role_ids).unsqueeze(0)                           # [1, 7, d]
        x = x + roles + self.slot_pos_embed + self.type_embed
        # Inject has_held onto hold slot only (index 1)
        has_held_expanded = has_held.view(B, 1, 1) * self.has_held_embed              # [B, 1, d]
        x[:, 1:2, :] = x[:, 1:2, :] + has_held_expanded
        return x


class PieceStateEmbed(nn.Module):
    """[was_rotation(1), srs_onehot(7)] -> [B, 1, d_model]."""

    def __init__(self, d_model: int):
        super().__init__()
        self.srs_size = 7  # -1..5 shifted to 0..6
        self.proj = nn.Linear(1 + self.srs_size, d_model)
        self.type_embed = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.type_embed, std=0.02)

    def forward(self, was_rotation: torch.Tensor, srs_index: torch.Tensor) -> torch.Tensor:
        # was_rotation [B] float, srs_index [B] long (already shifted 0..6)
        srs_onehot = F.one_hot(srs_index, num_classes=self.srs_size).float()         # [B, 7]
        x = torch.cat([was_rotation.unsqueeze(1), srs_onehot], dim=1)
        x = self.proj(x).unsqueeze(1) + self.type_embed                              # [B, 1, d]
        return x


class MetaEmbed(nn.Module):
    """[combo_scaled, b2b, garbage_scaled] -> [B, 1, d_model]."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(3, d_model)
        self.type_embed = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.type_embed, std=0.02)

    def forward(self, combo: torch.Tensor, back_to_back: torch.Tensor,
                pending_garbage: torch.Tensor) -> torch.Tensor:
        # all [B] float, already normalized on the C++ side
        x = torch.stack([combo, back_to_back, pending_garbage], dim=1)               # [B, 3]
        x = self.proj(x).unsqueeze(1) + self.type_embed                              # [B, 1, d]
        return x


class ActionTokenEmbed(nn.Module):
    """N placements -> [B, N, d_model] with 2D pos embed indexed by lock_xy."""

    def __init__(self, board_height: int, board_width: int,
                 num_piece_types: int, d_model: int,
                 num_orientations: int = 4, num_spin_types: int = 3,
                 max_lines_cleared: int = 5):
        super().__init__()
        self.board_height = board_height
        self.board_width = board_width
        # Lock anchors live outside the visible board; see LOCK_*_OFFSET above.
        self.lock_grid_h, self.lock_grid_w = lock_grid_size(board_height, board_width)
        self.lock_x_offset = LOCK_X_OFFSET
        self.lock_y_offset = LOCK_Y_OFFSET
        # Per-field embeddings
        self.use_hold_embed = nn.Embedding(2, d_model)
        self.orient_embed = nn.Embedding(num_orientations, d_model)
        self.spin_embed = nn.Embedding(num_spin_types, d_model)
        self.piece_embed = nn.Embedding(num_piece_types, d_model)
        self.lines_cleared_embed = nn.Embedding(max_lines_cleared, d_model)
        # 2D lock position embedding (flattened)
        self.lock_pos_embed = nn.Parameter(torch.zeros(1, self.lock_grid_h * self.lock_grid_w, d_model))
        self.type_embed = nn.Parameter(torch.zeros(1, 1, d_model))
        self.mix = nn.Linear(d_model * 5, d_model)
        nn.init.trunc_normal_(self.lock_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.type_embed, std=0.02)

    def forward(self, use_hold: torch.Tensor, lock_x: torch.Tensor, lock_y: torch.Tensor,
                orientation: torch.Tensor, spin_type: torch.Tensor,
                piece_type: torch.Tensor, lines_cleared: torch.Tensor) -> torch.Tensor:
        # all shape [B, N] long
        feats = torch.cat([
            self.use_hold_embed(use_hold),
            self.orient_embed(orientation),
            self.spin_embed(spin_type),
            self.piece_embed(piece_type),
            self.lines_cleared_embed(lines_cleared),
        ], dim=-1)                                                                    # [B, N, 5d]
        x = self.mix(feats)                                                           # [B, N, d]
        # Shift the anchor into the widened grid. The clamps are a safety net for
        # padded slots / unexpected descriptors; real placements never trigger them.
        iy = (lock_y + self.lock_y_offset).clamp(0, self.lock_grid_h - 1)
        ix = (lock_x + self.lock_x_offset).clamp(0, self.lock_grid_w - 1)
        flat_idx = iy * self.lock_grid_w + ix                                         # [B, N]
        pos = self.lock_pos_embed.squeeze(0)[flat_idx]                                # [B, N, d]
        x = x + pos + self.type_embed
        return x


class PlacementTransformerNetwork(nn.Module):
    """Placement-based Tetris network with patch + action tokens and a
    [VALUE] query token. Policy is per-action (variable N); value comes
    from the VALUE token.
    """

    def __init__(self,
                 game_name: str = "moderntetris_placement",
                 board_channels: int = 1,
                 board_height: int = 20,
                 board_width: int = 10,
                 patch_size: int = 5,
                 num_piece_types: int = 7,
                 preview_size: int = 5,
                 d_model: int = 192,
                 n_heads: int = 6,
                 n_layers: int = 4,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1,
                 num_value_hidden_channels: int = 256,
                 discrete_value_size: int = 601,
                 winloss_value_size: int = 0):
        super().__init__()
        self.game_name = game_name
        self.board_channels = board_channels
        self.board_height = board_height
        self.board_width = board_width
        self.patch_size = patch_size
        self.num_piece_types = num_piece_types
        self.preview_size = preview_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.discrete_value_size = discrete_value_size
        # Two-player mode adds a separate zero-sum win/loss head (distributional
        # over {lose, draw, win}); 0 disables it (single-player).
        self.winloss_value_size = winloss_value_size

        self.patch_embed = BoardPatchEmbed(board_channels, board_height, board_width,
                                           patch_size, d_model)
        self.queue_embed = PieceQueueEmbed(preview_size, d_model)
        self.piece_state_embed = PieceStateEmbed(d_model)
        self.meta_embed = MetaEmbed(d_model)
        self.action_embed = ActionTokenEmbed(board_height, board_width,
                                             num_piece_types, d_model)
        self.value_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.value_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Policy head: context-aware path over post-encoder hidden states.
        self.policy_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        # Value head on VALUE token
        if discrete_value_size == 1:
            self.value_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, num_value_hidden_channels),
                nn.GELU(),
                nn.Linear(num_value_hidden_channels, 1),
                nn.Tanh(),
            )
        else:
            self.value_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, num_value_hidden_channels),
                nn.GELU(),
                nn.Linear(num_value_hidden_channels, discrete_value_size),
            )

        # Win/loss head (two-player only): distributional over winloss_value_size
        # classes (typically 3 = {lose, draw, win}) on the same VALUE token. The
        # module is always constructed (TorchScript needs a concrete submodule),
        # but its output is only emitted when winloss_value_size > 0; in
        # single-player it is an unused, untrained head.
        self._winloss_built_size = winloss_value_size if winloss_value_size > 0 else 3
        self.winloss_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_value_hidden_channels),
            nn.GELU(),
            nn.Linear(num_value_hidden_channels, self._winloss_built_size),
        )

    @torch.jit.export
    def get_type_name(self) -> str:
        return "placement_transformer"

    @torch.jit.export
    def get_winloss_value_size(self) -> int:
        return self.winloss_value_size

    @torch.jit.export
    def get_game_name(self) -> str:
        return self.game_name

    @torch.jit.export
    def get_discrete_value_size(self) -> int:
        return self.discrete_value_size

    @torch.jit.export
    def get_d_model(self) -> int:
        return self.d_model

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
                action_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Shapes:
          board               [B, C_board, H, W]  float
          current_piece       [B]                 long (0..6, NONE=7)
          hold_piece          [B]                 long (0..6, NONE=7)
          has_held            [B]                 float (0/1)
          preview             [B, preview_size]   long (0..6, NONE=7)
          was_rotation        [B]                 float (0/1)
          srs_index           [B]                 long (shifted -1..5 -> 0..6)
          combo               [B]                 float (normalized)
          back_to_back        [B]                 float (0/1)
          pending_garbage     [B]                 float (normalized)
          action_* (all):     [B, N_max]          long
          action_mask:        [B, N_max]          bool/uint8 (True = padding, ignored)
        """
        B = board.shape[0]

        patches = self.patch_embed(board)                                             # [B, n_patches, d]
        queue = self.queue_embed(current_piece, hold_piece, has_held, preview)        # [B, 7, d]
        piece_state = self.piece_state_embed(was_rotation, srs_index)                 # [B, 1, d]
        meta = self.meta_embed(combo, back_to_back, pending_garbage)                  # [B, 1, d]
        value_tok = self.value_token.expand(B, -1, -1)                                # [B, 1, d]
        actions_pre = self.action_embed(action_use_hold, action_lock_x, action_lock_y,
                                        action_orientation, action_spin_type,
                                        action_piece_type, action_lines_cleared)      # [B, N_max, d]

        seq = torch.cat([patches, queue, piece_state, meta, value_tok, actions_pre], dim=1)

        # key_padding_mask: True = ignore. Only action tokens can be padded.
        n_fixed = patches.shape[1] + queue.shape[1] + 2 + 1  # patches + queue + piece_state+meta + value
        fixed_mask = torch.zeros(B, n_fixed, dtype=torch.bool, device=board.device)
        full_mask = torch.cat([fixed_mask, action_mask.to(torch.bool)], dim=1)

        h = self.encoder(seq, src_key_padding_mask=full_mask)                         # [B, L, d]

        # Slice outputs
        value_idx = n_fixed - 1
        h_value = h[:, value_idx, :]                                                  # [B, d]
        h_actions = h[:, n_fixed:, :]                                                 # [B, N_max, d]

        # Policy logits: [B, N_max] over the post-encoder action hidden states.
        policy_logit = self.policy_head(h_actions).squeeze(-1)
        # Mask padded positions to large-negative so softmax assigns zero probability.
        # (torch.finfo is not available in TorchScript on some versions.)
        policy_logit = policy_logit.masked_fill(action_mask.to(torch.bool), -1e9)
        policy = torch.softmax(policy_logit, dim=1)

        if self.discrete_value_size == 1:
            value = self.value_head(h_value).squeeze(-1)                              # [B]
            out = {"policy_logit": policy_logit,
                   "policy": policy,
                   "value": value}
        else:
            value_logit = self.value_head(h_value)                                    # [B, V]
            value = torch.softmax(value_logit, dim=1)
            out = {"policy_logit": policy_logit,
                   "policy": policy,
                   "value_logit": value_logit,
                   "value": value}

        if self.winloss_value_size > 0:
            winloss_logit = self.winloss_head(h_value)                                # [B, Vw]
            out["winloss_logit"] = winloss_logit
            out["winloss"] = torch.softmax(winloss_logit, dim=1)
        return out


def _smoke_test():
    torch.manual_seed(0)
    B = 4
    N_max = 30
    preview_size = 5
    net = PlacementTransformerNetwork(preview_size=preview_size)
    net.eval()  # disable dropout so eager and scripted forwards match bitwise
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
    a_lock_x = torch.randint(-2, 9, (B, N_max))
    a_lock_y = torch.randint(-9, 19, (B, N_max))
    a_orient = torch.randint(0, 4, (B, N_max))
    a_spin = torch.randint(0, 3, (B, N_max))
    a_piece = torch.randint(0, 7, (B, N_max))
    a_clear = torch.randint(0, 5, (B, N_max))
    # varying N per sample: mask out tail randomly
    Ns = torch.tensor([10, 22, 15, 5])
    action_mask = torch.zeros(B, N_max, dtype=torch.bool)
    for i, n in enumerate(Ns.tolist()):
        action_mask[i, n:] = True

    out = net(board, current, hold, has_held, preview,
              was_rotation, srs_index,
              combo, b2b, garbage,
              a_use_hold, a_lock_x, a_lock_y, a_orient, a_spin, a_piece, a_clear,
              action_mask)
    assert out["policy"].shape == (B, N_max)
    assert out["value"].shape == (B, net.discrete_value_size)
    # Softmax should sum to 1 over unmasked positions
    row_sums = out["policy"].sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(B), atol=1e-4), row_sums
    # Padded positions should be exactly 0 probability
    for i, n in enumerate(Ns.tolist()):
        assert out["policy"][i, n:].abs().max().item() < 1e-6, out["policy"][i, n:]
    print("[smoke] forward OK, policy shape", out["policy"].shape,
          "value shape", out["value"].shape)

    # TorchScript export
    scripted = torch.jit.script(net)
    out2 = scripted(board, current, hold, has_held, preview,
                    was_rotation, srs_index,
                    combo, b2b, garbage,
                    a_use_hold, a_lock_x, a_lock_y, a_orient, a_spin, a_piece, a_clear,
                    action_mask)
    assert torch.allclose(out["policy"], out2["policy"], atol=1e-5)
    print("[smoke] TorchScript OK")


if __name__ == "__main__":
    _smoke_test()
