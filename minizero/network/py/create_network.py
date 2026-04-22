from .alphazero_network import AlphaZeroNetwork
from .muzero_network import MuZeroNetwork
from .muzero_atari_network import MuZeroAtariNetwork
from .placement_transformer_network import PlacementTransformerNetwork


def create_network(game_name="tietactoe",
                   num_input_channels=4,
                   input_channel_height=3,
                   input_channel_width=3,
                   num_hidden_channels=16,
                   hidden_channel_height=3,
                   hidden_channel_width=3,
                   num_action_feature_channels=1,
                   num_blocks=1,
                   action_size=9,
                   num_value_hidden_channels=256,
                   discrete_value_size=601,
                   network_type_name="alphazero",
                   placement_config=None):

    network = None
    if network_type_name == "alphazero":
        network = AlphaZeroNetwork(game_name,
                                   num_input_channels,
                                   input_channel_height,
                                   input_channel_width,
                                   num_hidden_channels,
                                   hidden_channel_height,
                                   hidden_channel_width,
                                   num_blocks,
                                   action_size,
                                   num_value_hidden_channels,
                                   discrete_value_size)
    elif network_type_name == "placement_transformer":
        cfg = placement_config or {}
        network = PlacementTransformerNetwork(
            game_name=game_name,
            board_channels=1,
            board_height=input_channel_height,
            board_width=input_channel_width,
            patch_size=5,
            num_piece_types=7,
            preview_size=cfg.get("preview_size", 5),
            d_model=cfg.get("d_model", 192),
            n_heads=cfg.get("n_heads", 6),
            n_layers=cfg.get("n_layers", 4),
            mlp_ratio=cfg.get("mlp_ratio", 4),
            dropout=cfg.get("dropout", 0.1),
            num_value_hidden_channels=num_value_hidden_channels,
            discrete_value_size=discrete_value_size,
        )
    elif network_type_name == "muzero":
        if "atari" in game_name:
            network = MuZeroAtariNetwork(game_name,
                                         num_input_channels,
                                         input_channel_height,
                                         input_channel_width,
                                         num_hidden_channels,
                                         hidden_channel_height,
                                         hidden_channel_width,
                                         num_action_feature_channels,
                                         num_blocks,
                                         action_size,
                                         num_value_hidden_channels,
                                         discrete_value_size)
        else:
            network = MuZeroNetwork(game_name,
                                    num_input_channels,
                                    input_channel_height,
                                    input_channel_width,
                                    num_hidden_channels,
                                    hidden_channel_height,
                                    hidden_channel_width,
                                    num_action_feature_channels,
                                    num_blocks,
                                    action_size,
                                    num_value_hidden_channels,
                                    discrete_value_size)
    else:
        assert False

    return network
