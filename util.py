import numpy as np


def np_state_from_game_state(game_state):
    return np.array([game_state["player_hp"], game_state["npc_hp"]])