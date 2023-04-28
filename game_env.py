import numpy as np

from game_fight import Moves, Fight
from util import np_state_from_game_state

ILLEGAL_MOVE_REWARD = -10

class FightGameEnv:
    def __init__(self):
        self.fight = None
        self.previous_state = None

    def reset(self):
        self.fight = Fight()
        self.previous_state = self.fight.get_state()
        return self.game_state()

    def step(self, action):
        action += 1     # we offset to be usable with the Moves enum
        self.previous_state = self.fight.get_state()
        success = self.fight.step(Moves(action))
        if not success:
            return self.game_state(), ILLEGAL_MOVE_REWARD, self.fight.done
        if self.fight.done:
            print(self.fight.get_state())
        return self.game_state(), self.reward_from_state(), self.fight.done

    def reward_from_state(self):
        state = self.fight.get_state()
        p_state = self.previous_state
        if state["player_hp"] == 0:
            return -1

        if state["npc_hp"] == 0:
            return 1

        print(f"current state: {state}, previous state: {p_state=}")
        hp_delta = state["player_hp"] - state["npc_hp"]
        p_hp_delta = p_state["player_hp"] - p_state["npc_hp"]
        reward = (hp_delta - p_hp_delta)/10.0 + (0.1 if hp_delta > 0 else 0)
        return reward

    def game_state(self):
        return np_state_from_game_state(self.fight.get_state())

