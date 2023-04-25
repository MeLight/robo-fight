import pickle
import random
from enum import Enum

import numpy as np

from util import np_state_from_game_state

INIT_HP = 50
INIT_ENERGY = 10

MODEL_RANDOM = 'random'


class Moves(Enum):
    PUNCH = 1
    KICK = 2
    # DEF = 3
    REST = 3


MOVES_MAP = {
    Moves.KICK: {"energy": 5, "damage": 4, "title": "Kick"},
    Moves.PUNCH: {"energy": 2, "damage": 2, "title": "Punch"},
    # Moves.DEF: {"energy": 3, "damage": 0.5, "title": "Punch"},
    Moves.REST: {"energy": -5, "damage": 0, "title": "Rest"},
}


class Fight:
    def __init__(self, npc_behavior=MODEL_RANDOM):
        self.player = Fighter("player")
        self.npc = Fighter("npc")
        self.done = False
        self.winner = 0
        self.npc_behavior = npc_behavior
        self.npc_model = None
        if self.npc_behavior != MODEL_RANDOM:
            with open(npc_behavior, 'rb') as f:
                self.npc_model = pickle.load(f)
        else:
            self.npc_model = MODEL_RANDOM

    def get_state(self):
        return {
            "player_hp": self.player.hp,
            "player_energy": self.player.energy,
            "npc_hp": self.npc.hp,
            "npc_energy": self.npc.energy,
            "done": self.done,
            "winner": self.winner
        }

    def step(self, action):
        move = Moves(action)
        success = self.player.make_move(move, self.npc)
        if not success:
            return False
        self.check_victory()
        if self.done:
            return self.get_state()

        self.npc_move()
        self.check_victory()
        return self.get_state()

    def npc_move(self):
        success = False
        attempts = 5
        while not success:
            attempts -= 1
            if attempts == 0:
                print(f"Model {self.npc_behavior} couldn't find a successful move for N attempts. Setting move to REST")
                move = Moves.REST
            else:
                action = model_action(self.npc_model, game_state=self.get_state())
                move = Moves(action)
            success = self.npc.make_move(move, self.player)

    def check_victory(self):
        if self.npc.hp <= 0:
            self.winner = 1
            self.done = True
            return 1

        if self.player.hp <= 0:
            self.winner = -1
            self.done = True
            return -1

        return 0


def model_action(model, game_state):
    if model == MODEL_RANDOM:
        action = random.choice(list(Moves))
    else:
        state = np_state_from_game_state(game_state)
        action_probs = model.predict(np.array([state]))
        action = np.argmax(action_probs[0])
        action += 1
    return action

    # TODO needs to be used by the NPC player when model is provided


class Fighter:
    def __init__(self, fighter_id):
        self.hp = INIT_HP
        self.energy = INIT_ENERGY
        self.fighter_id = fighter_id
        self.last_move = None

    def make_move(self, move_enum, opponent: "Fighter"):
        print(f"{self.fighter_id}: attempts {move_enum}")
        move = MOVES_MAP[move_enum]
        if self.energy - move["energy"] < 0:
            print(f"{self.fighter_id}: fails {move_enum}. Not enough energy")
            return False

        self.last_move = move_enum
        print(f"{self.fighter_id} -> [{move['title']} (dmg:{move['damage']})] -> {opponent.fighter_id} (current hp: {opponent.hp})")
        opponent.take_damage(move["damage"])
        self.energy -= move["energy"]
        # self.energy -= move_energy
        # if self.energy < 0:
        #     self.energy = 0
        return True

    def take_damage(self, dmg):
        print(f"{self.fighter_id} taking dmg {dmg} (current hp: {self.hp})")
        self.hp -= dmg
        if self.hp < 0:
            self.hp = 0
        print(f"{self.fighter_id} hp after dmg: {self.hp}")

