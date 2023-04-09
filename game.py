INIT_HP = 100
INIT_ENERGY = 100
KICK_DMG = 10
KICK_ENERGY = 10
PUNCH_DMG = 5
PUNCH_ENERGY = 5
BLOCK_ENERGY = 7


class Fight():
    def __init__(self):
        self.player = Fighter()
        self.npc = Fighter()

           
class Fighter():
    def __init__(self):
        self.hp = INIT_HP
        self.energy = INIT_ENERGY
