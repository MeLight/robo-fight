from game_fight import Fight, MOVES_MAP, Moves

done = False

fight = Fight()
while not done:
    game_state = fight.step(Moves.KICK)
    done = game_state["done"]
