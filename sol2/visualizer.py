#!/usr/bin/env python3
from game import Game
from players import *

h = HumanPlayer()
ai = RLPlayer(1, 1, 1)
ai2 = RLPlayer(0.2, 1, 1)
# ai.policy_net.load("11094007-18-10-10-best-linear-0.03.weights")
ai2.policy_net.load("best-linear-0.03.weights")

g = Game(ai, True, h, False, board_size=8)
g.run(show_board=True)
