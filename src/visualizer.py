#!/usr/bin/env python3
from game import Game
from players import *

h = HumanPlayer()
ai = RLPlayer()
ai.policy_net.load("8size-best-linear-0.03.weights")
ai2 = RLPlayer()
ai2.policy_net.load("11115844-20-10-8-best-linear-0.03.weights")

g = Game(ai, False, ai2, False, board_size=8)
g.run(show_board=True)
