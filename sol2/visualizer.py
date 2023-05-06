#!/usr/bin/env python3
from game import Game
from players import *

h = HumanPlayer()
ai = RLPlayer(1,1,1)
ai.policy_net.load("best-linear-0.03.weights")

g = Game()
g.addPlayer(ai)
g.addPlayer(ai)
g.run(True)
