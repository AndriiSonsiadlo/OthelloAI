from src.game import Game
from src.players import HumanPlayer, RLPlayer

h = HumanPlayer()

ai = RLPlayer(0)
ai.policy_net.load("./models/21140501-1-1-8-best-linear-0.03.weights")

ai2 = RLPlayer(1)
ai2.policy_net.load("./models/easy.weights")

g = Game(ai, False, ai2, False, board_size=8)
g.run(show_board=True)
