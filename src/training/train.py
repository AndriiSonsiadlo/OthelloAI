from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt

from src.game import Game
from src.players import RLPlayer

plt.ion()

board_size = 8
match_size = 20
n_epochs = 20

player = RLPlayer(0.99, 0.03, board_size=board_size)
rp = RLPlayer(0, 0, board_size=board_size)

player_wins = []
for e in range(1, n_epochs + 1):
    print(f"Epoch: {e}")

    player.wins = 0
    # Anneal the exploration rate
    player.epsilon = (np.exp(-0.017 * e) + 0.11) / 1.1
    player_gameplay_history = []

    # number of games to play for one epoch
    for _ in range(match_size):
        # print("Game: %d"%g)
        player.play_history = []

        # Initialize a new game
        g = Game(player_1=player, log_history_1=True, player_2=rp, log_history_2=False, board_size=board_size)
        g.run()
        # print(player.play_history)

        final_score = list(g.get_score().items())
        final_score.sort()
        ttl = sum(map(lambda x: x[1], final_score))
        # print(ttl)

        # Only deal with 1 of the players (The one we're updating the weights for)
        # player_score = int(final_score[0][1]/ttl >= 0.5)
        player_score = (final_score[0][1] / ttl - 0.5) * 2
        player.wins += player_score > 0
        # print(player_score)
        player_gameplay_history.append((player.play_history, player_score))

    print(player.epsilon, player.wins)
    player_wins.append(player.wins)
    for game, score in player_gameplay_history:
        player.play_history = game
        player.update_weights(score)

print("Wins:", sum(player_wins))
dest_path = "./models"

filename = f"{datetime.now().strftime('%d%H%M%S')}-{n_epochs}-{match_size}-{board_size}-best-linear-0.03"
player.policy_net.save(f"{dest_path}/{filename}.weights")
with open(f"{dest_path}/{filename}.csv", "w") as f:
    f.write("\n".join(map(str, player_wins)))
print("Saved:", filename)

plt.plot(player_wins)
plt.draw()
plt.ioff()
plt.show()
