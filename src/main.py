import argparse
import collections
import os
import sys
from pathlib import Path

from tqdm import tqdm

from src.game import Game
from src.gui_tkinter import OthelloGame
from src.players import HumanPlayer, RLPlayer
from src.training.train import main as train


def main():
    app_dir = sys._MEIPASS if getattr(sys, "frozen", False) else str(Path(__file__).parent.parent)

    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gui', action='store_true', default=False, help="Run GUI version of game")
    parser.add_argument('-b', '--hideboard', action='store_true', default=False, help="Hide board during gameplay, using for testing models where AI is playing against AI")
    parser.add_argument('-p1', '--player1', choices=["ai", "human"], type=str, default="human",
                        help="Type of Player 1")
    parser.add_argument('-p2', '--player2', choices=["ai", "human"], type=str, default="ai",
                        help="Type of Player 2")
    parser.add_argument('-w1', '--weights1', type=str, help="Path to weights file for player 1")
    parser.add_argument('-w2', '--weights2', type=str, help="Path to weights file for player 2")
    parser.add_argument('-m', '--mode', choices=["train", "play", "test"], type=str, default="play",
                        help="Mode to run program")
    parser.add_argument('-d', '--difficulty', choices=["easy", "medium", "hard"], type=str, default="medium",
                        help="Difficulty of AI. Sets weights to both players.")

    parser.add_argument('-n', '--n_epochs', type=int, default=20, help="Number of epochs to train model")
    parser.add_argument('-s', '--match_size', type=int, default=20, help="Number of games to play per epoch")
    parser.add_argument('-f', '--discount_factor', type=float, default=0.97, help="Discount factor for RL")
    parser.add_argument('-l', '--net_lr', type=float, default=0.03, help="Learning rate for neural network")

    args = parser.parse_args()
    p1 = args.player1
    p2 = args.player2
    w1 = args.weights1
    w2 = args.weights2
    mode = args.mode

    net_lr = args.net_lr
    discount_factor = args.discount_factor
    match_size = args.match_size
    n_epochs = args.n_epochs

    difficulty = {
        "easy": os.path.join(app_dir, "models", "hard.weights"),
        "medium": os.path.join(app_dir, "models", "medium.weights"),
        "hard": os.path.join(app_dir, "models", "easy.weights")
    }
    w1 = difficulty.get(w1, w1)
    w2 = difficulty.get(w2, w2)

    if args.difficulty:
        w1 = w2 = difficulty[args.difficulty]

    if p1 == "human":
        player_1 = HumanPlayer()
    else:
        player_1 = RLPlayer()
        player_1.policy_net.load(w1)

    if p2 == "human":
        player_2 = HumanPlayer()
    else:
        player_2 = RLPlayer()
        player_2.policy_net.load(w2)

    if mode == "play":
        if args.gui:
            if p1 != "human" and p2 != "ai":
                raise ValueError("GUI only supports human vs AI")
            else:
                game = OthelloGame(nn_weights=w2)
                game.start()
        else:
            game = Game(player_1, False, player_2, False, board_size=8)
            game.run(show_board=True)

        sys.exit(0)

    if mode == "train":
        train(n_epochs, match_size, discount_factor, net_lr)

        sys.exit(0)

    if mode == "test":
        results = collections.defaultdict(lambda: 0)

        game = Game(player_1, False, player_2, False, board_size=8)
        for i in tqdm(range(n_epochs), desc=f"Testing"):
            winner = game.run(show_board=not args.hideboard)
            game.clear_board()
            results[winner] += 1

        for player, wins in results.items():
            print(f"{player} won {wins} times")


if __name__ == '__main__':
    print(
        "GUI has limited functionality. Specifically, it only supports human vs AI.\n"
        "Please use the console version for full functionality: AI vs AI, AI vs Human, Human vs Human"
    )
    main()
