import argparse
import os
import sys
from pathlib import Path

from src.gui_tkinter import OthelloGame
from src.training.train import main as train

def main():
    app_dir = sys._MEIPASS if getattr(sys, "frozen", False) else str(Path(__file__).parent.parent)

    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gui', action='store_true', default=False)
    parser.add_argument('-p1', '--player1', choices=["ai", "human"], type=str, default="human")
    parser.add_argument('-p2', '--player2', choices=["ai", "human"], type=str, default="ai")
    parser.add_argument('-w1', '--weights1', type=str)
    parser.add_argument('-w2', '--weights2', type=str)
    parser.add_argument('-m', '--mode', choices=["train", "play"], type=str, default="play")
    parser.add_argument('-d', '--difficulty', choices=["easy", "medium", "hard"], type=str, default="medium")

    args = parser.parse_args()
    p1 = args.player1
    p2 = args.player2
    w1 = args.weights1
    w2 = args.weights2
    mode = args.mode

    if mode == "play":
        difficulty = {
            "easy": os.path.join(app_dir, "models", "easy.weights"),
            "medium": os.path.join(app_dir, "models", "medium.weights"),
            "hard": os.path.join(app_dir, "models", "hard.weights")
        }

        if w1 == "easy":
            w1 = difficulty["easy"]
        elif w1 == "medium":
            w1 = difficulty["medium"]
        elif w1 == "hard":
            w1 = difficulty["hard"]

        if w2 == "easy":
            w2 = difficulty["easy"]
        elif w2 == "medium":
            w2 = difficulty["medium"]
        elif w2 == "hard":
            w2 = difficulty["hard"]

        if args.difficulty and (p1 == "human" and p2 == "ai" or p1 == "ai" and p2 == "human") and not w2 and not w1:
            if args.difficulty == "easy":
                w1 = w2 = difficulty["easy"]
            elif args.difficulty == "medium":
                w1 = w2 = difficulty["medium"]
            elif args.difficulty == "hard":
                w1 = w2 = difficulty["hard"]

        if args.gui:
            if p1 != "human" and p2 != "ai":
                raise ValueError("GUI only supports human vs AI")
            else:
                game = OthelloGame(nn_weights=w2)
                game.start()
        else:
            from src.game import Game
            from src.players import HumanPlayer, RLPlayer

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

            game = Game(player_1, False, player_2, False, board_size=8)
            game.run(show_board=True)

        sys.exit(0)
    if mode == "train":
        train()


if __name__ == '__main__':
    main()
