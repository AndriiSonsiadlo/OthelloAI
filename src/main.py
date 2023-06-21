import argparse

from src.gui_tkinter import OthelloGame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gui', action='store_true', default=True)
    parser.add_argument('-p1', '--player1', choices=["ai", "human"], type=str, default="human")
    parser.add_argument('-p2', '--player2', choices=["ai", "human"], type=str, default="ai")
    parser.add_argument('-w1', '--weights', type=str)
    parser.add_argument('-w2', '--weights', type=str)

    args = parser.parse_args()
    p1 = args.player1
    p2 = args.player2
    w1 = args.weights
    w2 = args.weights

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

    exit(0)


if __name__ == '__main__':
    main()
