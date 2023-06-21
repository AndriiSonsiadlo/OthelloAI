from dataclasses import dataclass

from colorama import Back, Fore

import board
from src.players import PlayerModel


@dataclass
class Player:
    name: str
    model: PlayerModel
    id: int
    log_history: bool = False


class Game:
    def __init__(self, player_1, log_history_1, player_2, log_history_2, board_size=8):
        self.board = board.Board(board_size=board_size)
        self.players = [
            Player(name="Player 1", model=player_1, id=-1, log_history=log_history_1),
            Player(name="Player 2", model=player_2, id=1, log_history=log_history_2)
        ]

    def get_score(self):
        return self.board.get_score()

    def run(self, show_board=False):
        if show_board:
            print("\t" + Back.WHITE + Fore.BLACK + f"{'Game Start':^18}")
            print("\t" + Back.MAGENTA + Fore.BLACK + f"{'Player 1: ' + str(self.get_score()[self.players[0].id]):<18}")
            print("\t" + Back.MAGENTA + Fore.BLACK + f"{'Player 2: ' + str(self.get_score()[self.players[1].id]):<18}")
            self.board.print_board()

        move_number = 1
        n_passed = 0
        # Run until both players have passed
        while n_passed < 2:
            if show_board:
                print("\n\t" + Back.WHITE + Fore.BLACK + f"{'Move ' + str(move_number):^18}")
                print(
                    "\t" + Back.MAGENTA + Fore.BLACK + f"{'Player 1: ' + str(self.get_score()[self.players[0].id]):<18}")
                print("\t" + Back.CYAN + Fore.BLACK + f"{'Player 2: ' + str(self.get_score()[self.players[1].id]):<18}")

            n_passed = 0

            for i, player in enumerate(self.players, start=1):
                print("\t" + Back.YELLOW + Fore.BLACK + f"{str(i) + ' player`s turn':<18}") if show_board else None
                # Pass the player a function it can use to make a move
                # Player ids [-1, 1] are used to indicate which player is making the move
                did_move = player.model.play(
                    lambda r, c: self.board.update_board(player.id, r, c), self.board,
                    self.board.get_state(), player.id, log_history=player.log_history
                )

                if show_board:
                    self.board.print_board()

                if not did_move:
                    n_passed += 1
                if not all(self.board.get_score().values()):
                    break

            move_number += 1
            if not all(self.board.get_score().values()):
                break

        if show_board:
            player1_score, player2_score = list(map(str, self.get_score().values()))
            print("\n\n\t" + Back.WHITE + Fore.BLACK + f"{'Game Over':^18}")
            print("\t" + Back.YELLOW + Fore.BLACK + f"{'Winner: ' + ('Player 1' if player1_score > player2_score else 'Player 2'):<18}")
            print("\t" + Back.MAGENTA + Fore.BLACK + f"{'Player 1: ' + player1_score:<18}")
            print("\t" + Back.MAGENTA + Fore.BLACK + f"{'Player 2: ' + player2_score:<18}")
