import itertools
import sys
from copy import deepcopy

import numpy as np
from colorama import init, Fore, Back, Style

init(autoreset=True)


class Tile:
    BLACK = 1
    WHITE = -1
    EMPTY = 0

    @classmethod
    def get_enemy(cls, tile: int):
        if tile == cls.WHITE:
            return cls.BLACK
        elif tile == cls.BLACK:
            return cls.WHITE
        else:
            raise AssertionError("Invalid tile")


class Board:
    def __init__(self, board_size=8):
        self.board_size = board_size
        self.board = np.full((board_size, board_size), Tile.EMPTY, int)

        self.init_state()
        self.score = self.calculate_score()

    def init_state(self):
        els_to_center = self.board_size // 2
        self.board[els_to_center - 1][els_to_center - 1] = Tile.BLACK
        self.board[els_to_center][els_to_center] = Tile.BLACK
        self.board[els_to_center][els_to_center - 1] = Tile.WHITE
        self.board[els_to_center - 1][els_to_center] = Tile.WHITE

    def calculate_score(self):
        self.score = {
            Tile.BLACK: np.count_nonzero(self.board == Tile.BLACK),
            Tile.WHITE: np.count_nonzero(self.board == Tile.WHITE),
            Tile.EMPTY: np.count_nonzero(self.board == Tile.EMPTY)
        }
        return self.score

    def get_state(self):
        return self.board

    def is_pos_on_board(self, x, y):
        try:
            _ = self.board[y, x]
            return True
        except IndexError:
            return False

    def is_pos_empty(self, x, y):
        return self.board[y, x] == Tile.EMPTY

    def is_pos_tile(self, x, y, tile: int):
        return self.board[y, x] == tile

    def is_valid_move(self, x: int, y: int) -> bool:
        """
        Returns False if the player's move on space x and y is invalid.  If it is a valid move, returns True
        """
        if self.is_pos_on_board(x, y) and self.is_pos_empty(x, y):
            return True
        else:
            return False

    def settile(self, tile: int, xstart: int, ystart: int):
        # temporarily set the tile on the board.
        tiles_to_flip = []
        enemy_tile = Tile.get_enemy(tile)
        backup_board = deepcopy(self.board)
        self.board[ystart, xstart] = tile
        single_directions = (0, 1, -1)
        directions = list(itertools.product(single_directions,
                                            repeat=2))  # loop through all directions around flipped tile

        for xdirection, ydirection in directions:
            x, y = xstart, ystart
            x += xdirection  # first step in the direction
            y += ydirection  # first step in the direction

            if self.is_pos_on_board(x, y) and self.is_pos_tile(x, y, enemy_tile):
                # There is a piece belonging to the other player next to our piece.
                x += xdirection
                y += ydirection

                if not self.is_pos_on_board(x, y):
                    continue

                while is_on_board := self.is_pos_on_board(x, y) and self.is_pos_tile(x, y,
                                                                                     enemy_tile):
                    x += xdirection
                    y += ydirection

                if not is_on_board:
                    continue
                elif self.is_pos_tile(x, y, tile):
                    # There are pieces to flip over. Go in the reverse direction until we reach the original space, noting all the tilesalong the way.
                    while True:
                        x -= xdirection
                        y -= ydirection
                        if x == xstart and y == ystart:
                            break
                        tiles_to_flip.append([x, y])
                        self.board[y, x] = tile

        if not tiles_to_flip:
            self.board = backup_board
            return False
        else:
            return True

    def update_board(self, tile, y, x):
        result = self.is_valid_move(x, y)

        if result:
            is_set = self.settile(tile, x, y)
            self.calculate_score()
            return is_set
        else:
            return False

    def print_board(self):
        """
        Print board to terminal
        """

        def get_item(item):
            if item == Tile.BLACK:
                return Fore.WHITE + "|" + Fore.BLUE + "O"
            elif item == Tile.WHITE:
                return Fore.WHITE + "|" + Fore.RED + "O"
            else:
                return Fore.WHITE + "| "

        def get_row(row):
            return "".join(map(get_item, row))

        print("\t" + Back.BLACK + f"{'BOARD':^18}")
        print(
            "\t" + Back.BLACK + Fore.WHITE + f"  |{'|'.join(map(str, range(1, self.board_size + 1)))}")
        for i in range(self.board_size):
            print("\t" + Back.BLACK + Fore.WHITE + f"{i + 1:<2}{get_row(self.board[i]):<2}")
            sys.stdout.write(Style.RESET_ALL)
