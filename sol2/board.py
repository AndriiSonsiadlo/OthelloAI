import sys
from collections import defaultdict

import numpy as np

from colorama import init, Fore, Back, Style

init(autoreset=True)


class Board:
    BLACK = 1
    WHITE = -1

    def __init__(self, board_size=8):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), int)
        self.init_state()

        self.remaining_squares = board_size * board_size - 4
        self.score = {Board.BLACK: 2, Board.WHITE: 2}

    def init_state(self):
        self.board[3][3] = Board.BLACK
        self.board[4][4] = Board.BLACK
        self.board[4][3] = Board.WHITE
        self.board[3][4] = Board.WHITE

    def get_score(self):
        return self.score

    def get_state(self):
        return self.board

    def is_on_board(self, x, y):
        """
        Returns True if the coordinates are located on the board.
        """
        return 0 <= x < self.board_size and 0 <= y < self.board_size

    def update_board(self, tile, row, col):
        """
        @param int tile
            either 1 or -1
                 1 for player 1 (black)
                -1 for player 2 (white)
        @param int row
            0-board_size which row
        @param int col
            0-board_size which col
        @return bool
            true if valid
            false if invalid move - doesn't update board
        """
        result = self.is_valid_move(tile, row, col)
        if result:
            # Flip the disks
            self.board[row][col] = tile
            for row in result:
                self.board[row[0]][row[1]] = tile

            # Update the players' scores
            self.score[tile] += len(result) + 1

            # The gross expression is a mapping for -1 -> 1 and 1 -> -1
            # Rescales the range to [0,1] then mod 2 then rescale back to [-1,1]
            self.score[(((tile + 1) // 2 + 1) % 2) * 2 - 1] -= len(result)

            # Number of open squares decreases by 1
            self.remaining_squares -= 1

            return True

        else:
            return False

    def is_valid_move(self, tile, xstart, ystart):
        """
        @param int tile
            self.BLACK or self.WHITE
        @param int xstart
        @param int ystart
        Returns False if the player's move on space xstart, ystart is invalid.
        If it is a valid move, returns a list of spaces that would become the
        player's if they made a move here.
        """
        if not self.is_on_board(xstart, ystart) or self.board[xstart][ystart] != 0:
            return False

        # temporarily set the tile on the board.
        self.board[xstart][ystart] = tile

        otherTile = tile * -1

        tiles_to_flip = []
        # loop through all directions around flipped tile
        for xdirection, ydirection in ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)):
            x, y = xstart, ystart
            x += xdirection  # first step in the direction
            y += ydirection  # first step in the direction
            if self.is_on_board(x, y) and self.board[x][y] == otherTile:
                # There is a piece belonging to the other player next to our piece.
                x += xdirection
                y += ydirection
                if not self.is_on_board(x, y):
                    continue
                while self.board[x][y] == otherTile:
                    x += xdirection
                    y += ydirection
                    if not self.is_on_board(x, y):
                        # break out of while loop, then continue in for loop
                        break
                if not self.is_on_board(x, y):
                    continue
                if self.board[x][y] == tile:
                    # There are pieces to flip over. Go in the reverse direction
                    # until we reach the original space, noting all the tiles
                    # along the way.
                    while True:
                        x -= xdirection
                        y -= ydirection
                        if x == xstart and y == ystart:
                            break
                        tiles_to_flip.append([x, y])

        # restore the empty space
        self.board[xstart][ystart] = 0

        # If no tiles were flipped, this is not a valid move.
        return tiles_to_flip

    def print_board(self):
        """
        Print board to terminal
        """

        def get_item(item):
            if item == Board.BLACK:
                return Fore.WHITE + "|" + Fore.BLUE + "O"
            elif item == Board.WHITE:
                return Fore.WHITE + "|" + Fore.RED + "O"
            else:
                return Fore.WHITE + "| "

        def get_row(row):
            return "".join(map(get_item, row))

        print("\t" + Back.BLACK + f"{'BOARD':^18}")
        print("\t" + Back.BLACK + Fore.WHITE + f"  |{'|'.join(map(str, range(1, self.board_size+1)))}")
        for i in range(self.board_size):
            print("\t" + Back.BLACK + Fore.WHITE + f"{i + 1:<2}{get_row(self.board[i]):<2}")
            sys.stdout.write(Style.RESET_ALL)
