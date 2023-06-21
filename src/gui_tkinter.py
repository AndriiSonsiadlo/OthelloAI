import time
import tkinter as tk
from tkinter import messagebox
import numpy as np
import itertools
import random

from src import nn


# Constants
BOARD_SIZE = 8
CELL_SIZE = 60
EMPTY = 0
BLACK = 1
WHITE = 2
DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]


class RLPlayer:
    def __init__(self, discount_factor=1, net_lr=0.01, board_size=8):
        self.policy_net = nn.NN(
            [board_size ** 2, board_size ** 2 * 2, board_size ** 2 * 2, board_size ** 2, board_size ** 2], net_lr)
        self.epsilon = 0.6
        self.discount_factor = discount_factor
        self.play_history = []
        self.wins = 0

    def play(self, place_func, board, board_state, me, log_history=True):
        input_state = np.apply_along_axis(
            lambda x: int((x == me and 1) or (x != 0 and -1)), 1, board_state.reshape((BOARD_SIZE ** 2, 1))).reshape(
            (BOARD_SIZE ** 2, 1))
        made_move = False
        pos = None

        if np.random.random() < self.epsilon:
            positions = list(itertools.product(range(8), repeat=2))
            random.shuffle(positions)
            while not made_move and positions:
                pos = positions.pop()
                made_move = place_func(*pos, WHITE)

            if not made_move and not positions:
                return False

        else:
            out = self.policy_net.get_output(input_state)
            positions = [(v, i) for i, v in enumerate(out)]
            positions.sort(key=lambda x: x[0], reverse=True)

            while not made_move and positions:
                scalar_play_point = positions.pop()[1]
                pos = scalar_play_point // BOARD_SIZE, scalar_play_point % BOARD_SIZE
                made_move = place_func(*pos, WHITE)

            if not made_move and not positions:
                return False

        if log_history:
            self.play_history.append((np.copy(input_state), pos[0] * 8 + pos[1]))

        return True

    def update_weights(self, final_score):
        i = 0
        state, action = self.play_history[i]
        q = self.policy_net.get_output(state)
        n_play_history = len(self.play_history)
        while i < n_play_history:
            i += 1

            if i == n_play_history:
                q[action] = final_score

            else:
                state_, action_ = self.play_history[i]
                q_ = self.policy_net.get_output(state_)
                q[action] += self.discount_factor * np.max(q_)

            self.policy_net.back_prop(state, self.policy_net.mk_vec(q))

            if i != n_play_history:
                action, q = action_, q_


class OthelloGame:
    def __init__(self, nn_weights="./models/21140501-1-1-8-best-linear-0.03.weights"):
        self.board = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.board[3][3] = WHITE
        self.board[3][4] = BLACK
        self.board[4][3] = BLACK
        self.board[4][4] = WHITE

        self.root = tk.Tk()
        self.root.title("Othello")
        self.canvas = tk.Canvas(self.root, width=CELL_SIZE * BOARD_SIZE, height=CELL_SIZE * BOARD_SIZE, bg="green")
        self.canvas.pack()

        self.rl_player = RLPlayer()
        self.rl_player.policy_net.load(nn_weights)

    def draw_board(self):
        self.canvas.delete("all")
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                x1 = col * CELL_SIZE
                y1 = row * CELL_SIZE
                x2 = x1 + CELL_SIZE
                y2 = y1 + CELL_SIZE
                self.canvas.create_rectangle(x1, y1, x2, y2, fill="green", outline="black")
                if self.board[row][col] == BLACK:
                    self.canvas.create_oval(x1, y1, x2, y2, fill="black")
                elif self.board[row][col] == WHITE:
                    self.canvas.create_oval(x1, y1, x2, y2, fill="white")

    def is_valid_move(self, row, col, color):
        if self.board[row][col] != EMPTY:
            return False
        for direction in DIRECTIONS:
            if self.check_direction(row, col, direction, color):
                return True
        return False

    def check_direction(self, row, col, direction, color):
        x, y = direction
        r = row + x
        c = col + y
        if r < 0 or r >= BOARD_SIZE or c < 0 or c >= BOARD_SIZE or self.board[r][c] == EMPTY or self.board[r][
            c] == color:
            return False
        while self.board[r][c] != color:
            r += x
            c += y
            if r < 0 or r >= BOARD_SIZE or c < 0 or c >= BOARD_SIZE or self.board[r][c] == EMPTY:
                return False
        return True

    def flip_tiles(self, row, col, direction, color):
        x, y = direction
        r = row + x
        c = col + y
        while self.board[r][c] != color:
            self.board[r][c] = color
            r += x
            c += y

    def make_move(self, row, col, color):
        if not self.is_valid_move(row, col, color):
            return False
        self.board[row][col] = color
        for direction in DIRECTIONS:
            if self.check_direction(row, col, direction, color):
                self.flip_tiles(row, col, direction, color)
        self.draw_board()
        return True

    def is_game_over(self):
        if any(EMPTY in row for row in self.board):
            return False
        return True

    def count_tiles(self):
        black_count = sum(row.count(BLACK) for row in self.board)
        white_count = sum(row.count(WHITE) for row in self.board)
        return black_count, white_count

    def ai_move(self):
        if self.rl_player.play(self.make_move, self.board, np.array(self.board), BLACK):
            if self.is_game_over():
                self.end_game()
            return True
        return False

    def handle_click(self, event):
        col = event.x // CELL_SIZE
        row = event.y // CELL_SIZE
        if self.make_move(row, col, BLACK):
            if self.is_game_over():
                self.end_game()
            else:
                self.ai_move()

    def end_game(self):
        black_count, white_count = self.count_tiles()
        message = "Game Over\n\nBlack: {}\nWhite: {}".format(black_count, white_count)
        if black_count > white_count:
            message += "\n\nBlack wins!"
        elif white_count > black_count:
            message += "\n\nWhite wins!"
        else:
            message += "\n\nIt's a tie!"
        messagebox.showinfo("Game Over", message)
        self.root.quit()

    def start(self):
        self.draw_board()
        self.canvas.bind("<Button-1>", self.handle_click)
        self.root.mainloop()



