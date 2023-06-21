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

# Initialize the board
board = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]
board[3][3] = WHITE
board[3][4] = BLACK
board[4][3] = BLACK
board[4][4] = WHITE

# Initialize GUI
root = tk.Tk()
root.title("Othello")
canvas = tk.Canvas(root, width=CELL_SIZE * BOARD_SIZE, height=CELL_SIZE * BOARD_SIZE, bg="green")
canvas.pack()

# Draw the initial board
def draw_board():
    canvas.delete("all")
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            x1 = col * CELL_SIZE
            y1 = row * CELL_SIZE
            x2 = x1 + CELL_SIZE
            y2 = y1 + CELL_SIZE
            canvas.create_rectangle(x1, y1, x2, y2, fill="green", outline="black")
            if board[row][col] == BLACK:
                canvas.create_oval(x1, y1, x2, y2, fill="black")
            elif board[row][col] == WHITE:
                canvas.create_oval(x1, y1, x2, y2, fill="white")

# Check if a move is valid
def is_valid_move(row, col, color):
    if board[row][col] != EMPTY:
        return False
    for direction in DIRECTIONS:
        if check_direction(row, col, direction, color):
            return True
    return False

# Check if flipping tiles is possible in a direction
def check_direction(row, col, direction, color):
    x, y = direction
    r = row + x
    c = col + y
    if r < 0 or r >= BOARD_SIZE or c < 0 or c >= BOARD_SIZE or board[r][c] == EMPTY or board[r][c] == color:
        return False
    while board[r][c] != color:
        r += x
        c += y
        if r < 0 or r >= BOARD_SIZE or c < 0 or c >= BOARD_SIZE or board[r][c] == EMPTY:
            return False
    return True

# Flip tiles in a direction
def flip_tiles(row, col, direction, color):
    x, y = direction
    r = row + x
    c = col + y
    while board[r][c] != color:
        board[r][c] = color
        r += x
        c += y

# Make a move
def make_move(row, col, color):
    if not is_valid_move(row, col, color):
        return False
    board[row][col] = color
    for direction in DIRECTIONS:
        if check_direction(row, col, direction, color):
            flip_tiles(row, col, direction, color)
    draw_board()
    return True

# Check if the game is over
def is_game_over():
    if any(EMPTY in row for row in board):
        return False
    return True

# Count the number of tiles for each color
def count_tiles():
    black_count = sum(row.count(BLACK) for row in board)
    white_count = sum(row.count(WHITE) for row in board)
    return black_count, white_count

# RL Player using Policy Gradient
class RLPlayer:
    def __init__(self, discount_factor=1, net_lr=0.01, board_size=8):
        self.policy_net = nn.NN([board_size ** 2, board_size ** 2 * 2,  board_size ** 2 * 2,  board_size ** 2,  board_size ** 2], net_lr)
        self.epsilon = 0.6
        self.discount_factor = discount_factor
        self.play_history = []
        self.wins = 0

    def play(self, place_func, board, board_state, me, log_history=True):
        input_state = np.apply_along_axis(
            lambda x: int((x == me and 1) or (x != 0 and -1)), 1, board_state.reshape((BOARD_SIZE ** 2, 1))).reshape((BOARD_SIZE ** 2, 1))
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

# Human Player
class HumanPlayer:
    def play(self, place_func, board_state, board, me, log_history=True):
        while True:
            try:
                while len(pos := list(map(int, map(str.strip, input().strip().split(" "))))) != 2:
                    print("Please enter two numbers separated by a space.")

                if place_func(pos[0]-1, pos[1]-1):
                    return True
            except ValueError:
                pass

            print("Invalid values. Please enter two numbers (row and col) separated by a space.")

# Create RL and Human players
rl_player = RLPlayer()
rl_player.policy_net.load("./8size-best-linear-0.03.weights")
human_player = HumanPlayer()

# AI makes a move
def ai_move():
    if rl_player.play(make_move, board, np.array(board), BLACK):
        if is_game_over():
            end_game()
        return True
    return False

# Handle mouse click event
def handle_click(event):
    col = event.x // CELL_SIZE
    row = event.y // CELL_SIZE
    if make_move(row, col, BLACK):
        if is_game_over():
            end_game()
        else:
            time.sleep(0.5)
            ai_move()

# End the game and show the result
def end_game():
    black_count, white_count = count_tiles()
    message = "Game Over\n\nBlack: {}\nWhite: {}".format(black_count, white_count)
    if black_count > white_count:
        message += "\n\nBlack wins!"
    elif white_count > black_count:
        message += "\n\nWhite wins!"
    else:
        message += "\n\nIt's a tie!"
    messagebox.showinfo("Game Over", message)
    root.quit()

draw_board()
canvas.bind("<Button-1>", handle_click)
root.mainloop()
