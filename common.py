import torch
from reversi import *
import random

if torch.cuda.is_available():
    DEVICE='cuda'
else:
    DEVICE='cpu'

def randomize_game (max_num_moves: int) -> Reversi:
    while (True):
        game = Reversi(device=DEVICE)
        num_moves = random.randrange (0, max_num_moves+1)
        for _ in range (num_moves):
            if game.is_finished():
                break
            game.make_random_move()
        if not game.is_finished():
            return game
