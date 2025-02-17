import torch
from tqdm import tqdm
import os
import copy
import random
from reversi import *

RANDOM_MOVE_PROBABILITY=0.03
GAMMA = 0.99

STARTING_NUM_FREE_FIELDS = SIDE*SIDE - len(STARTING_BLACK_FIELDS) - len(STARTING_WHITE_FIELDS)
class Linear_skip_block (torch.nn.Module):

    def __init__ (self, num_neurons: int) -> None:
        super().__init__()
        self.first_layers = torch.nn.Sequential (
                torch.nn.Linear (num_neurons, num_neurons),
                torch.nn.ReLU(),
                torch.nn.Linear (num_neurons, num_neurons)
                )
        self.relu = torch.nn.ReLU()

    def forward (self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.first_layers (x))

class Reversi_AI_DQN (torch.nn.Module):

    def __init__ (self, num_blocks: int, hidden_layer_width: int) -> None:
        super().__init__()

        self.first_layer = torch.nn.Sequential(
                torch.nn.Linear (SIDE*SIDE*2, hidden_layer_width),
                torch.nn.ReLU()
                )

        self.middle_layers = torch.nn.Sequential (
                *[
                    Linear_skip_block (hidden_layer_width)
                    for _ in range (num_blocks)
                    ]
                )

        self.last_layer = torch.nn.Linear (hidden_layer_width, SIDE*SIDE)

    def forward (self, x) -> torch.Tensor:
        x = self.first_layer (x)
        x = self.middle_layers(x)
        x = self.last_layer (x)
        return x

def test_model_DQN (model: Reversi_AI_DQN, num_games: int):

    model.eval()
    num_won: float = 0
    for _ in tqdm(range (num_games)):
        game: Reversi = Reversi()
        model_player = BLACK + WHITE - game.current_player
        while not game.is_finished():
            if (game.current_player == model_player):
                game.place_optimal(model(game.get_board_state()))
            else:
                game.place_from_probabilities(torch.nn.functional.softmax(torch.randn (SIDE*SIDE), dim=0))
        if (game.get_winner () == model_player):
            num_won += 1
    return num_won / num_games

def train_AI_DQN (model: Reversi_AI_DQN, num_games: int, optimiser: torch.optim.AdamW):

    model.train()

    opponent_model = copy.deepcopy(model)

    for _ in tqdm(range (num_games)):


        game = Reversi ()

        if (random.randrange(0,2) == 0):
            game.place_optimal(opponent_model(game.get_board_state()))

        LEARNING_MODEL_PLAYER = game.current_player

        while (not game.is_finished()):

            before_move_score = game.get_player_num_tokens (LEARNING_MODEL_PLAYER)

            q_scores = model(game.get_board_state())

            if (random.random() < RANDOM_MOVE_PROBABILITY):
                move_taken = game.place_from_probabilities (torch.ones(SIDE*SIDE))
            else:
                move_taken = game.place_optimal (q_scores)

            while ((not game.is_finished()) and game.current_player != LEARNING_MODEL_PLAYER):
                game.place_optimal(opponent_model(game.get_board_state()))

            after_move_score = game.get_player_num_tokens (LEARNING_MODEL_PLAYER)
            target: torch.Tensor = after_move_score - before_move_score

            if not game.is_finished():
                target += torch.max (game.get_possibility_inf_mask() + model(game.get_board_state()))

            chosen_score = q_scores[move_taken:move_taken+1].squeeze()
            loss = (chosen_score - target)**2 / 2

            optimiser.zero_grad()

            loss.backward()

            optimiser.step()


def main ():

    model = Reversi_AI_DQN(5,400)
    optim = torch.optim.AdamW (model.parameters())

    if os.path.exists('model_weights.pth'):
        model.load_state_dict(torch.load('model_weights.pth', weights_only=True))

    torch.manual_seed (0);
    accuracy = test_model_DQN (model, 100)
    print ('start accuracy:', accuracy)
    while (True):
        torch.manual_seed (0);
        train_AI_DQN (model, 1000, optim)
        accuracy = test_model_DQN (model, 100)
        print ('end accuracy:', accuracy)
        torch.save(model.state_dict(), 'model_weights.pth')


main()
