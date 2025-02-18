import torch
from tqdm import tqdm
import os
import copy
import random

BLACK = 0
WHITE = 1
SIDE = 8
NUM_PLAYERS = 2
STARTING_BLACK_FIELDS = [SIDE * 3 + 3, SIDE*4+4]
STARTING_WHITE_FIELDS = [SIDE * 3 + 4, SIDE*4+3]
FIELD_CHARACTERS = ['B','W']

RANDOM_MOVE_PROBABILITY=0.03
GAMMA = 0.99

STARTING_NUM_FREE_FIELDS = SIDE*SIDE - len(STARTING_BLACK_FIELDS) - len(STARTING_WHITE_FIELDS)

class Reversi:

    def __init__(self) -> None:
        self.board_side = SIDE

        # Board is zeros where there is no token, and ones where there is token.
        # current_player_board is the board of currently deciding player's tokens,
        # other_player_board is the board of the other player's tokens

        # Initialise boards to zeros
        self.current_player_board = torch.zeros (self.board_side**2)
        self.other_player_board = torch.zeros (self.board_side**2)

        # Set the starting fields to ones
        self.current_player_board[STARTING_BLACK_FIELDS] = 1
        self.other_player_board[STARTING_WHITE_FIELDS] = 1

        self.current_player = BLACK

        self.finished: bool = False

        self.calculated_possibilities: torch.Tensor | None = None

    def count_in_direction (self, place_index: int, index_offset: int, num_steps: int) -> int:
        # Function counts the number of other player's tokens before current player token.
        # place_index is the beginning place where we begin counting
        # index offset is the offset we need to advance to get to next token in line
        # num_steps is the number of steps we can go before getting to the end of the board

        num_seen = 0
        current_index: int = place_index + index_offset

        for _ in range (num_steps):
            if (self.other_player_board[current_index] > 0):
                num_seen += 1
            elif (self.current_player_board[current_index] > 0):
                return num_seen
            else:
                return 0
            current_index += index_offset
        return 0

    def place_in_direction (self, place_index: int, index_offset: int, num_steps: int) -> None:
        # Flips tokens in a straight line. Variables named the same as in count_in_direction
        # Assumes the placing is valid

        current_index = place_index + index_offset

        for _ in range (num_steps):
            if (self.other_player_board[current_index] > 0):
                self.current_player_board[current_index] = 1
                self.other_player_board[current_index] = 0
            else:
                return
            current_index += index_offset

    def get_board_state (self) -> torch.Tensor:
        # Returns the whole board state as a tensor
        return torch.cat ((self.current_player_board, self.other_player_board))

    def get_labeled_fields (self) -> list[torch.Tensor]:
        # Creates a list that indexed by player color returns their board

        fields: list[torch.Tensor] = [torch.Tensor() for _ in range(NUM_PLAYERS)]

        if (self.current_player == BLACK):
            fields[BLACK] = self.current_player_board
            fields[WHITE] = self.other_player_board
        else:
            fields[WHITE] = self.current_player_board
            fields[BLACK] = self.other_player_board

        return fields

    def get_game_state (self) -> tuple [int, list[torch.Tensor]]:
        # Returns current player and the number of tokens on the board of each player
        # as a list indexed by their color

        fields: list[torch.Tensor] = self.get_labeled_fields()
        scores: list[torch.Tensor] = [torch.sum(fields[player]) for player in range (NUM_PLAYERS)]
        return (self.current_player, scores)

    def get_player_num_tokens (self, player: int) -> torch.Tensor:
        _, num_tokens = self.get_game_state()
        return num_tokens[player]

    def _place_helper (self, place_x: int, place_y: int, is_checking: bool) -> bool:
        # If is_checking is true, checks if current player flips any tokens by placing
        # theirs in the given position. Otherwise flips all tokens from the current
        # position

        x_places = [place_x, self.board_side, self.board_side - 1 - place_x]
        y_places = [place_y, self.board_side, self.board_side - 1 - place_y]
        offsets = [-1,0,1]

        place_index = place_y * self.board_side + place_x

        num_changed = 0

        for x in range (3):
            for y in range (3):

                index_offset = offsets[y] * self.board_side + offsets[x]

                if (index_offset == 0):
                    continue

                num_steps = min(x_places[x], y_places[y])

                num_in_direction = self.count_in_direction (place_index, index_offset, num_steps)
                num_changed += num_in_direction

                if (num_in_direction > 0):
                    if (is_checking):
                        return True
                    else:
                        self.place_in_direction (place_index, index_offset, num_steps)

        return (num_changed > 0)

    def can_place (self, place_x: int, place_y: int) -> float:
        # Checks if current player can place a token in given field and returns 1 if yes, otherwise 0

        place_index = place_y * self.board_side + place_x
        if (self.current_player_board [place_index] + self.other_player_board[place_index] > 0):
            return 0
        return float(self._place_helper (place_x, place_y, True))

    def change_turn (self) -> None:
        # Changes the current player

        self.current_player_board, self.other_player_board = self.other_player_board, self.current_player_board
        self.current_player = BLACK+WHITE - self.current_player

        self.calculated_possibilities = None

    def place (self, place_x: int, place_y: int) -> None:
        # Plays a turn by the current player placing token in given place

        self.current_player_board[place_y * self.board_side + place_x] = 1
        self._place_helper (place_x, place_y, False)
        self.calculated_possibilities = None

        if (torch.sum (self.other_player_board) == 0):
            self.current_player_board[:] = 1
            self.finished=True
            return
        self.change_turn()

        if (torch.sum (self.current_player_board + self.other_player_board) == self.board_side * self.board_side):
            self.finished = True
            return

        if (torch.sum(self.get_possibilities()) == 0):
            self.change_turn()

            if (torch.sum(self.get_possibilities()) == 0):
                self.finished=True

    def get_possibilities (self) -> torch.Tensor:
        # Returns the places where current player can place their token
        if self.calculated_possibilities is not None:
            return self.calculated_possibilities.clone()

        self.calculated_possibilities = torch.empty (self.board_side ** 2)
        for place_x in range (self.board_side):
            for place_y in range (self.board_side):
                self.calculated_possibilities[place_y * self.board_side + place_x] = self.can_place (place_x, place_y)

        return self.calculated_possibilities.clone()

    def get_possibility_inf_mask (self) -> torch.Tensor:
        mask = torch.zeros (self.board_side ** 2)
        mask[self.get_possibilities() == 0] = float ('-inf')
        return mask

    def place_from_probabilities (self, probabilities: torch.Tensor) -> int:
        # Plays a turn by the current player by sampling from given probabilities of
        # legal fields

        mask = self.get_possibilities ()
        masked_probabilities = mask * probabilities
        masked_probabilities_sum = torch.sum (masked_probabilities)
        if (masked_probabilities_sum == 0):
            masked_probabilities[:] = mask/torch.sum(mask)
        else:
            masked_probabilities /= masked_probabilities_sum
        dist = torch.distributions.categorical.Categorical (masked_probabilities)
        index = int(dist.sample())

        self.place (index % self.board_side, index // self.board_side)
        return index

    def place_optimal (self, values: torch.Tensor):
        mask = self.get_possibility_inf_mask()
        index = int(torch.argmax(values + mask))
        self.place (index % self.board_side, index // self.board_side)
        return index

    def is_finished (self):
        return self.finished

    def get_winner (self) -> int:
        _, score = self.get_game_state()
        max_score = score[0]
        max_score_index = 0
        for i in range (1, len(score)):
            if (score[i] > max_score):
                max_score = score[i]
                max_score_index = i
        return max_score_index

    def write (self, show_possibilities: bool = True) -> None:

        fields: list[torch.Tensor] = self.get_labeled_fields()

        possibilities: torch.Tensor = self.get_possibilities ()

        print ('current player:', FIELD_CHARACTERS[self.current_player])

        for y in range (self.board_side):
            for x in range (self.board_side):
                if (fields[BLACK][self.board_side * y + x] > 0):
                    print (FIELD_CHARACTERS[BLACK], end='')
                elif (fields[WHITE][self.board_side * y + x] > 0):
                    print (FIELD_CHARACTERS[WHITE], end='')
                elif (show_possibilities and possibilities[self.board_side * y + x] > 0):
                    print ('X', end='')
                else:
                    print ('.', end='')
            print ()


class Reversi_AI_DQN (torch.nn.Module):

    def __init__ (self, num_hidden_layers: int, hidden_layer_width: int) -> None:
        super().__init__()

        assert (num_hidden_layers >= 1)

        self.first_layer = torch.nn.Sequential(
                torch.nn.Linear (SIDE*SIDE*2, hidden_layer_width),
                torch.nn.ReLU()
                )

        self.middle_layers = torch.nn.Sequential (
                *[
                    torch.nn.Sequential (
                        torch.nn.Linear (hidden_layer_width, hidden_layer_width),
                        torch.nn.ReLU(),
                        )
                    for _ in range (num_hidden_layers - 1)
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

    model = Reversi_AI_DQN(3,200)
    optim = torch.optim.AdamW (model.parameters(), lr=1e-5)

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
