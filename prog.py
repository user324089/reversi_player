import torch
from tqdm import tqdm

BLACK = 0
WHITE = 1
SIDE = 8
NUM_PLAYERS = 2
STARTING_BLACK_FIELDS = [SIDE * 3 + 3, SIDE*4+4]
STARTING_WHITE_FIELDS = [SIDE * 3 + 4, SIDE*4+3]
FIELD_CHARACTERS = ['B','W']

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

    def place (self, place_x: int, place_y: int) -> None:
        # Plays a turn by the current player placing token in given place

        self.current_player_board[place_y * self.board_side + place_x] = 1
        self._place_helper (place_x, place_y, False)
        if (torch.sum (self.other_player_board) == 0):
            self.current_player_board[:] = 1
            return
        self.change_turn()
        if (torch.sum(self.get_possibilities()) == 0):
            self.change_turn()

    def get_possibilities (self) -> torch.Tensor:
        # Returns the places where current player can place their token

        possibilities: torch.Tensor = torch.empty (self.board_side ** 2)
        for place_x in range (self.board_side):
            for place_y in range (self.board_side):
                possibilities[place_y * self.board_side + place_x] = self.can_place (place_x, place_y)
        return possibilities

    def count_free_spaces (self) -> torch.Tensor:
        # Counts spaces not containing a token

        return self.board_side ** 2 - torch.sum (self.current_player_board + self.other_player_board)

    def place_from_probabilities (self, probabilities: torch.Tensor) -> int:
        # Plays a turn by the current player by sampling from given probabilities of
        # legal fields

        mask = self.get_possibilities ()
        masked_probabilities = mask * probabilities
        masked_probabilities_sum = torch.sum (masked_probabilities)
        if (masked_probabilities_sum == 0):
            masked_probabilities[:] = 1/(SIDE*SIDE)
        else:
            masked_probabilities /= masked_probabilities_sum
        dist = torch.distributions.categorical.Categorical (masked_probabilities)
        index = int(dist.sample())

        self.place (index % self.board_side, index // self.board_side)
        return index

    def is_finished (self):
        return self.count_free_spaces() == 0

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


class Reversi_AI (torch.nn.Module):

    def __init__ (self, num_hidden_layers: int, hidden_layer_width: int) -> None:
        super().__init__()

        assert (num_hidden_layers >= 1)

        self.first_layer = torch.nn.Sequential(
                torch.nn.Linear (SIDE*SIDE*2, hidden_layer_width),
                torch.nn.ReLU())

        self.middle_layers = torch.nn.Sequential (
                *[
                    torch.nn.Sequential (
                        torch.nn.Linear (hidden_layer_width, hidden_layer_width),
                        torch.nn.ReLU()
                        )
                    for _ in range (num_hidden_layers - 1)
                    ]
                )
        self.last_layer = torch.nn.Linear (hidden_layer_width, SIDE*SIDE)
        self.soft_max = torch.nn.Softmax (dim=0)

    def forward (self, x) -> torch.Tensor:
        x = self.first_layer (x)
        x = self.middle_layers(x)
        x = self.last_layer (x)
        x = self.soft_max (x)
        return x

    def print_first (self):
        print (*self.first_layer.parameters())
        print ('last layer grad:')
        for parameter in self.last_layer.parameters():
            print (parameter.grad)

def test_model (model: Reversi_AI, num_games: int):
    num_won: float = 0
    for _ in tqdm(range (num_games)):
        game: Reversi = Reversi()
        model_player = BLACK + WHITE - game.current_player
        while not game.is_finished():
            if (game.current_player == model_player):
                game.place_from_probabilities(model(game.get_board_state()))
            else:
                game.place_from_probabilities(torch.nn.functional.softmax(torch.randn (SIDE*SIDE), dim=0))
        if (game.get_winner () == model_player):
            num_won += 1
    return num_won / num_games


def train_AI (model: Reversi_AI, num_epochs: int, games_per_epoch, optimiser: torch.optim.Adam):
    for _ in tqdm(range (num_epochs)):

        optimiser.zero_grad()

        loss: torch.Tensor = torch.tensor(0, dtype=torch.float)

        for _ in range (games_per_epoch):
            game: Reversi = Reversi()
            states: list[tuple[int, list[torch.Tensor]]] = []
            log_probs: list[torch.Tensor] = []
            while (not game.is_finished()):
                states.append (game.get_game_state())
                board_state = game.get_board_state()
                move_probabilities = model(board_state)
                move_taken = game.place_from_probabilities (move_probabilities)
                move_taken_one_hot = torch.nn.functional.one_hot (torch.tensor(move_taken),
                                                                  SIDE*SIDE)

                log_probs.append (torch.log(torch.sum (move_probabilities * move_taken_one_hot)))

            _, end_state = game.get_game_state()

            for j, (log_prob, (current_player, current_state)) in enumerate(zip (log_probs, states, strict=True)):
                baseline = torch.tensor((STARTING_NUM_FREE_FIELDS - j)/2)
                total_reward: float = float(end_state[current_player] - current_state[current_player] - baseline)

                loss -= log_prob * total_reward

        loss /= games_per_epoch

        loss.backward()

        optimiser.step()


def main ():

    torch.manual_seed (0);
    model: Reversi_AI = Reversi_AI(2,100)

    print ('testing model')
    starting_accuracy = test_model (model, 100)
    print ('start accuracy:', starting_accuracy)
    print ('training model')
    train_AI (model, 100, 10, torch.optim.Adam (model.parameters()))
    print ('testing model')
    end_accuracy = test_model (model, 100)
    print ('end accuracy:', end_accuracy)

main()
