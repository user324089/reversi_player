import torch

BLACK = 0
WHITE = 1
SIDE = 8
NUM_PLAYERS = 2
STARTING_BLACK_FIELDS = [SIDE * 3 + 3, SIDE*4+4]
STARTING_WHITE_FIELDS = [SIDE * 3 + 4, SIDE*4+3]
FIELD_CHARACTERS = ['B','W']

class Reversi:

    def __init__(self) -> None:
        self.board_side = SIDE

        self.current_player_board = torch.zeros (self.board_side**2)
        self.other_player_board = torch.zeros (self.board_side**2)

        self.current_player_board[STARTING_BLACK_FIELDS] = 1
        self.other_player_board[STARTING_WHITE_FIELDS] = 1

        self.current_player = BLACK

    def count_in_direction (self, place_index: int, index_offset: int, num_steps: int) -> int:

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

        current_index = place_index + index_offset

        for _ in range (num_steps):
            if (self.other_player_board[current_index] > 0):
                self.current_player_board[current_index] = 1
                self.other_player_board[current_index] = 0
            else:
                return
            current_index += index_offset

    def get_board_state (self) -> torch.Tensor:
        return torch.cat ((self.current_player_board, self.other_player_board))

    def get_labeled_fields (self) -> list[torch.Tensor]:
        fields: list[torch.Tensor] = [torch.Tensor() for _ in range(NUM_PLAYERS)]

        if (self.current_player == BLACK):
            fields[BLACK] = self.current_player_board
            fields[WHITE] = self.other_player_board
        else:
            fields[WHITE] = self.current_player_board
            fields[BLACK] = self.other_player_board

        return fields

    def get_game_state (self) -> tuple [int, list[torch.Tensor]]:
        fields: list[torch.Tensor] = self.get_labeled_fields()
        scores: list[torch.Tensor] = [torch.sum(fields[player]) for player in range (NUM_PLAYERS)]
        return (self.current_player, scores)

    def _place_helper (self, place_x: int, place_y: int, is_checking: bool) -> bool:
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
        place_index = place_y * self.board_side + place_x
        if (self.current_player_board [place_index] + self.other_player_board[place_index] > 0):
            return 0
        return float(self._place_helper (place_x, place_y, True))

    def change_turn (self) -> None:
        self.current_player_board, self.other_player_board = self.other_player_board, self.current_player_board
        self.current_player = BLACK+WHITE - self.current_player

    def place (self, place_x: int, place_y: int) -> None:
        self.current_player_board[place_y * self.board_side + place_x] = 1
        self._place_helper (place_x, place_y, False)
        if (torch.sum (self.other_player_board) == 0):
            self.current_player_board[:] = 1
            return
        self.change_turn()
        if (torch.sum(self.get_possibilities()) == 0):
            self.change_turn()

    def get_possibilities (self) -> torch.Tensor:
        possibilities: torch.Tensor = torch.empty (self.board_side ** 2)
        for place_x in range (self.board_side):
            for place_y in range (self.board_side):
                possibilities[place_y * self.board_side + place_x] = self.can_place (place_x, place_y)
        return possibilities

    def place_from_probabilities (self, probabilities: torch.Tensor) -> int:
        mask = (1 - self.get_possibilities ()) * float('-inf')
        probabilities -= mask
        index = int(torch.argmax (mask).item())
        self.place (index % self.board_side, index // self.board_side)
        return index

    def is_finished (self):
        return torch.sum (self.current_player_board + self.other_player_board) == SIDE**2

    def write (self, show_possibilities: bool = True) -> None:

        fields: list[torch.Tensor] = self.get_labeled_fields()

        possibilities: torch.Tensor = game.get_possibilities ()

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


game: Reversi = Reversi()


while (not game.is_finished()):
    game.write()
    print()
    print()
        
    probabilities = torch.randn (SIDE*SIDE)
    game.place_from_probabilities (probabilities)
