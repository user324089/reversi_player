import torch
import bitboard as bb

BLACK = 0
WHITE = 1
SIDE = 8
NUM_PLAYERS = 2
STARTING_BLACK_FIELDS = [SIDE * 3 + 3, SIDE*4+4]
STARTING_WHITE_FIELDS = [SIDE * 3 + 4, SIDE*4+3]
FIELD_CHARACTERS = ['B','W']

STARTING_NUM_FREE_FIELDS = SIDE*SIDE - len(STARTING_BLACK_FIELDS) - len(STARTING_WHITE_FIELDS)

class Reversi:

    def __init__(self, device='cpu') -> None:
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

        self.current_player_bitboard: int = \
            (1 << STARTING_BLACK_FIELDS[0]) | (1 << STARTING_BLACK_FIELDS[1])
        self.other_player_bitboard: int = \
            (1 << STARTING_WHITE_FIELDS[0]) | (1 << STARTING_WHITE_FIELDS[1])

        self.current_player = BLACK

        self.finished: bool = False

        self.calculated_possibilities: torch.Tensor | None = None

        self.device=device

    def update_board(self, captured: int) -> None:
        for pos in range(64):
            if (captured >> pos) & 1:
                self.current_player_board[pos] = 1
                self.other_player_board[pos] = 0

    # places token on bitboard
    # returns the bitboard of all new tokens belonging to the current player
    def place_bitboard(self, index: int) -> int:
        captured = 0
        for shift_func in bb.shift_funcs:
            mask = shift_func(1 << index)
            captured_tmp = 0

            while mask & self.other_player_bitboard:
                captured_tmp |= mask
                mask = shift_func(mask)

            if mask & self.current_player_bitboard:
                captured |= captured_tmp

        self.current_player_bitboard |= 1 << index
        self.current_player_bitboard ^= captured
        self.other_player_bitboard ^= captured
        return captured | (1 << index)
    
    def get_all_moves(self) -> int:
        # Returns a bitboard will all possible moves for the current player
        empty = ~(self.current_player_bitboard | self.other_player_bitboard) & ((1 << 64) - 1)
        moves = 0
        for shift_func in bb.shift_funcs:
            # Only consider cells that are adjacent to the other player's tokens
            mask = shift_func(self.current_player_bitboard) & self.other_player_bitboard

            while mask:
                moves |= shift_func(mask) & empty
                mask = shift_func(mask) & self.other_player_bitboard

        return moves

    def equal_boards(self) -> bool:
        board_curr = bb.board_to_bitboard(self.current_player_board)
        board_other = bb.board_to_bitboard(self.other_player_board)
        return board_curr == self.current_player_bitboard \
               and board_other == self.other_player_bitboard

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
        return torch.cat ((self.current_player_board, self.other_player_board)).to(self.device)

    def get_labeled_fields (self) -> list[torch.Tensor]:
        # Creates a list that indexed by player color returns their board

        fields: list[torch.Tensor] = [torch.Tensor(), torch.Tensor()]
        fields[self.current_player] = self.current_player_board
        fields[self.current_player ^ 1] = self.other_player_board

        return fields

    def get_game_scores(self) -> torch.Tensor:
        # Returns the number of tokens on the board of each player as a tensor
        scores = torch.empty(2)
        scores[self.current_player] = torch.tensor(float(self.current_player_bitboard.bit_count()))
        scores[self.current_player ^ 1] = torch.tensor(float(self.other_player_bitboard.bit_count()))
        return scores


    def get_game_state (self) -> tuple[int, torch.Tensor]:
        # Returns current player and the number of tokens 
        # on the board of each player as a tensor
        scores = self.get_game_scores()
        return (self.current_player, scores)

    def get_player_num_tokens (self, player: int) -> torch.Tensor:
        return torch.tensor(float(self.current_player_bitboard.bit_count() 
                                  if player == self.current_player 
                                  else self.other_player_bitboard.bit_count()))

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
        self.current_player_bitboard, self.other_player_bitboard = self.other_player_bitboard, self.current_player_bitboard
        self.current_player = BLACK+WHITE - self.current_player

        self.calculated_possibilities = None

    def place (self, place_x: int, place_y: int) -> None:
        # Plays a turn by the current player placing token in given place

        new_tokens = self.place_bitboard(place_y * 8 + place_x)
        self.update_board(new_tokens)

        self.calculated_possibilities = None

        if self.other_player_bitboard == 0:
            self.current_player_board[:] = 1
            self.current_player_bitboard = (1 << 64) - 1
            self.finished=True
            return
        self.change_turn()

        if (self.current_player_bitboard | self.other_player_bitboard) == ((1 << 64) - 1):
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

        self.calculated_possibilities = bb.bitboard_to_board(self.get_all_moves())
        return self.calculated_possibilities.clone()

    def get_possibility_inf_mask (self) -> torch.Tensor:
        mask = torch.zeros (self.board_side ** 2).to(self.device)
        mask[self.get_possibilities() == 0] = float ('-inf')
        return mask

    def place_from_probabilities (self, probabilities: torch.Tensor) -> int:
        # Plays a turn by the current player by sampling from given probabilities of
        # legal fields

        mask = self.get_possibilities ()
        masked_probabilities = mask * probabilities.to('cpu')
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
        return int(torch.argmax(torch.tensor(score)))

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
