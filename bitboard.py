import torch

# Edge masks to prevent illegal shifts
RIGHT_EDGE = 0x0101010101010101
LEFT_EDGE = 0x8080808080808080
TOP_EDGE = 0xFF00000000000000
BOTTOM_EDGE = 0x00000000000000FF

def shift_left(board: int) -> int:
    return (board & ~LEFT_EDGE) << 1

def shift_right(board: int) -> int:
    return (board & ~RIGHT_EDGE) >> 1

def shift_up(board: int) -> int:
    return (board & ~TOP_EDGE) << 8

def shift_down(board: int) -> int:
    return (board & ~BOTTOM_EDGE) >> 8

def shift_up_left(board: int) -> int:
    return (board & ~TOP_EDGE & ~LEFT_EDGE) << 9

def shift_up_right(board: int) -> int:
    return (board & ~TOP_EDGE & ~RIGHT_EDGE) << 7

def shift_down_left(board: int) -> int:
    return (board & ~BOTTOM_EDGE & ~LEFT_EDGE) >> 7

def shift_down_right(board: int) -> int:
    return (board & ~BOTTOM_EDGE & ~RIGHT_EDGE) >> 9

shift_funcs = [shift_left, shift_right, shift_up, shift_down, shift_up_left, shift_up_right, shift_down_left, shift_down_right]

def board_to_bitboard(board: torch.Tensor) -> int:
    bitboard = 0
    for pos in range(64):
        if board[pos] == 1:
            bitboard |= 1 << pos
    return bitboard

def bitboard_to_board(bitboard: int) -> torch.Tensor:
    board = torch.zeros(64)
    for pos in range(64):
        if bitboard & (1 << pos):
            board[pos] = 1
    return board