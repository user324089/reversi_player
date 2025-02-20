import torch
from tqdm import tqdm
import os
import copy
import random
from reversi import *
from model import *

RANDOM_MOVE_PROBABILITY=0.03
EXPLORING_MOVE_PROBABILITY=0.1
EXPLORING_K = 3
RANDOM_OPPONENT_MOVE_PROBABILITY=0.2

BEGINNING_RANDOM_MOVE_MAX_COUNT = 40
REPLAY_BUFFER_SIZE = 3000
LEARNING_BATCH_SIZE = 500

WINNING_REWARD = 40
LOSING_REWARD = -40

if torch.cuda.is_available():
    DEVICE='cuda'
else:
    DEVICE='cpu'


def test_model_DQN (model: Reversi_AI_DQN, num_games: int):

    model.eval()
    num_won: float = 0
    for _ in tqdm(range (num_games)):
        game: Reversi = Reversi(device=DEVICE)
        model_player = BLACK + WHITE - game.current_player
        while not game.is_finished():
            if (game.current_player == model_player):
                game.place_optimal(model(game.get_board_state()))
            else:
                game.make_random_move()
        if (game.get_winner () == model_player):
            num_won += 1
    return num_won / num_games

def randomize_game () -> Reversi:
    while (True):
        game = Reversi(device=DEVICE)
        num_moves = random.randrange (0, BEGINNING_RANDOM_MOVE_MAX_COUNT+1)
        for _ in range (num_moves):
            if game.is_finished():
                break
            game.make_random_move()
        if not game.is_finished():
            return game

def train_AI_DQN (model: Reversi_AI_DQN, num_moves: int, target_net_delay: int, optimiser: torch.optim.AdamW):

    model.train()

    opponent_model = copy.deepcopy(model)
    opponent_model.requires_grad_(False)
    opponent_model.eval()

    state_dataset = torch.empty(REPLAY_BUFFER_SIZE, 2*SIDE*SIDE).to(DEVICE)
    next_state_dataset: torch.Tensor = torch.empty(REPLAY_BUFFER_SIZE, 2*SIDE*SIDE).to(DEVICE)
    next_state_valid_mask_dataset: torch.Tensor = torch.empty(REPLAY_BUFFER_SIZE, SIDE*SIDE).to(DEVICE)
    action_dataset = torch.zeros (REPLAY_BUFFER_SIZE, dtype=torch.int).to(DEVICE)
    reward_dataset = torch.zeros (REPLAY_BUFFER_SIZE).to(DEVICE)
    is_finished_dataset = torch.zeros(REPLAY_BUFFER_SIZE, dtype=torch.bool).to(DEVICE)

    game = Reversi (device=DEVICE)
    LEARNING_MODEL_PLAYER = game.current_player

    target_model = model # this gets changed at the beginning of the for loop. this assignment
                                         # only prevents pyright error

    for i in tqdm(range (num_moves)):

        if i % target_net_delay == 0:
            target_model = copy.deepcopy(model)
            target_model.requires_grad_(False)
            target_model.eval()

        before_move_score = game.get_player_num_tokens (LEARNING_MODEL_PLAYER)

        state = game.get_board_state()

        state_dataset[i % REPLAY_BUFFER_SIZE] = state

        q_scores = model(state)

        move_version_rand_float = random.random()
        if (move_version_rand_float < RANDOM_MOVE_PROBABILITY):
            move_taken = game.make_random_move()
        elif (move_version_rand_float < RANDOM_MOVE_PROBABILITY + EXPLORING_MOVE_PROBABILITY):
            field_values = game.get_possibility_inf_mask() + q_scores
            explored_bound = torch.topk (field_values, k=EXPLORING_K)[0][-1]
            move_taken = game.place_from_probabilities (field_values >= explored_bound)
        else:
            move_taken = game.place_optimal (q_scores)

        action_dataset[i % REPLAY_BUFFER_SIZE] = move_taken

        while ((not game.is_finished()) and game.current_player != LEARNING_MODEL_PLAYER):
            if (random.random() < RANDOM_OPPONENT_MOVE_PROBABILITY):
                game.make_random_move()
            else:
                game.place_optimal(opponent_model(game.get_board_state()))

        #after_move_score = game.get_player_num_tokens (LEARNING_MODEL_PLAYER)
        reward_dataset[i%REPLAY_BUFFER_SIZE] = 0 #after_move_score - before_move_score
        is_finished_dataset[i%REPLAY_BUFFER_SIZE] = game.is_finished()
        next_state_dataset[i%REPLAY_BUFFER_SIZE] = game.get_board_state()
        next_state_valid_mask_dataset[i%REPLAY_BUFFER_SIZE] = game.get_possibility_inf_mask()

        if (game.is_finished()):
            if game.get_winner() == LEARNING_MODEL_PLAYER:
                reward_dataset[i%REPLAY_BUFFER_SIZE] += WINNING_REWARD
            elif game.get_winner() == (LEARNING_MODEL_PLAYER^1):
                reward_dataset[i%REPLAY_BUFFER_SIZE] += LOSING_REWARD

            game = randomize_game()
            LEARNING_MODEL_PLAYER = game.current_player

        if i < LEARNING_BATCH_SIZE:
            continue

        indices = random.sample (range(min(i, REPLAY_BUFFER_SIZE)), LEARNING_BATCH_SIZE)
        targets = torch.max (target_model (next_state_dataset[indices]) + next_state_valid_mask_dataset[indices], dim=1)[0]
        targets[is_finished_dataset[indices]] = 0
        targets += reward_dataset[indices]
        outputs = model(state_dataset[indices])[torch.arange (LEARNING_BATCH_SIZE),action_dataset[indices]]
        loss = torch.nn.functional.mse_loss (outputs, targets)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()



def main ():

    model = create_model()
    model.to(DEVICE)
    optim = torch.optim.AdamW (model.parameters())

    if os.path.exists('model_weights.pth'):
        model.load_state_dict(torch.load('model_weights.pth', weights_only=True))

    torch.manual_seed (0);
    accuracy = test_model_DQN (model, 100)
    print ('start accuracy:', accuracy)
    while (True):
        torch.manual_seed (0);
        train_AI_DQN (model, 20000, 200, optim)
        accuracy = test_model_DQN (model, 100)
        print ('end accuracy:', accuracy)
        torch.save(model.state_dict(), 'model_weights.pth')


main()
