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

class AI_trainer:
    def __init__(self, target_net_delay: int, opponent_model_delay: int):
        self.target_net_delay: int = target_net_delay
        self.opponent_model_delay: int = opponent_model_delay
        self.num_trained_moves: int = 0

        self.state_dataset = torch.empty(REPLAY_BUFFER_SIZE, 2*SIDE*SIDE).to(DEVICE)
        self.next_state_dataset: torch.Tensor = torch.empty(REPLAY_BUFFER_SIZE, 2*SIDE*SIDE).to(DEVICE)
        self.next_state_valid_mask_dataset: torch.Tensor = torch.empty(REPLAY_BUFFER_SIZE, SIDE*SIDE).to(DEVICE)
        self.action_dataset = torch.zeros (REPLAY_BUFFER_SIZE, dtype=torch.int).to(DEVICE)
        self.reward_dataset = torch.zeros (REPLAY_BUFFER_SIZE).to(DEVICE)
        self.is_finished_dataset = torch.zeros(REPLAY_BUFFER_SIZE, dtype=torch.bool).to(DEVICE)

        self.game = Reversi (device=DEVICE)
        self.LEARNING_MODEL_PLAYER = self.game.current_player


    def train (self, model: Reversi_AI_DQN, num_moves: int, optimiser: torch.optim.Optimizer):
        for _ in tqdm(range (num_moves)):

            if self.num_trained_moves % self.opponent_model_delay == 0:
                self.opponent_model = copy.deepcopy(model)
                self.opponent_model.requires_grad_(False)
                self.opponent_model.eval()

            if self.num_trained_moves % self.target_net_delay == 0:
                self.target_model = copy.deepcopy(model)
                self.target_model.requires_grad_(False)
                self.target_model.eval()

            before_move_score = self.game.get_player_num_tokens (self.LEARNING_MODEL_PLAYER)

            state = self.game.get_board_state()

            self.state_dataset[self.num_trained_moves % REPLAY_BUFFER_SIZE] = state

            q_scores = model(state)

            move_version_rand_float = random.random()
            if (move_version_rand_float < RANDOM_MOVE_PROBABILITY):
                move_taken = self.game.make_random_move()
            elif (move_version_rand_float < RANDOM_MOVE_PROBABILITY + EXPLORING_MOVE_PROBABILITY):
                field_values = self.game.get_possibility_inf_mask() + q_scores
                explored_bound = torch.topk (field_values, k=EXPLORING_K)[0][-1]
                move_taken, _ = self.game.place_from_probabilities (field_values >= explored_bound)
            else:
                move_taken = self.game.place_optimal (q_scores)

            self.action_dataset[self.num_trained_moves % REPLAY_BUFFER_SIZE] = move_taken

            while ((not self.game.is_finished()) and self.game.current_player != self.LEARNING_MODEL_PLAYER):
                if (random.random() < RANDOM_OPPONENT_MOVE_PROBABILITY):
                    self.game.make_random_move()
                else:
                    self.game.place_optimal(self.opponent_model(self.game.get_board_state()))

            #after_move_score = game.get_player_num_tokens (LEARNING_MODEL_PLAYER)
            self.reward_dataset[self.num_trained_moves%REPLAY_BUFFER_SIZE] = 0 #after_move_score - before_move_score
            self.is_finished_dataset[self.num_trained_moves%REPLAY_BUFFER_SIZE] = self.game.is_finished()
            self.next_state_dataset[self.num_trained_moves%REPLAY_BUFFER_SIZE] = self.game.get_board_state()
            self.next_state_valid_mask_dataset[self.num_trained_moves%REPLAY_BUFFER_SIZE] = self.game.get_possibility_inf_mask()

            if (self.game.is_finished()):
                if self.game.get_winner() == self.LEARNING_MODEL_PLAYER:
                    self.reward_dataset[self.num_trained_moves%REPLAY_BUFFER_SIZE] += WINNING_REWARD
                elif self.game.get_winner() == (self.LEARNING_MODEL_PLAYER^1):
                    self.reward_dataset[self.num_trained_moves%REPLAY_BUFFER_SIZE] += LOSING_REWARD

                self.game = randomize_game()
                self.LEARNING_MODEL_PLAYER = self.game.current_player

            self.num_trained_moves += 1

            if self.num_trained_moves < LEARNING_BATCH_SIZE:
                continue

            indices = random.sample (range(min(self.num_trained_moves, REPLAY_BUFFER_SIZE)), LEARNING_BATCH_SIZE)
            targets = torch.max (self.target_model (self.next_state_dataset[indices]) + self.next_state_valid_mask_dataset[indices], dim=1)[0]
            targets[self.is_finished_dataset[indices]] = 0
            targets += self.reward_dataset[indices]
            outputs = model(self.state_dataset[indices])[torch.arange (LEARNING_BATCH_SIZE),self.action_dataset[indices]]
            loss = torch.nn.functional.mse_loss (outputs, targets)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

def main ():

    model = create_model()
    model.to(DEVICE)
    optim = torch.optim.AdamW (model.parameters())
    trainer: AI_trainer = AI_trainer(target_net_delay=300, opponent_model_delay=5000)

    if os.path.exists('model_weights.pth'):
        model.load_state_dict(torch.load('model_weights.pth', weights_only=True))

    torch.manual_seed (0);
    accuracy = test_model_DQN (model, 100)
    print ('start accuracy:', accuracy)
    while (True):
        torch.manual_seed (0);
        trainer.train (model, 20000, optim)
        accuracy = test_model_DQN (model, 100)
        print ('end accuracy:', accuracy)
        torch.save(model.state_dict(), 'model_weights.pth')


main()
