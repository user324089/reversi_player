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

class AI_trainer_DQN:
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

            move_taken = self.game.place_optimal (q_scores)

            self.action_dataset[self.num_trained_moves % REPLAY_BUFFER_SIZE] = move_taken

            while ((not self.game.is_finished()) and self.game.current_player != self.LEARNING_MODEL_PLAYER):
                self.game.make_random_move()

            after_move_score = self.game.get_player_num_tokens (self.LEARNING_MODEL_PLAYER)
            self.reward_dataset[self.num_trained_moves%REPLAY_BUFFER_SIZE] = after_move_score - before_move_score
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

def test_model_policy (model: Reversi_AI_policy, num_games: int):
    num_won: float = 0
    for _ in tqdm(range (num_games)):
        game: Reversi = Reversi()
        model_player = game.current_player
        while not game.is_finished():
            if (game.current_player == model_player):
                game.place_from_probabilities(model(game.get_board_state(), game.get_possibility_inf_mask()))
            else:
                game.place_from_probabilities(torch.ones(SIDE**2))
        if (game.get_winner () == model_player):
            num_won += 1
    return num_won / num_games


def train_AI_policy (model: Reversi_AI_policy, num_epochs: int, games_per_epoch, optimiser: torch.optim.Optimizer):
    for _ in tqdm(range (num_epochs)):

        optimiser.zero_grad()

        loss: torch.Tensor = torch.tensor(0, dtype=torch.float)

        moves_made: float = 0

        for _ in range (games_per_epoch):
            game: Reversi = randomize_game()
            states: list[tuple[int, torch.Tensor]] = []
            log_probs: list[torch.Tensor] = []
            while (not game.is_finished()):
                moves_made += 1
                states.append (game.get_game_state())
                board_state = game.get_board_state()
                move_probabilities = model(board_state, game.get_possibility_inf_mask())
                _, log_prob = game.place_from_probabilities (move_probabilities)

                log_probs.append (log_prob)

            _, end_state = game.get_game_state()

            for j, (log_prob, (current_player, current_state)) in enumerate(zip (log_probs, states, strict=True)):
                baseline = torch.tensor((STARTING_NUM_FREE_FIELDS - j)/2)
                total_reward: float = float(end_state[current_player] - current_state[current_player] - baseline)

                loss -= log_prob * total_reward

        loss /= moves_made

        loss.backward()

        optimiser.step()

def main_DQN ():
    model = create_model_DQN()
    model.to(DEVICE)
    optim = torch.optim.AdamW (model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.95)
    trainer: AI_trainer_DQN = AI_trainer_DQN(target_net_delay=300, opponent_model_delay=5000)

    if os.path.exists('model_weights_dqn.pth'):
        model.load_state_dict(torch.load('model_weights_dqn.pth', weights_only=True))

    accuracy = test_model_DQN (model, 100)
    print ('start accuracy:', accuracy)
    while (True):
        print ('current lr:', scheduler.get_last_lr())
        trainer.train (model, 20000, optim)
        scheduler.step()
        accuracy = test_model_DQN (model, 100)
        print ('end accuracy:', accuracy)
        torch.save(model.state_dict(), 'model_weights_dqn.pth')

def main_policy ():
    model = create_model_policy()
    optim = torch.optim.AdamW (model.parameters())

    if os.path.exists('model_weights_policy.pth'):
        model.load_state_dict(torch.load('model_weights_policy.pth', weights_only=True))

    accuracy = test_model_policy (model, 100)
    print ('start accuracy:', accuracy)
    while (True):
        train_AI_policy (model, 100, 10, optim)
        accuracy = test_model_policy (model, 100)
        print ('end accuracy:', accuracy)
        torch.save(model.state_dict(), 'model_weights_policy.pth')

def main ():
    #main_policy()
    main_DQN()


main()
