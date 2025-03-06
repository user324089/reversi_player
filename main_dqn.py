from model import *
from tqdm import tqdm
import copy
import random
from common import *
import os

BEGINNING_RANDOM_MOVE_MAX_COUNT = 40

REPLAY_BUFFER_SIZE = 3000
LEARNING_BATCH_SIZE = 500

WINNING_REWARD = 40
LOSING_REWARD = -40

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

                self.game = randomize_game(BEGINNING_RANDOM_MOVE_MAX_COUNT)
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

if __name__ == '__main__':
    main ()
