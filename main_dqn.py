from model import *
from tqdm import tqdm
import copy
import random
from common import *
import os
import dotenv
import sys

# default values
BEGINNING_RANDOM_MOVE_MAX_COUNT = 40

REPLAY_BUFFER_SIZE = 3000
LEARNING_BATCH_SIZE = 500

WINNING_REWARD = 10
LOSING_REWARD = -10

STARTING_LEARNING_RATE = 5e-6
LEARNING_RATE_DECREASE_EXPONENT = 0.9

TARGET_NET_DELAY = 1000

NUM_OPPONENT_SUGGESTED_FIELDS = 3
OPPONENT_SUGGESTION_STRENGTH=0.5
SELF_PLAY_THRESHOLD = 0.9

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
    def __init__(self, target_net_delay: int):
        self.target_net_delay: int = target_net_delay
        self.num_trained_moves: int = 0

        self.state_dataset = torch.empty(REPLAY_BUFFER_SIZE, 2*SIDE*SIDE).to(DEVICE)
        self.next_state_dataset: torch.Tensor = torch.empty(REPLAY_BUFFER_SIZE, 2*SIDE*SIDE).to(DEVICE)
        self.next_state_valid_mask_dataset: torch.Tensor = torch.empty(REPLAY_BUFFER_SIZE, SIDE*SIDE).to(DEVICE)
        self.action_dataset = torch.zeros (REPLAY_BUFFER_SIZE, dtype=torch.int).to(DEVICE)
        self.reward_dataset = torch.zeros (REPLAY_BUFFER_SIZE).to(DEVICE)
        self.is_finished_dataset = torch.zeros(REPLAY_BUFFER_SIZE, dtype=torch.bool).to(DEVICE)

        self.game = Reversi (device=DEVICE)
        self.LEARNING_MODEL_PLAYER = self.game.current_player


    def train (self, model: Reversi_AI_DQN, num_moves: int, optimiser: torch.optim.Optimizer, is_self_play: bool):
        for _ in tqdm(range (num_moves)):

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
                if is_self_play:
                    enemy_q_scores = model (self.game.get_board_state())
                    enemy_q_scores += self.game.get_possibility_inf_mask ()
                    suggested_fields = (enemy_q_scores >= torch.topk (enemy_q_scores, NUM_OPPONENT_SUGGESTED_FIELDS)[0][-1]).to(torch.float)
                    probabilities = (torch.ones (SIDE*SIDE) + suggested_fields * OPPONENT_SUGGESTION_STRENGTH).to(DEVICE)
                    self.game.place_from_probabilities (probabilities)
                else:
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

def load_env_variables ():
    dotenv.load_dotenv()

    try:
        win_reward = os.getenv ('DQN_WIN_REWARD')
        if win_reward is not None:
            global WINNING_REWARD
            WINNING_REWARD = int(win_reward)
        lose_reward = os.getenv ('DQN_LOSE_REWARD')
        if lose_reward is not None:
            global LOSING_REWARD
            LOSING_REWARD = int(lose_reward)
        random_move_count = os.getenv ('DQN_RANDOM_MOVE_COUNT')
        if random_move_count is not None:
            global BEGINNING_RANDOM_MOVE_MAX_COUNT
            BEGINNING_RANDOM_MOVE_MAX_COUNT = int(random_move_count)
        replay_buffer_size = os.getenv ('DQN_REPLAY_BUFFER_SIZE')
        if replay_buffer_size is not None:
            global REPLAY_BUFFER_SIZE
            REPLAY_BUFFER_SIZE = int(replay_buffer_size)
        learning_batch_size = os.getenv ('DQN_BATCH_SIZE')
        if learning_batch_size is not None:
            global LEARNING_BATCH_SIZE
            LEARNING_BATCH_SIZE = int (learning_batch_size)
        start_learning_rate = os.getenv ('DQN_START_LEARNING_RATE')
        if start_learning_rate is not None:
            global STARTING_LEARNING_RATE
            STARTING_LEARNING_RATE = float(start_learning_rate)
        learning_rate_decrease_exponent = os.getenv ('DQN_LR_DECREASE_EXPONENT')
        if learning_rate_decrease_exponent is not None:
            global LEARNING_RATE_DECREASE_EXPONENT
            LEARNING_RATE_DECREASE_EXPONENT = float (learning_rate_decrease_exponent)
        target_net_delay = os.getenv ('DQN_TARGET_NET_DELAY')
        if target_net_delay is not None:
            global TARGET_NET_DELAY
            TARGET_NET_DELAY = int(target_net_delay)
        num_opponent_suggested_fields = os.getenv ('DQN_NUM_OPPONENT_SUGGESTED_FIELDS')
        if num_opponent_suggested_fields is not None:
            global NUM_OPPONENT_SUGGESTED_FIELDS
            NUM_OPPONENT_SUGGESTED_FIELDS = int(num_opponent_suggested_fields)
        opponent_suggestion_strength = os.getenv ('DQN_OPPONENT_SUGGESTION_STRENGTH')
        if opponent_suggestion_strength is not None:
            global OPPONENT_SUGGESTION_STRENGTH
            OPPONENT_SUGGESTION_STRENGTH = float(opponent_suggestion_strength)
        self_play_threshold = os.getenv ('DQN_SELF_PLAY_THRESHOLD')
        if self_play_threshold is not None:
            global SELF_PLAY_THRESHOLD
            SELF_PLAY_THRESHOLD = float (SELF_PLAY_THRESHOLD)

    except ValueError:
        print ('failed to parse env variables')
        sys.exit (1)

def main ():

    load_env_variables ()

    model = create_model_DQN()
    model.to(DEVICE)
    optim = torch.optim.AdamW (model.parameters(), lr=STARTING_LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=LEARNING_RATE_DECREASE_EXPONENT)
    trainer: AI_trainer_DQN = AI_trainer_DQN(target_net_delay=TARGET_NET_DELAY)

    if os.path.exists('model_weights_dqn.pth'):
        model.load_state_dict(torch.load('model_weights_dqn.pth', weights_only=True))

    is_self_play = False

    accuracy = test_model_DQN (model, 1000)
    if (accuracy > SELF_PLAY_THRESHOLD):
        is_self_play = True

    print ('start accuracy:', accuracy)
    while (True):
        print ('current lr:', scheduler.get_last_lr())
        trainer.train (model, 20000, optim, is_self_play)
        scheduler.step()
        accuracy = test_model_DQN (model, 1000)
        print ('end accuracy:', accuracy)
        if (accuracy > SELF_PLAY_THRESHOLD):
            is_self_play = True
        torch.save(model.state_dict(), 'model_weights_dqn.pth')

if __name__ == '__main__':
    main ()
