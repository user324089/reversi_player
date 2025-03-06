import torch
from reversi import *
from model import *
from tqdm import tqdm
from common import *
import os

BEGINNING_RANDOM_MOVE_MAX_COUNT = 40

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
            game: Reversi = randomize_game(BEGINNING_RANDOM_MOVE_MAX_COUNT+1)
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


def main ():
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


if __name__ == '__main__':
    main ()
