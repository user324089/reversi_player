import torch
from reversi import SIDE

class Linear_skip_block (torch.nn.Module):

    def __init__ (self, num_neurons: int) -> None:
        super().__init__()
        self.first_layers = torch.nn.Sequential (
                torch.nn.Linear (num_neurons, num_neurons),
                torch.nn.ReLU(),
                torch.nn.Linear (num_neurons, num_neurons)
                )
        self.relu = torch.nn.ReLU()

    def forward (self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.first_layers (x))

class Reversi_AI_DQN (torch.nn.Module):

    def __init__ (self, num_blocks: int, hidden_layer_width: int) -> None:
        super().__init__()

        self.first_layer = torch.nn.Sequential(
                torch.nn.Linear (SIDE*SIDE*2, hidden_layer_width),
                torch.nn.ReLU()
                )

        self.middle_layers = torch.nn.Sequential (
                *[
                    Linear_skip_block (hidden_layer_width)
                    for _ in range (num_blocks)
                    ]
                )

        self.last_layer = torch.nn.Linear (hidden_layer_width, SIDE*SIDE)

    def forward (self, x) -> torch.Tensor:
        x = self.first_layer (x)
        x = self.middle_layers(x)
        x = self.last_layer (x)
        return x

class Reversi_AI_policy (torch.nn.Module):

    def __init__ (self, num_blocks: int, hidden_layer_width: int) -> None:
        super().__init__()

        self.first_layer = torch.nn.Sequential(
                torch.nn.Linear (SIDE*SIDE*2, hidden_layer_width),
                torch.nn.ReLU()
                )

        self.middle_layers = torch.nn.Sequential (
                *[
                    Linear_skip_block (hidden_layer_width)
                    for _ in range (num_blocks)
                    ]
                )

        self.last_layer = torch.nn.Linear (hidden_layer_width, SIDE*SIDE)
        self.soft_max = torch.nn.Softmax (dim=0)

    def forward (self, x, mask) -> torch.Tensor:
        x = self.first_layer (x)
        x = self.middle_layers(x)
        x = self.last_layer (x)
        x += mask
        x = self.soft_max (x)
        return x

def create_model_policy () -> Reversi_AI_policy:
    return Reversi_AI_policy(2,300)

def create_model_DQN () -> Reversi_AI_DQN:
    return Reversi_AI_DQN(1,200)
