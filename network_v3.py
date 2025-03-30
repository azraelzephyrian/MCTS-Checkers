import torch
import torch.nn as nn
import torch.nn.functional as F

# If you have these constants from your game.py, import them. Otherwise define them:
EMPTY = 0
BLACK_MAN = 1
BLACK_KING = 2
RED_MAN = -1
RED_KING = -2

class CheckersNet(nn.Module):
    """
    A simple network for Checkers with separate policy and value heads.
    - Input:  (batch_size, 5, 8, 8)
        * 5 channels: [black_man, black_king, red_man, red_king, current_player]
    - Output: (policy, value)
      * policy: (batch_size, action_size) in log-prob space
      * value:  (batch_size, 1) in [-1, +1]
    """

    def __init__(self, board_size=8, in_channels=5, num_filters=64, action_size=128):
        """
        :param board_size: Size of the board (8 for checkers).
        :param in_channels: Number of input channels. Now 5 (4 piece planes + 1 player plane).
        :param num_filters: Number of feature maps in the conv layers.
        :param action_size: Number of distinct actions your policy head should output.
                            For instance, 8*8*possible_moves or a flattened approach.
        """
        super(CheckersNet, self).__init__()

        # Common trunk (two simple convolutional layers with batch normalization)
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)  # Add batch normalization
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)  # Add batch normalization

        # Policy head
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * board_size * board_size, action_size)

        # Value head
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(board_size * board_size, 128)  # Increased from 64
        self.value_fc2 = nn.Linear(128, 64)                      # Added an intermediate layer
        self.value_fc3 = nn.Linear(64, 1)                        # Final output layer

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Custom weight initialization."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        :param x: A tensor of shape (batch_size, 5, 8, 8)
                  where x[:,4,:,:] is the "current player" plane (1 or 0).
        :return: (policy, value)
            policy: log probabilities over 'action_size' moves
            value:  scalar in [-1, +1]
        """

        # Common trunk with batch normalization
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Policy head
        p = F.relu(self.policy_conv(x))               # shape: (batch_size, 2, 8, 8)
        p = p.view(p.size(0), -1)                     # flatten to (batch_size, 2*8*8)
        p = self.policy_fc(p)                         # (batch_size, action_size)
        policy = F.log_softmax(p, dim=1)              # log-probs

        # Value head
        v = F.relu(self.value_conv(x))                # shape: (batch_size, 1, 8, 8)
        v = v.view(v.size(0), -1)                     # flatten to (batch_size, 8*8)
        v = F.relu(self.value_fc1(v))                 # (batch_size, 128)
        v = F.relu(self.value_fc2(v))                 # (batch_size, 64)
        v = torch.tanh(self.value_fc3(v))             # (batch_size, 1) range ~[-1, +1]

        return policy, v
    
def encode_board(game) -> torch.Tensor:
    """
    Encode the CheckersGame board into a 5×8×8 float tensor:
        channel 0: black men
        channel 1: black kings
        channel 2: red men
        channel 3: red kings
        channel 4: current player (1.0 if it's black's turn, 0.0 otherwise)

    :param game: an instance of CheckersGame
    :return: A torch float32 tensor of shape (5, 8, 8)
    """
    board_tensor = torch.zeros((5, 8, 8), dtype=torch.float32)

    for r in range(8):
        for c in range(8):
            piece = game.board[r][c]
            if piece == BLACK_MAN:
                board_tensor[0, r, c] = 1.0
            elif piece == BLACK_KING:
                board_tensor[1, r, c] = 1.0
            elif piece == RED_MAN:
                board_tensor[2, r, c] = 1.0
            elif piece == RED_KING:
                board_tensor[3, r, c] = 1.0
            # else it's EMPTY (0)

    # Mark the current player's plane
    # If black is to move, fill channel 4 with 1.0, else leave it 0.0
    if game.current_player in (BLACK_MAN, BLACK_KING):
        board_tensor[4, :, :] = 1.0

    return board_tensor

def move_to_index(move: tuple, board_size=8) -> int:
    """
    Map a move (r1, c1, r2, c2, captures) -> integer in [0, board_size^4).
    Ignores the captures for indexing.
    
    Example:
       index = r1*(8^3) + c1*(8^2) + r2*8 + c2
    
    :param move: (r1, c1, r2, c2, captures)
    :param board_size: typically 8 for standard checkers
    :return: integer index in [0, board_size^4)
    """
    (r1, c1, r2, c2, captures) = move
    return r1*(board_size**3) + c1*(board_size**2) + r2*board_size + c2

def index_to_move(index: int, board_size=8):
    """
    Reverse of move_to_index. (Assumes no captures in indexing.)
    Returns (r1, c1, r2, c2, []).
    """
    r1 = index // (board_size**3)
    remain = index % (board_size**3)

    c1 = remain // (board_size**2)
    remain = remain % (board_size**2)

    r2 = remain // board_size
    c2 = remain % board_size

    return (r1, c1, r2, c2, [])


if __name__ == "__main__":
    # Quick test
    net = CheckersNet(board_size=8, in_channels=5, num_filters=64, action_size=128)

    # Fake batch input: batch_size=2, shape (2,5,8,8)
    x = torch.zeros((2, 5, 8, 8), dtype=torch.float32)
    # Let the second plane's "player plane" differ for demonstration:
    x[0,4,:,:] = 1.0  # e.g. black to move
    x[1,4,:,:] = 0.0  # e.g. red to move

    policy, value = net(x)

    print("Policy shape:", policy.shape)  # (2, 128)
    print("Value shape:", value.shape)    # (2, 1)
    # e.g.:
    # Policy shape: torch.Size([2, 128])
    # Value shape: torch.Size([2, 1])