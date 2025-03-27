# main.py

from game import CheckersGame
from MCTS_NN import MCTS_NN, Node
from network_v1 import CheckersNet, encode_board
import torch

def play_user_vs_nn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = CheckersNet(action_size=8**4)
    net.to(device)
    net.load_state_dict(torch.load("checkers_net.pth", map_location=device))
    net.eval()

    nn_mcts = MCTS_NN(net=net, c_puct=1.0, action_size=8**4)

    game = CheckersGame()
    root = Node(state=game)

    while not game.is_game_over():
        game.print_board()
        if game.current_player in (1, 2):  # Black's turn (NN)
            print("NN is thinking...")
            move = nn_mcts.search(root, n_simulations=100)
        else:  # Red's turn (User)
            print("Your turn! Enter your move as: start_row start_col end_row end_col")
            user_input = input("Move: ")
            try:
                r1, c1, r2, c2 = map(int, user_input.split())
                captures = []  # For now, assume user inputs non-capturing moves
                move = (r1, c1, r2, c2, captures)

                if move not in game.get_legal_moves():
                    raise ValueError("Illegal move!")
            except Exception as e:
                print("Invalid input. Please try again.")
                continue

        game.make_move(move)

        move_key = (move[0], move[1], move[2], move[3], tuple(move[4]))
        if move_key in root.children:
            root = root.children[move_key]
            root.parent = None
        else:
            root = Node(state=game)

    game.print_board()
    winner = game.get_winner()
    if winner == 1:
        print("NN wins!")
    elif winner == -1:
        print("You win!")
    else:
        print("It's a draw!")


if __name__ == "__main__":
    play_user_vs_nn()