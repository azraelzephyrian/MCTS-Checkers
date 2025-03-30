# engine_api.py

from game import CheckersGame
from MCTS_NN import MCTS_NN, Node
from network_v3 import CheckersNet
import torch

# Global variables (can later be moved into per-session storage)
game = None
root = None
nn_mcts = None
net = None

def init_game():
    global game, root, nn_mcts, net

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if net is None:
        net = CheckersNet(action_size=8**4)
        net.to(device)
        net.load_state_dict(torch.load("C:/Users/alexi/OneDrive/Desktop/my_code_projects/MCTS-Checkers/latest_iteration.pth", map_location=device))
        net.eval()

    nn_mcts = MCTS_NN(net=net, c_puct=1.0, action_size=8**4)

    game = CheckersGame()
    root = Node(state=game)

def get_board_state():
    if game is None:
        return None
    return {
        "board": game.get_board_state(),  # You may need to define this method in CheckersGame
        "current_player": game.current_player
    }

def get_legal_moves():
    global game
    if game is None:
        return None
    return game.get_legal_moves()


def make_user_move(r1, c1, r2, c2):
    global game, root

    print(f"User move attempt: ({r1}, {c1}) -> ({r2}, {c2})")
    legal_moves = game.get_legal_moves()
    print("Legal moves:", legal_moves)
    for move in legal_moves:
        sr, sc, er, ec, captures = move
        if sr == r1 and sc == c1 and er == r2 and ec == c2:
            print("Match found. Executing move.")
            game.make_move(move)
            move_key = (sr, sc, er, ec, tuple(captures))
            if move_key in root.children:
                root = root.children[move_key]
                root.parent = None
            else:
                root = Node(state=game)
            return {"success": True}

    return {"success": False, "error": "Illegal move!"}





def make_nn_move(n_simulations=100):
    global game, root, nn_mcts

    if game.is_game_over():
        return {"success": False, "error": "Game is already over"}

    legal_moves = game.get_legal_moves()
    print("Legal moves:", legal_moves)

    # Get NN's proposed move (it might not contain captures if encoded poorly)
    proposed_move = nn_mcts.search(root, n_simulations=n_simulations)

    # Match the actual legal move based on coordinates to ensure proper captures
    r1, c1, r2, c2 = proposed_move[0], proposed_move[1], proposed_move[2], proposed_move[3]
    actual_move = None
    for move in legal_moves:
        sr, sc, er, ec, captures = move
        if sr == r1 and sc == c1 and er == r2 and ec == c2:
            actual_move = move
            break

    if actual_move is None:
        return {"success": False, "error": "NN proposed illegal move!"}

    game.make_move(actual_move)
    move_key = (actual_move[0], actual_move[1], actual_move[2], actual_move[3], tuple(actual_move[4]))

    if move_key in root.children:
        root = root.children[move_key]
        root.parent = None
    else:
        root = Node(state=game)

    return {"success": True, "nn_move": actual_move}


def get_game_status():
    if game is None:
        return {"status": "no_game"}

    if game.is_game_over():
        winner = game.get_winner()
        if winner == 1:
            return {"status": "over", "winner": "nn"}
        elif winner == -1:
            return {"status": "over", "winner": "user"}
        else:
            return {"status": "over", "winner": "draw"}
    else:
        return {"status": "ongoing", "current_player": game.current_player}
