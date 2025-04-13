# engine_api.py

from game import CheckersGame
from MCTS_NN import MCTS_NN, Node
from network_v3 import CheckersNet
import torch
import numpy as np
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
        net.load_state_dict(torch.load("latest_iteration.pth", map_location=device))
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
    print(f"Current board state before move:")
    game.print_board()
    
    legal_moves = game.get_legal_moves()
    print("Legal moves:", legal_moves)
    
    # Find all possible moves matching the coordinates
    matching_moves = []
    for move in legal_moves:
        sr, sc, er, ec, captures = move
        if sr == r1 and sc == c1 and er == r2 and ec == c2:
            matching_moves.append(move)
    
    if len(matching_moves) > 1:
        print(f"WARNING: Multiple moves match these coordinates: {matching_moves}")
        # Select the move with the most captures (likely the correct one)
        selected_move = max(matching_moves, key=lambda m: len(m[4]))
        print(f"Selected move with most captures: {selected_move}")
    elif len(matching_moves) == 1:
        selected_move = matching_moves[0]
    else:
        return {"success": False, "error": "Illegal move!"}
    
    print(f"Executing move: {selected_move}")
    game.make_move(selected_move)
    print("Board state after move:")
    game.print_board()
    
    move_key = (selected_move[0], selected_move[1], selected_move[2], selected_move[3], tuple(selected_move[4]))
    if move_key in root.children:
        root = root.children[move_key]
        root.parent = None
    else:
        # Log when tree knowledge is lost
        print(f"Move not found in MCTS tree, creating new node")
        root = Node(state=game)
    
    return {"success": True}

def validate_board_state():
    """Validate that the board is in a consistent state"""
    if game is None:
        return True
        
    # Check piece counts
    black_count = sum(game.is_black_piece(game.get_piece(r, c)) for r in range(8) for c in range(8))
    red_count = sum(game.is_red_piece(game.get_piece(r, c)) for r in range(8) for c in range(8))
    
    # Check that pieces are on valid squares (should be on squares where r+c is even)
    valid_squares = True
    for r in range(8):
        for c in range(8):
            if game.get_piece(r, c) != 0 and (r+c) % 2 != 0:
                valid_squares = False
                print(f"Error: Piece at ({r},{c}) on invalid square!")
    
    print(f"Board validation: Black pieces: {black_count}, Red pieces: {red_count}, Valid squares: {valid_squares}")
    return valid_squares and (black_count <= 12 and red_count <= 12)


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