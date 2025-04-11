from game import CheckersGame, MoveEncoder, encode_board
from MCTS_NN import MCTS_NN, Node, apply_move_path
from network_v3 import CheckersNet
import torch
import pickle
from typing import List, Tuple


# Global variables
game = None
root = None
nn_mcts = None
net = None
encoder = MoveEncoder() # ✅ Add encoder

def init_game():
    global game, root, nn_mcts, net, encoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ Load encoder
    with open("C:/Users/alexi/OneDrive/Desktop/my_code_projects/MCTS-Checkers/outputs/move_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    # ✅ Use encoder size to init net
    if net is None:
        action_size = len(encoder)
        net = CheckersNet(action_size=action_size)
        net.to(device)
        net.load_state_dict(torch.load("C:/Users/alexi/OneDrive/Desktop/my_code_projects/MCTS-Checkers/latest_iteration.pth", map_location=device))
        net.eval()

    nn_mcts = MCTS_NN(net=net, c_puct=1.0, encoder=encoder)  # ✅ Match encoder

    game = CheckersGame()
    print("[Init] Current player:", game.current_player)  # ✅ Add this line
    root = Node(state=game)



def get_board_state():
    if game is None:
        return None
    return {
        "board": game.get_board_state(),
        "current_player": game.current_player
    }

def get_legal_moves():
    global game, encoder
    if game is None:
        return None

    legal_moves = game.get_legal_moves()
    paths = []
    for (r1, c1, r2, c2, captures) in legal_moves:
        if captures:
            path = [(r1, c1)] + captures + [(r2, c2)]
        else:
            path = [(r1, c1), (r2, c2)]
        paths.append(path)

    # 🧠 Rebuild encoder with latest paths
    if encoder:
        encoder.rebuild_from_paths(paths)

    return paths



def get_legal_moves():
    global game, encoder
    if game is None:
        return None

    legal_moves = game.get_legal_moves()
    paths = []

    for (r1, c1, r2, c2, captures) in legal_moves:
        path = [(r1, c1)]
        r_curr, c_curr = r1, c1

        for cr, cc in captures:
            dr, dc = cr - r_curr, cc - c_curr
            r_curr, c_curr = cr + dr, cc + dc
            path.append((r_curr, c_curr))

        if not captures:
            path = [(r1, c1), (r2, c2)]  # simple move

        paths.append(path)

    print("[Encoder Rebuild] Paths being registered:")
    for i, path in enumerate(paths):
        print(f"[{i}] {path}")

    if encoder:
        encoder.rebuild_from_paths(paths)

    return paths


def make_user_move(r1, c1, r2, c2):
    global game, root, encoder

    print(f"\n[User Move Attempt] ({r1}, {c1}) → ({r2}, {c2})")
    legal_moves = game.get_legal_moves()
    print(f"[Backend] Current player: {game.current_player}")
    print(f"[Backend] Legal moves ({len(legal_moves)}):")

    for move in legal_moves:
        sr, sc, er, ec, captures = move

        # Reconstruct full move path
        path = [(sr, sc)]
        r_curr, c_curr = sr, sc
        for cr, cc in captures:
            dr, dc = cr - r_curr, cc - c_curr
            r_curr, c_curr = cr + dr, cc + dc
            path.append((r_curr, c_curr))

        if not captures:
            path = [(sr, sc), (er, ec)]

        print(f"→ {path}")

        if path[0] == (r1, c1) and path[-1] == (r2, c2):
            print(f"[Match] Found path: {path}")
            move_applied = game.make_move(move)

            if not move_applied:
                return {"success": False, "error": "Move could not be applied"}

            action_id = encoder.encode(tuple(path))
            print(f"[Encode] action_id: {action_id}")

            if action_id in root.children:
                print("[Tree] Advancing root to matched child")
                root = root.children[action_id]
                root.parent = None
            else:
                print("[Tree] Resetting root node (not in children)")
                root = Node(state=game)

            return {"success": True}

    print("[No Match] Move did not match any known legal full paths.")
    return {"success": False, "error": "Invalid move or partial sequence"}





def make_user_move_path(path: List[Tuple[int, int]]):
    global game, root, encoder

    move_applied = apply_move_path(game, path)
    if not move_applied:
        return {"success": False, "error": "Illegal move!"}

    action_id = encoder.encode(tuple(tuple(p) for p in path))
    if action_id in root.children:
        root = root.children[action_id]
        root.parent = None
    else:
        root = Node(state=game)

    return {"success": True}



import torch.nn.functional as F

def make_nn_move(n_simulations=100):
    global game, encoder, root, net

    print("\n[NN Move] Called. Current player:", game.current_player)

    legal_paths = get_legal_moves()  # Returns full move paths (list of [(r, c), ..., (r, c)])
    if not legal_paths:
        print("[NN Move] No legal moves available.")
        return {"success": False, "error": "No legal moves available."}

    # ✅ Encode board into tensor
    try:
        board_tensor = encode_board(game)  # Should return [C, H, W]
        board_tensor = board_tensor.unsqueeze(0).to(next(net.parameters()).device)  # -> [1, C, H, W]
    except Exception as e:
        print(f"[NN Move] Failed to encode board: {e}")
        return {"success": False, "error": f"Board encoding failed: {e}"}

    # ✅ Forward pass
    try:
        net.eval()
        with torch.no_grad():
            policy_logits, value = net(board_tensor)  # Expected output: [1, action_space], [1, 1]
    except Exception as e:
        print(f"[NN Move] Net forward pass failed: {e}")
        return {"success": False, "error": f"Forward pass failed: {e}"}

    action_probs = F.softmax(policy_logits[0], dim=0)  # Get shape [action_space]

    # ✅ Encode all legal move paths into IDs
    legal_ids = []
    for path in legal_paths:
        try:
            legal_ids.append(encoder.encode(tuple(path)))
        except Exception as e:
            print(f"[Encoder] Failed to encode path {path}: {e}")

    # ✅ Filter valid legal moves (skip those outside bounds of action_probs)
    legal_probs = [(i, action_probs[i].item()) for i in legal_ids if i < len(action_probs)]

    if not legal_probs:
        print("[Warning] No valid legal moves found in policy head.")
        return {"success": False, "error": "No valid actions"}

    # ✅ Select highest-probability move
    selected_id = max(legal_probs, key=lambda x: x[1])[0]
    print("[NN Move] Selected action ID:", selected_id)

    try:
        print(f"[Debug] Encoder has {len(encoder)} paths")
        if selected_id >= len(encoder):
            print(f"[Error] Selected ID {selected_id} is out of range")
            return {"success": False, "error": f"Action ID {selected_id} not in encoder."}

        move_path = encoder.decode(selected_id)
        if not move_path:
            raise ValueError("Decoded move path is None or empty")
        print("[NN Move] Decoded move path:", move_path)
    except Exception as e:
        print(f"[NN Move] Failed to decode selected ID {selected_id}: {e}")
        return {"success": False, "error": "Could not decode action"}

    # ✅ Apply move
    move_applied = apply_move_path(game, move_path)
    if not move_applied:
        print(f"[NN Move] Failed to apply move path: {move_path}")
        return {"success": False, "error": "Move application failed"}

    # ✅ Update tree root
    if selected_id in root.children:
        root = root.children[selected_id]
        root.parent = None
        print("[Tree] Advanced to selected child node")
    else:
        root = Node(state=game)
        print("[Tree] Reset root (child not found)")

    return {"success": True, "nn_move": move_path}





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
