# MCTS_NN.py

import math
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
from network_v1 import CheckersNet, encode_board, move_to_index, index_to_move

# from game import CheckersGame     # Your checkers environment
# from network import CheckersNet, encode_board, move_to_index, index_to_move






EMPTY = 0
BLACK_MAN = 1
BLACK_KING = 2
RED_MAN = -1
RED_KING = -2

def apply_move_path(game, path):
    """
    Applies a full move path (including multi-jumps) to the game object.
    Returns True if successful, False if illegal.
    """
    if path is None or len(path) < 2:
        return False

    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        mid_r = (r1 + r2) // 2
        mid_c = (c1 + c2) // 2
        dr, dc = r2 - r1, c2 - c1

        piece = game.get_piece(r1, c1)

        # Regular move
        if abs(dr) == 1 and abs(dc) == 1:
            if game.get_piece(r2, c2) != EMPTY:
                return False
            game.set_piece(r2, c2, piece)
            game.set_piece(r1, c1, EMPTY)

        # Jump move
        elif abs(dr) == 2 and abs(dc) == 2:
            mid_piece = game.get_piece(mid_r, mid_c)
            if not game.is_opponent(piece, mid_piece):
                return False
            if game.get_piece(r2, c2) != EMPTY:
                return False
            game.set_piece(r2, c2, piece)
            game.set_piece(r1, c1, EMPTY)
            game.set_piece(mid_r, mid_c, EMPTY)

        else:
            return False

        game._maybe_king(r2, c2)

    game.switch_player()
    return True

class Node:
    """
    A node in the MCTS tree.
    - state: the CheckersGame object.
    - parent: parent Node in the search tree.
    - children: { move -> child Node } dictionary, where move is hashable.
    - visit_count: how many times this node was visited during MCTS.
    - total_value: sum of simulation (value) results passing through this node.
    - policy_prior: probability prior from the NN for choosing this node's move from its parent.
    - value_est: the value returned by the NN the first time we expand this node.
                 (Optional, you can store in the parent or handle differently.)
    """
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children: Dict[int, "Node"] = {}  # ✅ children keyed by action_id
        self.visit_count = 0
        self.total_value = 0.0
        self.policy_prior = 0.0
        self.value_est = 0.0

    def is_leaf(self) -> bool:
        """Leaf node if it has no children."""
        return len(self.children) == 0

    def is_root(self) -> bool:
        """Root node if it has no parent."""
        return self.parent is None

    def __repr__(self):
        return (f"Node(visits={self.visit_count}, value={self.total_value}, "
                f"children={len(self.children)}, policy_prior={self.policy_prior:.4f})")


class MCTS_NN:
    """
    An MCTS class guided by a neural network for policy (move priors) and value
    (position evaluation). We use a PUCT-like formula for selection.
    """

    def __init__(self, net, encoder, c_puct: float = 1.0):
        """
        :param net: A PyTorch model with signature: net(board_input) -> (policy_logits, value)
                    - policy_logits shape: (batch_size, action_size)
                    - value shape: (batch_size, 1) in [-1, +1]
        :param c_puct: Exploration constant in the PUCT formula.
        :param action_size: The size of the policy output (e.g., 128 or 4096).
        """
        self.encoder = encoder
        self.net = net
        self.c_puct = c_puct
        self.action_size = len(encoder)

    def search(self, root: Node, encoder, n_simulations: int = 800) -> int:
        for _ in range(n_simulations):
            node = self._select(root, encoder)  # ✅ CORRECT ORDER

            if not node.state.is_game_over():
                self._expand(node, encoder)
            self._backpropagate(node, node.value_est)

        best_action_id, _ = self._best_child(root, explore=False)
        return best_action_id  # ✅ return action_id instead of move

    def _select(self, node: Node, encoder) -> Node:

        """
        Selection phase: descend the tree by choosing child nodes with the highest PUCT,
        until we reach a leaf node or a terminal state.
        """
        while not node.is_leaf() and not node.state.is_game_over():
            move, node = self._best_child(node, explore=True)
        return node

    def _expand(self, node, encoder):
        board_input = encode_board(node.state).unsqueeze(0).to(next(self.net.parameters()).device)

        with torch.no_grad():
            policy_logits, value_est = self.net(board_input)

        policy_logits = policy_logits[0]
        node.value_est = value_est.item()

        legal_moves = node.state.get_legal_moves()
        move_paths = []
        for move in legal_moves:
            r1, c1, r2, c2, captures = move
            path = [(r1, c1)] + captures + [(r2, c2)] if captures else [(r1, c1), (r2, c2)]
            move_paths.append(path)

        legal_ids = [encoder.encode(path) for path in move_paths]

        mask = torch.full_like(policy_logits, float('-inf'))
        valid_ids = [idx for idx in legal_ids if 0 <= idx < policy_logits.shape[0]]
        if not valid_ids:
            print("[Warning] No valid actions in policy head for this board state.")
            return

        if not valid_ids:
            return  # ✅ Skip expansion — no valid actions in this node

        mask = torch.full_like(policy_logits, float('-inf'))
        for idx in valid_ids:
            mask[idx] = 0.0

        masked_logits = policy_logits + mask
        policy_probs = F.softmax(masked_logits, dim=0)


        masked_logits = policy_logits + mask
        policy_probs = F.softmax(masked_logits, dim=0)

        for path, action_id in zip(move_paths, legal_ids):
            if action_id not in valid_ids:
                continue  # skip out-of-bounds actions
            new_state = node.state.clone()
            if not apply_move_path(new_state, path):
                continue
            child_node = Node(state=new_state, parent=node)
            child_node.policy_prior = policy_probs[action_id].item()
            node.children[action_id] = child_node



    def _backpropagate(self, node: Node, value: float):
        """
        Backpropagate the value up the tree. By default, we treat 'value'
        as from the perspective of the node's current player. If you want to
        keep it from the root perspective, you can flip signs each step.
        """
        current = node
        # If you want to flip perspective each step, do something like:
        # sign = 1.0
        while current is not None:
            current.visit_count += 1
            # Add the value from the node's perspective
            current.total_value += value
            # sign = -sign
            current = current.parent

    def _best_child(self, node: Node, explore: bool) -> Tuple[int, Node]:
        best_id = None
        best_node = None
        best_score = float('-inf')

        sum_visits = sum(child.visit_count for child in node.children.values()) + 1e-8

        for action_id, child in node.children.items():
            q = child.total_value / child.visit_count if child.visit_count > 0 else 0.0
            if explore:
<<<<<<< Updated upstream
                # PUCT formula
                u = self.c_puct * child.policy_prior * (math.sqrt(sum_visits) / (1 + child.visit_count))
=======
                u = self.c_puct * child.policy_prior * math.sqrt(sum_visits) / (1 + child.visit_count)
>>>>>>> Stashed changes
                score = q + u
            else:
                score = child.visit_count

            if score > best_score:
                best_score = score
                best_id = action_id
                best_node = child

        return best_id, best_node
