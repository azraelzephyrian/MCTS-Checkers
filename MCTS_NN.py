# MCTS_NN.py

import math
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
from network_v3 import CheckersNet, encode_board, move_to_index, index_to_move

# from game import CheckersGame     # Your checkers environment
# from network import CheckersNet, encode_board, move_to_index, index_to_move

EMPTY = 0
BLACK_MAN = 1
BLACK_KING = 2
RED_MAN = -1
RED_KING = -2

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
        self.children: Dict[
            Tuple[int, int, int, int, Tuple[Tuple[int,int], ...]],
            "Node"
        ] = {}
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

    def __init__(self, net, c_puct: float = 1.0, action_size: int = 128):
        """
        :param net: A PyTorch model with signature: net(board_input) -> (policy_logits, value)
                    - policy_logits shape: (batch_size, action_size)
                    - value shape: (batch_size, 1) in [-1, +1]
        :param c_puct: Exploration constant in the PUCT formula.
        :param action_size: The size of the policy output (e.g., 128 or 4096).
        """
        self.net = net
        self.c_puct = c_puct
        self.action_size = action_size

    def search(self, root: Node, n_simulations: int = 800) -> Tuple[int, int, int, int, List[Tuple[int,int]]]:
        """
        Perform MCTS from the given 'root' node, for 'n_simulations' iterations.
        After that, return the move leading to the most visited child node.
        """
        for _ in range(n_simulations):
            node = self._select(root)
            if not node.state.is_game_over():
                self._expand(node)
            # No random rollout. The value is from node.value_est (assigned in expand).
            # We interpret node.value_est from the perspective of the node's current player or root.
            self._backpropagate(node, node.value_est)

        # Choose the child with the highest visit count for the final move.
        best_move, best_child = self._best_child(root, explore=False)
        return best_move

    def _select(self, node: Node) -> Node:
        """
        Selection phase: descend the tree by choosing child nodes with the highest PUCT,
        until we reach a leaf node or a terminal state.
        """
        while not node.is_leaf() and not node.state.is_game_over():
            move, node = self._best_child(node, explore=True)
        return node

    def _expand(self, node: Node):
        """
        Expand the node by:
        1) Calling the neural net to get (policy_logits, value_est).
        2) Masking invalid moves to get a distribution over legal moves.
        3) Creating child nodes for each legal move, storing policy_prior in each child.
        4) Storing the node's own value_est from the NN for later backpropagation.
        """
        # 1) Encode the board state into a tensor
        board_input = encode_board(node.state).unsqueeze(0)  # shape (1, 4, 8, 8) for example
        # Move to the same device as the net
        board_input = board_input.to(next(self.net.parameters()).device)

        # 2) Forward pass
        with torch.no_grad():
            policy_logits, value_est = self.net(board_input)  # shapes: (1, action_size), (1, 1)

        # Extract from batch dimension
        policy_logits = policy_logits[0]  # shape (action_size,)
        node.value_est = value_est.item() # a single scalar in [-1, +1]

        # 3) Gather legal moves
        legal_moves = node.state.get_legal_moves()  # e.g. list of (r1, c1, r2, c2, captures)
        if not legal_moves:
            # No moves => terminal. Usually we do nothing. Node is a leaf.
            return

        # Convert each move to an index
        legal_indices = [move_to_index(m) for m in legal_moves]  # each index is in [0, action_size)
        
        # 4) Mask invalid moves => build a mask tensor
        mask = torch.full_like(policy_logits, float('-inf'))  # same shape as policy_logits
        for idx in legal_indices:
            mask[idx] = 0.0  # valid move => 0, invalid => -inf

        masked_logits = policy_logits + mask
        policy_probs = F.softmax(masked_logits, dim=0)  # shape (action_size,)

        # 5) Create child nodes for each legal move, storing policy_prior
        for move, idx in zip(legal_moves, legal_indices):
            p = policy_probs[idx].item()
            # Clone game state for the child
            new_state = node.state.clone()
            new_state.make_move(move)
            child_node = Node(state=new_state, parent=node)
            child_node.policy_prior = p

            # Move must be hashable to store as a dict key:
            # (r1,c1,r2,c2, tuple_of_captures)
            move_key = (move[0], move[1], move[2], move[3], tuple(move[4]))
            node.children[move_key] = child_node

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

    def _best_child(self, node: Node, explore: bool) -> Tuple[
                    Tuple[int, int, int, int, Tuple[Tuple[int,int], ...]], Node]:
        """
        Return (move, child_node) with the highest PUCT if explore=True,
        or highest visit_count if explore=False.
        """
        best_move = None
        best_node = None
        best_score = float('-inf')

        # sum_child_visits is needed for PUCT
        sum_visits = sum(child.visit_count for child in node.children.values()) + 1e-8

        for move, child in node.children.items():
            if child.visit_count == 0:
                q = 0.0
            else:
                q = child.total_value / child.visit_count  # average value

            if explore:
                # PUCT formula
                u = self.c_puct * child.policy_prior * (math.sqrt(sum_visits) / (1 + child.visit_count))
                score = q + u
            else:
                # For final move selection, pick the child with highest visit_count
                score = child.visit_count

            if score > best_score:
                best_score = score
                best_move = move
                best_node = child

        return best_move, best_node