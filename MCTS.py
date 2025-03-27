import math
import random
from typing import Optional, Dict, Tuple, List

# Assume we have a CheckersGame class from game.py
# from game import CheckersGame, ...

class Node:
    """
    A node in the MCTS tree.
    - state: the CheckersGame object (or any representation of the game state).
    - parent: parent Node in the search tree.
    - children: { move -> child Node } dictionary.
    - visit_count: how many times this node was visited during MCTS.
    - total_value: sum of simulation results passing through this node.
    - policy_prior: (optional) a neural-network-provided prior probability for choosing this node's move from its parent.
    """
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children: Dict[
            Tuple[int, int, int, int, List[Tuple[int,int]]],
            "Node"
        ] = {}
        self.visit_count = 0
        self.total_value = 0.0
        self.policy_prior = 0.0  # If using a neural net policy

    def is_leaf(self) -> bool:
        """Leaf node if it has no children."""
        return len(self.children) == 0

    def is_root(self) -> bool:
        """Root node if it has no parent."""
        return self.parent is None

    def __repr__(self):
        return (f"Node(visits={self.visit_count}, value={self.total_value}, "
                f"children={len(self.children)})")


class MCTS:
    """
    A basic MCTS class using UCT for selection, random rollouts, and
    backpropagation of results.
    """
    def __init__(self, c: float = 1.4):
        """
        :param c: Exploration constant in the UCT formula.
        """
        self.c = c

    def search(self, root: Node, n_simulations: int = 1000) -> Tuple[int, int, int, int, List[Tuple[int,int]]]:
        """
        Run MCTS starting from 'root' for 'n_simulations' iterations.
        Return the move (edge) leading to the most visited child.
        """
        for _ in range(n_simulations):
            node = self._select(root)
            if not node.state.is_game_over():
                node = self._expand(node)
            result = self._simulate(node)
            self._backpropagate(node, result)

        # After simulations, pick the child with the highest visit_count
        best_move, best_child = self._best_child(root, consider_exploration=False)
        return best_move

    def _select(self, node: Node) -> Node:
        """
        Selection phase: descend the tree by choosing child nodes with the highest UCT,
        until reaching a leaf node.
        """
        while not node.is_leaf() and not node.state.is_game_over():
            move, node = self._best_child(node, consider_exploration=True)
        return node

    def _expand(self, node: Node) -> Node:
        """
        Expansion phase: take one of the legal moves from this leaf node, 
        create a child node for it, and return that child.
        """
        moves = node.state.get_legal_moves()
        if not moves:
            # No legal moves—this is a terminal position
            return node

        # For demonstration, just pick one move to expand (typical MCTS expands one random child).
        move = random.choice(moves)
        # Create a cloned state
        new_state = node.state.clone()
        new_state.make_move(move)

        child_node = Node(state=new_state, parent=node)
        # Optionally, if you had a policy prior from a neural net, you'd store it here:
        # child_node.policy_prior = some_policy_value
        hashable_move = (move[0], move[1], move[2], move[3], tuple(move[4]))
        node.children[hashable_move] = child_node
        return child_node

    def _simulate(self, node: Node) -> float:
        """
        Simulation (rollout) phase: from 'node', play until the game ends using random moves.
        Return +1 for a black win, -1 for a red win, 0 for a draw.
        
        In a zero-sum setting, we often interpret +1 from the perspective of the
        player-to-move at the original node. Alternatively, you can store the exact winner
        and handle perspective flips during backpropagation.
        """
        temp_state = node.state.clone()

        # Play out randomly
        while not temp_state.is_game_over():
            moves = temp_state.get_legal_moves()
            if not moves:
                break
            move = random.choice(moves)
            temp_state.make_move(move)

        winner = temp_state.get_winner()
        # winner = +1 (Black), -1 (Red), 0 (Draw)
        return float(winner) if winner is not None else 0.0

    def _backpropagate(self, node: Node, result: float):
        """
        Backpropagation: propagate the rollout result up the tree.
        
        If you consider +1 as a win for the node.state's current player,
        you might need to flip signs as you move up the tree. Another common approach:
        store the perspective of the 'root player' and keep consistent with that.
        
        In simple form, we can just add the result or flip it each step for two-player zero-sum.
        """
        # We'll assume 'result' is from the viewpoint of "Black = +1, Red = -1"
        # and that node.state.current_player is either BLACK or RED.
        current_node = node
        while current_node is not None:
            current_node.visit_count += 1
            # We might flip sign based on whose turn it was in current_node. 
            # For simplicity, let's assume 'result' is from the perspective of the root player,
            # so we do not flip in each step. 
            current_node.total_value += result
            current_node = current_node.parent

    def _best_child(self, node: Node, consider_exploration: bool) -> Tuple[
                    Tuple[int, int, int, int, List[Tuple[int,int]]], Node]:
        """
        Given a node, return (move, child_node) with the highest UCT score if consider_exploration = True,
        or highest visit_count if consider_exploration = False.
        """
        best_move = None
        best_node = None
        best_score = float('-inf')

        # For parent's total visits in UCT, we use node.visit_count.
        parent_visits = node.visit_count if node.visit_count > 0 else 1

        for move, child in node.children.items():
            if consider_exploration:
                # UCT formula
                if child.visit_count == 0:
                    # If unvisited, give it an effectively infinite score to ensure it gets explored
                    score = float('inf')
                else:
                    avg_value = child.total_value / child.visit_count
                    # Add policy_prior factor if using a PUCT variant; for standard UCT, ignore it
                    # policy_term = self.c * child.policy_prior * ...
                    
                    # Standard UCT
                    exploration_term = self.c * math.sqrt(math.log(parent_visits) / child.visit_count)
                    score = avg_value + exploration_term
            else:
                # For final move selection, pick highest-visit child
                score = child.visit_count

            if score > best_score:
                best_score = score
                best_move = move
                best_node = child

        return best_move, best_node