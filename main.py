from game import CheckersGame
from MCTS import MCTS, Node

def main():
    # Create an initial game state
    game = CheckersGame()

    # Create the root node
    root = Node(state=game)

    # Create an MCTS instance
    mcts_solver = MCTS(c=1.4)

    # Run MCTS for some simulations
    best_move = mcts_solver.search(root, n_simulations=1000)
    print("MCTS suggests move:", best_move)

    # You could then apply the move to the game:
    game.make_move(best_move)
    game.print_board()

if __name__ == "__main__":
    main()