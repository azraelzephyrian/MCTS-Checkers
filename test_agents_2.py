import random
from game import CheckersGame
from MCTS import MCTS
from MCTS_NN import MCTS_NN, Node
from network_v1 import CheckersNet, encode_board
import torch

def play_game(agent1, agent2, verbose=False):
    game = CheckersGame()
    root1 = Node(state=game.clone())  # Clone for NN-Assisted MCTS
    root2 = Node(state=game.clone())  # Clone for Naive MCTS

    current_agent = agent1
    current_root = root1
    other_root = root2

    while not game.is_game_over():
        # Determine which agent is making the move
        agent_name = "NN-Assisted MCTS" if current_agent == agent1 else "Naive MCTS"
        #print(f"Current Agent: {agent_name}")

        # Debug: Check legal moves
        legal_moves = current_root.state.get_legal_moves()
        #print(f"{agent_name} Legal Moves: {legal_moves}")
        if not legal_moves:
            print(f"{agent_name} has no legal moves!")
            break

        # Perform the search to determine the move
        move = current_agent.search(current_root, n_simulations=30)

        if move is None:
            print(f"Error: {agent_name} returned None as the move.")
            print("Current board state:")
            game.print_board()
            print("Legal moves for current player:", current_root.state.get_legal_moves())
            break

        # Apply the move to the main game state
        game.make_move(move)

        # Debug: Verify synchronization between game and current_root
        #print("Main Game Board:")
        #game.print_board()
        #print("Current Root State Board:")
        #current_root.state.print_board()

        # Update roots for both agents independently
        move_key = (move[0], move[1], move[2], move[3], tuple(move[4]))

        # Update current agent's root
        if move_key in current_root.children:
            next_root = current_root.children[move_key]
            next_root.parent = None  # Detach from parent to avoid memory overhead
        else:
            print(f"Warning: Move key {move_key} not found in {agent_name}'s tree. Rebuilding tree.")
            next_root = Node(state=game.clone())  # Rebuild tree from the cloned game state

        # Update the root of the other agent with a fresh clone of the updated game state
        other_root = Node(state=game.clone())  # Reset tree for the other agent

        # Swap roles
        current_root = other_root
        current_agent = agent2 if current_agent == agent1 else agent1

        if verbose:
            game.print_board()

    return game.get_winner()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = CheckersNet(action_size=8**4)
    net.to(device)
    net.load_state_dict(torch.load("C:/Users/alexi/OneDrive/Desktop/my_code_projects/MCTS-Checkers/latest_iteration.pth", map_location=device))
    net.eval()

    naive_mcts = MCTS(c=1.4)
    nn_mcts = MCTS_NN(net=net, c_puct=1.0, action_size=8**4)

    naive_wins = 0
    nn_wins = 0
    draws = 0

    for i in range(100):
        print(f"Game {i + 1}:")
        result = play_game(nn_mcts, naive_mcts, verbose=False)
        
        if result == 1:
            nn_wins += 1
            print("Result: NN-Assisted MCTS wins!")
        elif result == -1:
            naive_wins += 1
            print("Result: Naive MCTS wins!")
        else:
            draws += 1
            print("Result: Draw!")

    print("\nFinal Results:")
    print(f"NN-Assisted MCTS Wins: {nn_wins}")
    print(f"Naive MCTS Wins: {naive_wins}")
    print(f"Draws: {draws}")

if __name__ == "__main__":
    main()