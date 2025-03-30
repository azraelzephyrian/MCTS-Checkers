# train.py

import math
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from typing import List, Tuple
from game import CheckersGame
from network_v1 import CheckersNet, encode_board, move_to_index, index_to_move
from MCTS_NN import Node, MCTS_NN

def self_play_game(net, mcts_simulations=30, temperature=1.0) -> List[Tuple[torch.Tensor, List[float], float]]:
    """
    Play a single game of checkers in self-play mode using MCTS guided by 'net'.
    Return a list of training samples (board_tensor, pi, z).
      - board_tensor: shape (4,8,8) representing the state
      - pi: MCTS visit distribution over all possible moves (8^4 = 4096)
      - z: final outcome from the perspective of the player who moved at that state
    :param net: a CheckersNet instance
    :param mcts_simulations: how many MCTS simulations per move
    :param temperature: controls whether we sample from the MCTS distribution (1.0) or pick argmax (0.0)
    """
    game_data = []
    game = CheckersGame()
    root_node = Node(state=game)
    mcts_solver = MCTS_NN(net=net, c_puct=1.0, action_size=8**4)

    while not game.is_game_over():
        # 1) Run MCTS from the current root node
        for _ in range(mcts_simulations):
            leaf = mcts_solver._select(root_node)
            if not leaf.state.is_game_over():
                mcts_solver._expand(leaf)
            # The leaf's value_est is used for backprop.
            mcts_solver._backpropagate(leaf, leaf.value_est)

        # 2) Build the MCTS visit distribution (pi)
        visit_counts = [0]*(8**4)
        for move_key, child in root_node.children.items():
            # move_key = (r1, c1, r2, c2, tuple_of_captures)
            # convert back to a full 5-tuple with captures as list
            move_tuple = (move_key[0], move_key[1], move_key[2], move_key[3], list(move_key[4]))
            idx = move_to_index(move_tuple)
            visit_counts[idx] = child.visit_count

        total_visits = sum(visit_counts)
        if total_visits > 0:
            pi = [vc / total_visits for vc in visit_counts]
        else:
            # terminal or no children
            pi = [0]*(8**4)

        # 3) Record (board_tensor, pi, current_player)
        board_tensor = encode_board(game)
        current_player = game.current_player  # +1 for black, -1 for red
        game_data.append((board_tensor, pi, current_player))

        # 4) Select a move from pi
        if temperature > 0.0:
            # sample from pi
            move_index = random.choices(range(8**4), weights=pi, k=1)[0]
        else:
            # pick argmax
            move_index = max(range(8**4), key=lambda i: pi[i])

        # check if no valid moves
        if pi[move_index] == 0:
            break

        # 5) Convert index to a move
        (r1, c1, r2, c2, captures) = index_to_move(move_index)

        # Make the move in the real game
        game.make_move((r1, c1, r2, c2, captures))

        # Move root to the child
        move_key = (r1, c1, r2, c2, tuple(captures))
        if move_key in root_node.children:
            root_node = root_node.children[move_key]
            root_node.parent = None  # new root
        else:
            # forced to create a new root if something didn't match
            root_node = Node(state=game)

    # final outcome
    winner = game.get_winner()  # +1 for black, -1 for red, 0 for draw

    # 6) Assign outcomes from each player's perspective
    final_samples = []
    for (board_tensor, pi, player) in game_data:
        # if player is black, outcome = winner; if red, outcome = -winner
        if player in (1,2):  # black side
            z = float(winner)
        else:
            z = float(-winner)
        final_samples.append((board_tensor, pi, z))

    return final_samples


def alpha_zero_train_step(net, optimizer, batch, reg_const=1e-4):
    """
    Perform one training step with the standard AlphaZero loss:
        L = (z - v)^2  -  π^T log p  +  c ||θ||^2
    We'll rely on 'weight_decay=reg_const' in optimizer for L2 reg, so we won't manually add that.
    
    :param net: CheckersNet
    :param optimizer: torch.optim.Optimizer with weight_decay = reg_const
    :param batch: list of (board_tensor, pi, z) samples
    """
    # Prepare data
    board_batch = torch.stack([item[0] for item in batch])   # shape [B, 4, 8, 8]
    pi_batch = torch.tensor([item[1] for item in batch], dtype=torch.float32)  # shape [B, action_size]
    z_batch = torch.tensor([item[2] for item in batch], dtype=torch.float32)   # shape [B]

    device = next(net.parameters()).device
    board_batch = board_batch.to(device)
    pi_batch = pi_batch.to(device)
    z_batch = z_batch.to(device)

    # Forward
    policy_logits, value_pred = net(board_batch)  # shapes [B, action_size], [B,1]
    value_pred = value_pred.view(-1)  # [B]

    # Convert logits -> log probs
    log_policy = F.log_softmax(policy_logits, dim=1)  # [B, action_size]

    # Value loss = MSE( z, v )
    value_loss = F.mse_loss(value_pred, z_batch)

    # Policy loss = - pi^T * log_policy => cross-entropy
    policy_loss = -(pi_batch * log_policy).sum(dim=1).mean()

    # Combined loss
    loss = 1.5*value_loss + policy_loss

    # Backprop
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item(), policy_loss.item(), value_loss.item()


def train(net, optimizer, replay_data, batch_size=32):
    """
    Shuffle 'replay_data' and do mini-batch SGD updates using alpha_zero_train_step.
    Returns average losses for logging.
    """
    random.shuffle(replay_data)
    batches = [replay_data[i:i+batch_size] for i in range(0, len(replay_data), batch_size)]

    net.train()
    total_loss, total_policy_loss, total_value_loss = 0, 0, 0
    for batch in batches:
        loss, p_loss, v_loss = alpha_zero_train_step(net, optimizer, batch)
        total_loss += loss
        total_policy_loss += p_loss
        total_value_loss += v_loss

    n = len(batches)
    return (total_loss/n, total_policy_loss/n, total_value_loss/n)

from collections import deque
import signal
import threading
import time
import torch
import torch.optim as optim
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

def main():
    # Signal handler and user input to trigger stop
    stop_signal = {"stop": False}  # Use a mutable object to modify in signal handler

    def handle_signal(signum, frame):
        print("\nReceived stop signal. Saving progress...")
        stop_signal["stop"] = True

    def wait_for_stop():
        while not stop_signal["stop"]:
            user_input = input("Type 'stop' to save and exit: ").strip().lower()
            if user_input == "stop":
                stop_signal["stop"] = True

    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, handle_signal)

    # Start the input listener in a separate thread
    input_thread = threading.Thread(target=wait_for_stop, daemon=True)
    input_thread.start()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_size = 8**4

    net = CheckersNet(action_size=action_size)
    net.to(device)

    try:
        net.load_state_dict(torch.load("/Users/alexi/OneDrive/Desktop/my_code_projects/MCTS/outputs/fast_latest_iteration.pth", map_location=device))
        print("Loaded model from /Users/alexi/OneDrive/Desktop/my_code_projects/MCTS/outputs/fast_latest_iteration.pth")
    except FileNotFoundError:
        print("No checkpoint found. Starting from scratch.")

    optimizer = optim.Adam([
    {"params": net.policy_fc.parameters(), "lr": 1e-3, "weight_decay": 1e-4},
    {"params": net.value_fc1.parameters(), "lr": 1e-4, "weight_decay": 1e-4},
    {"params": net.value_fc2.parameters(), "lr": 1e-4, "weight_decay": 1e-4},
    ])

    # Set a fixed size for the replay buffer
    REPLAY_BUFFER_MAX_SIZE = 100000
    try:
        replay_buffer = torch.load("/Users/alexi/OneDrive/Desktop/my_code_projects/MCTS/outputs/fast_replay_buffer_latest.pth")
        print("Loaded replay buffer from /Users/alexi/OneDrive/Desktop/my_code_projects/MCTS/outputs/fast_replay_buffer_latest.pth")
        replay_buffer = deque(replay_buffer, maxlen=REPLAY_BUFFER_MAX_SIZE)  # Convert to deque
    except FileNotFoundError:
        print("No replay buffer found. Starting fresh.")
        replay_buffer = deque(maxlen=REPLAY_BUFFER_MAX_SIZE)

    num_iterations = 300
    games_per_iteration = 5
    mcts_simulations = 30
    iteration_start = 0  # Default starting iteration

    # Check if CSV file exists and determine where to start
    csv_file = "/Users/alexi/OneDrive/Desktop/my_code_projects/MCTS/outputs/fast_training_metrics.csv"
    existing_iterations = 0
    if os.path.exists(csv_file):
        with open(csv_file, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            rows = list(reader)
            if rows:
                existing_iterations = int(rows[-1][0])  # Get the last logged iteration
        print(f"Resuming training from iteration {existing_iterations + 1}.")
        iteration_start = existing_iterations
    else:
        # Initialize CSV file for recording metrics
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Iteration", "Loss", "Policy Loss", "Value Loss"])  # Header row

    iteration_times = []  # Store times for completed iterations
    iteration_start = existing_iterations  # Default starting iteration
    iter_idx = iteration_start - 1  # Initialize iter_idx to a safe value for early stop

    for iter_idx in range(iteration_start, num_iterations):
        start_time = time.time()

        if stop_signal["stop"]:
            # Save current progress with specific filenames when interrupted
            torch.save(net.state_dict(), f"/Users/alexi/OneDrive/Desktop/my_code_projects/MCTS/outputs/fast_model_iter_{iter_idx}.pth")
            torch.save(list(replay_buffer), f"/Users/alexi/OneDrive/Desktop/my_code_projects/MCTS/outputs/fast_replay_buffer_iter_{iter_idx}.pth")  # Convert deque to list for saving
            print(f"Training stopped. Saved model as model_iter_{iter_idx}.pth and replay buffer as replay_buffer_iter_{iter_idx}.pth")
            break

        iteration_data = []
        for _ in range(games_per_iteration):
            game_data = self_play_game(
                net, 
                mcts_simulations=mcts_simulations,
                temperature=1.0
            )
            iteration_data.extend(game_data)

        replay_buffer.extend(iteration_data)  # Deque manages the size automatically
        avg_loss, avg_ploss, avg_vloss = train(net, optimizer, list(replay_buffer), batch_size=32)
        print(f"Iter {iter_idx + 1}/{num_iterations}: loss={avg_loss:.4f}, policy_loss={avg_ploss:.4f}, value_loss={avg_vloss:.4f}")

        # Append metrics to CSV file
        try:
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([iter_idx + 1, avg_loss, avg_ploss, avg_vloss])
            print(f"Metrics saved for iteration {iter_idx + 1}: loss={avg_loss:.4f}, policy_loss={avg_ploss:.4f}, value_loss={avg_vloss:.4f}")
        except Exception as e:
            print(f"Error writing metrics for iteration {iter_idx + 1}: {e}")

        # Save the most recent state as "/Users/alexi/OneDrive/Desktop/my_code_projects/MCTS/outputs/fast_v3_latest_iteration"
        torch.save(net.state_dict(), "/Users/alexi/OneDrive/Desktop/my_code_projects/MCTS/outputs/fast_latest_iteration.pth")
        torch.save(list(replay_buffer), "/Users/alexi/OneDrive/Desktop/my_code_projects/MCTS/outputs/fast_replay_buffer_latest.pth")  # Convert deque to list for saving
        print(f"Checkpoint saved as /Users/alexi/OneDrive/Desktop/my_code_projects/MCTS/outputs/fast_latest_iteration.pth and /Users/alexi/OneDrive/Desktop/my_code_projects/MCTS/outputs/fast_v3_replay_buffer_latest.pth")

        # Measure iteration time
        iteration_time = time.time() - start_time
        iteration_times.append((iter_idx + 1, iteration_time))  # Store iteration and time
        print(f"Iteration {iter_idx + 1} took {iteration_time:.2f} seconds.")

        # Project future times using linear regression
        if len(iteration_times) > 1:
            # Prepare data for regression
            X = np.array([x[0] for x in iteration_times]).reshape(-1, 1)  # Iteration indices
            y = np.array([x[1] for x in iteration_times])  # Times

            # Fit regression model
            reg = LinearRegression()
            reg.fit(X, y)

            # Predict remaining iterations
            projected_time_per_iteration = reg.predict(np.array([[iter_idx + 1]])).item()
            remaining_iterations = num_iterations - (iter_idx + 1)
            expected_completion_time = projected_time_per_iteration * remaining_iterations

            # Print projection
            print(f"Projected time per iteration: {projected_time_per_iteration:.2f} seconds")
            print(f"Projected time to completion: {expected_completion_time / 60:.2f} minutes")

    if not stop_signal["stop"]:
        # Final save if training completes without interruption
        torch.save(net.state_dict(), f"model_iter_{iter_idx}.pth")
        torch.save(list(replay_buffer), f"replay_buffer_iter_{iter_idx}.pth")
        print("Training complete. Final model saved to checkers_net.pth")

if __name__ == "__main__":
    main()