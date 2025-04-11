# train.py

import math
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from typing import List, Tuple
from game import CheckersGame, MoveEncoder
from network_v3 import CheckersNet, encode_board, move_to_index, index_to_move
from MCTS_NN import Node, MCTS_NN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMPTY = 0
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







def self_play_game(net, encoder, mcts_simulations=30, temperature=1.0) -> List[Tuple[torch.Tensor, List[float], float]]:
    game_data = []
    game = CheckersGame()
    root_node = Node(state=game)
    mcts_solver = MCTS_NN(net=net, c_puct=1.0, encoder = encoder)  # match encoder size

    turn = 0
    while not game.is_game_over():
        # 1) Run MCTS
        for _ in range(mcts_simulations):
            leaf = mcts_solver._select(root_node, encoder)
            if not leaf.state.is_game_over():
                mcts_solver._expand(leaf, encoder)
            mcts_solver._backpropagate(leaf, leaf.value_est)

        # 2) Build the visit distribution pi over all known full-path actions
        pi = [0.0] * len(encoder)
        for action_id, child in root_node.children.items():
            if 0 <= action_id < len(pi):
                pi[action_id] = child.visit_count

        total_visits = sum(pi)
        if total_visits > 0:
            pi = [v / total_visits for v in pi]
        else:
            break  # no moves → resign?

        # 3) Record sample
        board_tensor = encode_board(game)
        current_player = game.current_player
        game_data.append((board_tensor, pi, current_player))

        # 4) Sample a move index from pi
        if temperature > 0.0:
            move_id = random.choices(range(len(pi)), weights=pi, k=1)[0]
        else:
            move_id = max(range(len(pi)), key=lambda i: pi[i])

        if pi[move_id] == 0:
            break  # fallback if illegal

        # 5) Decode the move path and apply it
        path = encoder.decode(move_id)
        move_applied = apply_move_path(game, path)  # you'll define this

        if not move_applied:
            break  # invalid move (shouldn't happen)

        # 6) Move root
        if move_id in root_node.children:
            root_node = root_node.children[move_id]
            root_node.parent = None
        else:
            root_node = Node(state=game)

        turn += 1

    # 7) Finish game
    winner = game.get_winner()
    if winner is None:
        print("[Warning] Game ended without a winner — treating as draw.")
        winner = 0.0
    final_samples = []
    for (board_tensor, pi, player) in game_data:
        z = float(winner) if player in (1, 2) else float(-winner)
        final_samples.append((board_tensor, pi, z))

    print("Self-play game finished.")
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

    policy_loss = -(pi_batch * log_policy).sum(dim=1).mean()
    value_loss = F.mse_loss(value_pred, z_batch)

    # ✅ Add entropy: -sum(pi * log_pi)
    entropy = -(pi_batch * log_policy).sum(dim=1).mean().item()

    # ✅ Compute true MSE (same as value_loss, but separated for clarity/logging)
    value_mse = F.mse_loss(value_pred, z_batch).item()

    loss = value_loss + 1.5 * policy_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item(), policy_loss.item(), value_loss.item(), entropy, value_mse

def pad_policy(pi: List[float], target_size: int) -> List[float]:
    """
    Pads or truncates a policy vector to match the target action size.
    """
    if len(pi) > target_size:
        return pi[:target_size]
    elif len(pi) < target_size:
        return pi + [0.0] * (target_size - len(pi))
    return pi


def train(net, optimizer, replay_data, batch_size=32):
    """
    Shuffle 'replay_data' and do mini-batch SGD updates using alpha_zero_train_step.
    Returns average loss, policy loss, value loss, entropy, and value MSE for logging.
    """
    random.shuffle(replay_data)
    batches = [replay_data[i:i + batch_size] for i in range(0, len(replay_data), batch_size)]

    net.train()
    total_loss, total_policy_loss, total_value_loss = 0, 0, 0
    total_entropy, total_value_mse = 0, 0
    current_action_size = net.policy_fc.out_features

    for batch in batches:
        # Pad pi if needed
        padded_batch = []
        for board_tensor, pi, z in batch:
            if len(pi) != current_action_size:
                pi = pad_policy(pi, current_action_size)
            padded_batch.append((board_tensor, pi, z))
        assert len(pi) == current_action_size, f"[ERROR] Mismatched pi length {len(pi)} (expected {current_action_size})"


        loss, p_loss, v_loss, entropy, value_mse = alpha_zero_train_step(net, optimizer, padded_batch)
        total_loss += loss
        total_policy_loss += p_loss
        total_value_loss += v_loss
        total_entropy += entropy
        total_value_mse += value_mse

    n = len(batches)
    return (
        total_loss / n,
        total_policy_loss / n,
        total_value_loss / n,
        total_entropy / n,
        total_value_mse / n
    )

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
import pickle

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # 🔄 Initialize or load encoder
    encoder_path = "/Users/alexi/OneDrive/Desktop/my_code_projects/MCTS-Checkers/outputs/move_encoder.pkl"
    try:
        with open(encoder_path, "rb") as f:
            encoder = pickle.load(f)
        print("Loaded MoveEncoder.")
    except FileNotFoundError:
        encoder = MoveEncoder()
        print("No encoder found. Starting fresh.")

    # 🔄 Initialize network using current action_size
    action_size = len(encoder) if len(encoder) > 0 else 4096  # fallback if empty
    net = CheckersNet(action_size=action_size)
    net.to(device)
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
    #action_size = 8**4
    action_size = len(encoder)
    net = CheckersNet(action_size=action_size)
    net.to(device)

    try:
        net.load_state_dict(torch.load("/Users/alexi/OneDrive/Desktop/my_code_projects/MCTS-Checkers/outputs/fast_latest_iteration.pth", map_location=device))
        print("Loaded model from /Users/alexi/OneDrive/Desktop/my_code_projects/MCTS-Checkers/outputs/fast_latest_iteration.pth")
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
        replay_buffer = torch.load("/Users/alexi/OneDrive/Desktop/my_code_projects/MCTS-Checkers/outputs/fast_replay_buffer_latest.pth")
        print("Loaded replay buffer from /Users/alexi/OneDrive/Desktop/my_code_projects/MCTS-Checkers/outputs/fast_replay_buffer_latest.pth")
        replay_buffer = deque(replay_buffer, maxlen=REPLAY_BUFFER_MAX_SIZE)  # Convert to deque
    except FileNotFoundError:
        print("No replay buffer found. Starting fresh.")
        replay_buffer = deque(maxlen=REPLAY_BUFFER_MAX_SIZE)

    num_iterations = 2000
    games_per_iteration = 5
    mcts_simulations = 30
    iteration_start = 0  # Default starting iteration

    # Check if CSV file exists and determine where to start
    csv_file = "/Users/alexi/OneDrive/Desktop/my_code_projects/MCTS-Checkers/outputs/fast_training_metrics.csv"
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
            writer.writerow(["Iteration", "Loss", "Policy Loss", "Value Loss", "Entropy", "Value MSE"])  # Header row

    iteration_times = []  # Store times for completed iterations
    iteration_start = existing_iterations  # Default starting iteration
    iter_idx = iteration_start - 1  # Initialize iter_idx to a safe value for early stop

    for iter_idx in range(iteration_start, num_iterations):

        start_time = time.time()

        if stop_signal["stop"]:
            # Save current progress with specific filenames when interrupted
            torch.save(net.state_dict(), f"/Users/alexi/OneDrive/Desktop/my_code_projects/MCTS-Checkers/outputs/fast_model_iter_{iter_idx}.pth")
            torch.save(list(replay_buffer), f"/Users/alexi/OneDrive/Desktop/my_code_projects/MCTS-Checkers/outputs/fast_replay_buffer_iter_{iter_idx}.pth")  # Convert deque to list for saving
            print(f"Training stopped. Saved model as model_iter_{iter_idx}.pth and replay buffer as replay_buffer_iter_{iter_idx}.pth")
            break

        iteration_data = []

        for _ in range(games_per_iteration):
            # BEFORE generating self-play data:
            new_action_size = len(encoder)
            if new_action_size != net.policy_fc.out_features:
                print(f"[Info] Reinitializing network for action_size={new_action_size}")
                net = CheckersNet(action_size=new_action_size).to(device)

            # BEFORE generating self-play data:
            game_data = self_play_game(
                net, 
                encoder,  # ✅ pass encoder
                mcts_simulations=mcts_simulations,
                temperature=1.0
            )
            iteration_data.extend(game_data)

        replay_buffer.extend(iteration_data)

        # 🔄 Update network action size if encoder grew
        avg_loss, avg_ploss, avg_vloss, avg_entropy, avg_value_mse = train(net, optimizer, list(replay_buffer), batch_size=32)
        print(f"Iter {iter_idx + 1}/{num_iterations}: "
                f"loss={avg_loss:.4f}, "
                f"policy_loss={avg_ploss:.4f}, "
                f"value_loss={avg_vloss:.4f}, "
                f"entropy={avg_entropy:.4f}, "
                f"value_mse={avg_value_mse:.4f}")


        # Append metrics to CSV file
        try:
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([iter_idx + 1, avg_loss, avg_ploss, avg_vloss, avg_entropy, avg_value_mse])
            print(f"Metrics saved for iteration {iter_idx + 1}: loss={avg_loss:.4f}, policy_loss={avg_ploss:.4f}, value_loss={avg_vloss:.4f}")
        except Exception as e:
            print(f"Error writing metrics for iteration {iter_idx + 1}: {e}")

        with open(encoder_path, "wb") as f:
            pickle.dump(encoder, f)
        print(f"Saved encoder to {encoder_path}")

        # Save the most recent state as "/Users/alexi/OneDrive/Desktop/my_code_projects/MCTS-Checkers/outputs/fast_v3_latest_iteration"
        torch.save(net.state_dict(), "/Users/alexi/OneDrive/Desktop/my_code_projects/MCTS-Checkers/outputs/fast_latest_iteration.pth")
        torch.save(list(replay_buffer), "/Users/alexi/OneDrive/Desktop/my_code_projects/MCTS-Checkers/outputs/fast_replay_buffer_latest.pth")  # Convert deque to list for saving
        print(f"Checkpoint saved as /Users/alexi/OneDrive/Desktop/my_code_projects/MCTS-Checkers/outputs/fast_latest_iteration.pth and /Users/alexi/OneDrive/Desktop/my_code_projects/MCTS-Checkers/outputs/fast_v3_replay_buffer_latest.pth")

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