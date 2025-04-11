from flask import Flask, request, jsonify
from flask import send_from_directory
from engine_api import (
    init_game,
    get_board_state,
    make_user_move,
    make_user_move_path,
    make_nn_move,
    get_game_status,
    get_legal_moves
)

app = Flask(__name__)

@app.route("/")
def index():
    return send_from_directory("templates", "index.html")

@app.route("/start", methods=["POST"])
def start_game():
    init_game()
    return jsonify({"message": "New game started", "state": get_board_state()})

@app.route("/state", methods=["GET"])
def state():
    state = get_board_state()
    if state is None:
        return jsonify({"error": "No active game"}), 400
    return jsonify(state)

@app.route("/legal", methods=["GET"])
def legal_moves():
    moves = get_legal_moves()
    if moves is None:
        return jsonify({"error": "No game in progress"}), 400
    return jsonify({"legal_moves": moves})

@app.route("/move", methods=["POST"])
def move():
    data = request.get_json()
    path = data.get("path", None)

    if path is None or not isinstance(path, list) or len(path) < 2:
        return jsonify({"success": False, "error": "Invalid move path"}), 400

    result = make_user_move_path(path)  # You'll define this
    if not result["success"]:
        return jsonify(result), 400

    return jsonify({"success": True, "state": get_board_state()})



@app.route("/nn-move", methods=["POST"])
def nn_move():
    result = make_nn_move()
    if not result["success"]:
        return jsonify(result), 400
    return jsonify({"success": True, "nn_move": result["nn_move"], "state": get_board_state()})


@app.route("/status", methods=["GET"])
def status():
    return jsonify(get_game_status())


if __name__ == "__main__":
    app.run(debug=True)
