from typing import List, Tuple, Optional

EMPTY = 0
BLACK_MAN = 1
BLACK_KING = 2
RED_MAN = -1
RED_KING = -2

class CheckersGame:
    """
    A simple Checkers environment for two players: Black (positive) and Red (negative).
    The board is an 8x8 array, where positive values represent Black pieces and negative
    values represent Red pieces.

    Pieces:
        - BLACK_MAN = 1
        - BLACK_KING = 2
        - RED_MAN   = -1
        - RED_KING  = -2
        - EMPTY     = 0

    Coordinates:
        We'll use (row, col) with row=0 at the top and col=0 at the left.

    Movement:
        Black moves "down" (row increasing) by default, Red moves "up" (row decreasing).
        A King can move in any diagonal direction.
    """

    def __init__(self):
        # Initialize the board in the standard starting position
        self.board = [[EMPTY for _ in range(8)] for _ in range(8)]
        self.current_player = BLACK_MAN  # Black moves first in many versions
        self.no_capture_count = 0

        # Place 12 black pieces on "dark" squares (r + c even) in rows 0..2
        for r in range(3):
            for c in range(8):
                if (r + c) % 2 == 0:
                    self.board[r][c] = BLACK_MAN

        # Place 12 red pieces on "dark" squares (r + c even) in rows 5..7
        for r in range(5, 8):
            for c in range(8):
                if (r + c) % 2 == 0:
                    self.board[r][c] = RED_MAN

    def clone(self) -> 'CheckersGame':
        """
        Return a deep copy of the current game state for experimentation.
        """
        new_game = CheckersGame.__new__(CheckersGame)
        new_game.board = [row[:] for row in self.board]
        new_game.current_player = self.current_player
        new_game.no_capture_count = self.no_capture_count
        return new_game

    def in_bounds(self, r: int, c: int) -> bool:
        """Check if (r, c) is on the 8x8 board."""
        return 0 <= r < 8 and 0 <= c < 8

    def get_piece(self, r: int, c: int) -> int:
        """Return the piece code at board[r][c]."""
        return self.board[r][c]

    def set_piece(self, r: int, c: int, piece: int):
        """Set board[r][c] to the given piece code."""
        self.board[r][c] = piece

    def is_black_piece(self, piece: int) -> bool:
        return piece in (BLACK_MAN, BLACK_KING)

    def is_red_piece(self, piece: int) -> bool:
        return piece in (RED_MAN, RED_KING)

    def is_king(self, piece: int) -> bool:
        return abs(piece) == 2

    def is_opponent(self, piece1: int, piece2: int) -> bool:
        """Check if piece1 and piece2 belong to opposite sides."""
        return piece1 * piece2 < 0

    def get_current_player_sign(self) -> int:
        """
        Returns +1 if it's black's turn, -1 if it's red's turn.
        This helps identify which pieces belong to the current player.
        """
        return 1 if self.current_player in (BLACK_MAN, BLACK_KING) else -1

    def get_legal_moves(self) -> List[Tuple[int, int, int, int, List[Tuple[int, int]]]]:
        """
        Returns a list of legal moves in the form:
            (start_row, start_col, end_row, end_col, captured_positions)

        Where captured_positions is a list of (r, c) squares of opponent pieces captured.

        Moves can be normal single-step moves or capturing jumps (possibly multiple).
        In checkers, capturing is mandatory, so if captures exist, we only return capture moves.
        """
        sign = self.get_current_player_sign()

        all_moves = []
        capture_moves = []

        for r in range(8):
            for c in range(8):
                piece = self.get_piece(r, c)
                # Make sure this piece belongs to the current player
                if piece != EMPTY and ((sign > 0) == self.is_black_piece(piece)):
                    piece_moves = self._get_piece_moves(r, c)
                    for move in piece_moves:
                        if move[4]:  # there is at least one capture
                            capture_moves.append(move)
                        else:
                            all_moves.append(move)

        # If any capture moves are available, they must be taken
        if capture_moves:
            return capture_moves
        else:
            return all_moves

    def _get_piece_moves(self, r: int, c: int) -> List[Tuple[int, int, int, int, List[Tuple[int, int]]]]:
        """
        Generate all possible moves (including multi-jumps) for a single piece
        at position (r, c). Returns a list of moves in the format:
            (start_r, start_c, end_r, end_c, captured_positions).
        """
        piece = self.get_piece(r, c)
        if piece == EMPTY:
            return []

        # 1) First check for any capturing moves
        capture_sequences = []
        capture_directions = self._get_capture_directions(piece)
        self._find_captures(
            r, c, piece,
            captured_so_far=[],
            capture_sequences=capture_sequences,
            directions=capture_directions,
            origin=(r, c)
        )

        if capture_sequences:
            # If we have captures, return those only
            return capture_sequences

        # 2) Otherwise, return possible non-capturing moves (only forward if man, or all 4 if king)
        non_capture_moves = []
        move_directions = self._get_normal_move_directions(piece)
        for dr, dc in move_directions:
            nr, nc = r + dr, c + dc
            # Check in-bounds, same color square, and empty
            if (self.in_bounds(nr, nc) and
                (nr + nc) % 2 == (r + c) % 2 and
                self.get_piece(nr, nc) == EMPTY):
                non_capture_moves.append((r, c, nr, nc, []))

        return non_capture_moves

    def _get_normal_move_directions(self, piece: int) -> List[Tuple[int, int]]:
        """
        Return directions for a normal (non-capturing) move:
            - If King, all 4 diagonals
            - If Man, only forward diagonals
        """
        if self.is_king(piece):
            return [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        else:
            # Man can only move forward
            return [(1, -1), (1, 1)] if self.is_black_piece(piece) else [(-1, -1), (-1, 1)]

    def _get_capture_directions(self, piece: int) -> List[Tuple[int, int]]:
        """
        Return directions used for capturing:
            - If King, all 4 diagonals
            - If Man, all 4 diagonals as well (allow backward capture)
        """
        return [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    def _find_captures(
        self,
        r: int,
        c: int,
        piece: int,
        captured_so_far: List[Tuple[int, int]],
        capture_sequences: List[Tuple[int, int, int, int, List[Tuple[int, int]]]],
        directions: List[Tuple[int, int]],
        origin: Optional[Tuple[int, int]] = None,
    ):
        """
        Recursive helper to find all multi-jump sequences from (r, c).
        Prevents capturing the same square twice. Appends complete capture chains.
        """
        if origin is None:
            origin = (r, c)

        found_any_capture = False

        for dr, dc in directions:
            middle_r, middle_c = r + dr, c + dc
            landing_r, landing_c = r + 2 * dr, c + 2 * dc

            if not self.in_bounds(middle_r, middle_c) or not self.in_bounds(landing_r, landing_c):
                continue

            mid_piece = self.get_piece(middle_r, middle_c)
            land_piece = self.get_piece(landing_r, landing_c)

            # 🚫 Prevent recapturing the same piece
            if (middle_r, middle_c) in captured_so_far:
                continue

            if (
                mid_piece != EMPTY and
                self.is_opponent(piece, mid_piece) and
                land_piece == EMPTY and
                (landing_r + landing_c) % 2 == (r + c) % 2
            ):
                found_any_capture = True

                cloned_game = self.clone()

                # Remove captured piece and move the current piece
                cloned_game.set_piece(middle_r, middle_c, EMPTY)
                cloned_game.set_piece(r, c, EMPTY)
                cloned_game.set_piece(landing_r, landing_c, piece)
                cloned_game._maybe_king(landing_r, landing_c)

                # Update capture path
                new_captures = captured_so_far + [(middle_r, middle_c)]
                next_piece = cloned_game.get_piece(landing_r, landing_c)
                next_dirs = cloned_game._get_capture_directions(next_piece)

                # Recursive call from new position
                cloned_game._find_captures(
                    landing_r,
                    landing_c,
                    next_piece,
                    new_captures,
                    capture_sequences,
                    next_dirs,
                    origin,
                )

        if not found_any_capture and captured_so_far:
            sr, sc = origin
            capture_sequences.append((sr, sc, r, c, captured_so_far))


    def _maybe_king(self, r: int, c: int):
        """
        If a man has reached the opposite side, make it a king.
        """
        piece = self.get_piece(r, c)
        if self.is_black_piece(piece) and r == 7 and piece == BLACK_MAN:
            self.set_piece(r, c, BLACK_KING)
        elif self.is_red_piece(piece) and r == 0 and piece == RED_MAN:
            self.set_piece(r, c, RED_KING)

    def make_move(self, move: Tuple[int, int, int, int, List[Tuple[int, int]]]) -> bool:
        """
        Execute a move on the board, including captures. Returns True if move is valid, else False.
        move format: (start_r, start_c, end_r, end_c, captured_positions).
        """
        (r1, c1, r2, c2, captures) = move
        piece = self.get_piece(r1, c1)

        if piece == EMPTY:
            return False
        if self.get_piece(r2, c2) != EMPTY:
            return False

        # Move piece
        self.set_piece(r1, c1, EMPTY)
        self.set_piece(r2, c2, piece)

        # Remove captured pieces
        if captures:
            for (cr, cc) in captures:
                self.set_piece(cr, cc, EMPTY)
            # If there was a capture, reset counter
            self.no_capture_count = 0
        else:
            # If no capture, increment counter
            self.no_capture_count += 1

        # Possibly king the piece
        self._maybe_king(r2, c2)

        # Switch player
        if self.current_player in (BLACK_MAN, BLACK_KING):
            self.current_player = RED_MAN
        else:
            self.current_player = BLACK_MAN

        return True

    def is_game_over(self) -> bool:
        # 1) No legal moves
        if not self.get_legal_moves():
            return True

        # 2) One side has no pieces
        black_count = sum(self.is_black_piece(self.board[r][c]) for r in range(8) for c in range(8))
        red_count = sum(self.is_red_piece(self.board[r][c]) for r in range(8) for c in range(8))
        if black_count == 0 or red_count == 0:
            return True

        # 3) Too many consecutive non-capturing moves => draw
        if self.no_capture_count >= 40:
            return True

        return False

    def get_winner(self) -> Optional[int]:
        """
        Returns:
            +1 if Black wins,
            -1 if Red wins,
             0 if Draw,
            None if the game is not over.
        """
        if not self.is_game_over():
            return None

        # If no moves available, the current player to move loses => other side wins
        moves = self.get_legal_moves()
        if not moves:
            return -self.get_current_player_sign()

        # Check piece counts
        black_count = sum(self.is_black_piece(self.board[r][c]) for r in range(8) for c in range(8))
        red_count = sum(self.is_red_piece(self.board[r][c]) for r in range(8) for c in range(8))
        if black_count == 0:
            return -1  # Red wins
        elif red_count == 0:
            return 1   # Black wins

        # No-capture draw
        if self.no_capture_count >= 40:
            return 0  # draw

        return 0  # fallback: treat other end-states as a draw if needed

    def print_board(self):
        """
        Display the board in a simple ASCII format.
        """
        print("  " + " ".join(str(c) for c in range(8)))
        for r in range(8):
            row_str = []
            for c in range(8):
                piece = self.board[r][c]
                if piece == BLACK_MAN:
                    row_str.append("b")
                elif piece == BLACK_KING:
                    row_str.append("B")
                elif piece == RED_MAN:
                    row_str.append("r")
                elif piece == RED_KING:
                    row_str.append("R")
                else:
                    row_str.append(".")
            print(f"{r} " + " ".join(row_str))
        print()

    def get_board_state(self):
        """
        Returns the board as a 2D list of integers, where:
        - 0 = empty
        - 1 = black man
        - 2 = black king
        - -1 = red man
        - -2 = red king
        """
        return [row[:] for row in self.board]
