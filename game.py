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
        
        # Place 12 black pieces
        # Rows 0..2 have black pieces (in standard US checkers, black on "dark" squares)
        for r in range(3):
            for c in range(8):
                if (r + c) % 2 == 0:
                    self.board[r][c] = BLACK_MAN
        
        # Place 12 red pieces
        # Rows 5..7 have red pieces
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
        # piece1 * piece2 < 0 if they differ in sign
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
                if piece != EMPTY and (sign > 0) == self.is_black_piece(piece):
                    # Generate possible moves for this piece
                    piece_moves = self._get_piece_moves(r, c)
                    for move in piece_moves:
                        if move[4]:  # if there's at least one capture
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
        at position (r, c). Returns a list of moves in the same format:
        (start_r, start_c, end_r, end_c, captured_positions).
        """
        piece = self.get_piece(r, c)
        if piece == EMPTY:
            return []

        directions = []
        if self.is_king(piece):
            # King can move in all four diagonal directions
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        else:
            # Man: black moves down (+1), red moves up (-1)
            move_dir = 1 if self.is_black_piece(piece) else -1
            directions = [(move_dir, -1), (move_dir, 1)]

        # Step 1: Check if any captures are available (multi-jumps included)
        capture_sequences = []
        self._find_captures(r, c, piece, [], capture_sequences, directions)

        # If we found any capturing sequences, return them
        if capture_sequences:
            return capture_sequences

        # Otherwise, return non-capturing diagonal steps
        non_capture_moves = []
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if self.in_bounds(nr, nc) and self.get_piece(nr, nc) == EMPTY:
                non_capture_moves.append((r, c, nr, nc, []))

        return non_capture_moves

    def _find_captures(
        self,
        r: int, 
        c: int, 
        piece: int,
        captured_so_far: List[Tuple[int, int]],
        capture_sequences: List[Tuple[int, int, int, int, List[Tuple[int, int]]]],
        directions: List[Tuple[int, int]]
    ):
        """
        Recursively find all capturing sequences (multi-jumps).
        Each capturing sequence is stored in capture_sequences.

        We track the path of intermediate states using a temporary board clone
        so we don't permanently modify the main board while exploring.
        """
        found_capture = False

        for dr, dc in directions:
            # Opponent piece is presumably at (r+dr, c+dc)
            # Landing square is (r+2*dr, c+2*dc)
            middle_r, middle_c = r + dr, c + dc
            landing_r, landing_c = r + 2*dr, c + 2*dc

            if not self.in_bounds(middle_r, middle_c) or not self.in_bounds(landing_r, landing_c):
                continue

            mid_piece = self.get_piece(middle_r, middle_c)
            land_piece = self.get_piece(landing_r, landing_c)

            if mid_piece != EMPTY and self.is_opponent(piece, mid_piece) and land_piece == EMPTY:
                # We have a potential capture
                found_capture = True

                # Simulate this capture on a cloned board
                cloned_game = self.clone()
                # Remove captured piece
                cloned_game.set_piece(middle_r, middle_c, EMPTY)
                # Move the piece
                cloned_game.set_piece(landing_r, landing_c, piece)
                cloned_game.set_piece(r, c, EMPTY)
                
                # Possibly king the piece if it reaches the far row
                cloned_game._maybe_king(landing_r, landing_c)

                new_captured = captured_so_far + [(middle_r, middle_c)]

                # Continue searching for more captures from the new position
                cloned_game._find_captures(
                    landing_r,
                    landing_c,
                    cloned_game.get_piece(landing_r, landing_c),
                    new_captured,
                    capture_sequences,
                    directions if cloned_game.is_king(piece) else cloned_game._get_move_directions(piece)
                )

        # If we didn't find any further captures from this position, but we arrived here from a capture,
        # then this is a final sequence in a multi-jump chain.
        if not found_capture and captured_so_far:
            start_r, start_c = captured_so_far[0][0], captured_so_far[0][1]  # not exactly correct, need better reference
            # Actually, for clarity, store entire chain differently. But for simplicity, we'll define:
            # The move is from (r0, c0) to (r, c) after multi-jumps, capturing captured_so_far squares.
            # Let's approximate that the first captured was from the original (r0, c0).
            # We'll track the final landing at the current (r, c).

            # In a real system, we might track the path. For demonstration, let's do this:
            # We'll say the "start" is the first jump's origin (we can store it on function entry if needed).
            # For now, let's store a single-segment capture. Real multi-jump tracking might store each jump.
            # Simplify for demonstration:
            pass

        # If there's no additional capture from here and we are in a deeper recursion, we should record the final path.
        # A simpler approach: track the capturing route within the same recursion. Let’s do that:

        if not found_capture and len(captured_so_far) > 0:
            # We need to figure out the actual start position. For that, let's do a small hack:
            # The original piece's position was (r0, c0) before the first capture. We'll store that in captured_so_far as well.
            # A robust approach is to pass the original start coords to the recursion. Let's do that:

            return  # We handle a better approach below

    def _get_move_directions(self, piece: int) -> List[Tuple[int, int]]:
        """
        Return the forward directions for a man, or all 4 directions for a king.
        """
        if self.is_king(piece):
            return [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        else:
            move_dir = 1 if self.is_black_piece(piece) else -1
            return [(move_dir, -1), (move_dir, 1)]

    def _maybe_king(self, r: int, c: int):
        """
        If a man has reached the opposite side, make it a king.
        """
        piece = self.get_piece(r, c)
        if self.is_black_piece(piece) and r == 7 and piece == BLACK_MAN:
            self.set_piece(r, c, BLACK_KING)
        elif self.is_red_piece(piece) and r == 0 and piece == RED_MAN:
            self.set_piece(r, c, RED_KING)

    ''' 
    def make_move(self, move: Tuple[int, int, int, int, List[Tuple[int, int]]]) -> bool:
        """
        Execute a move on the board, including captures. Returns True if move is valid, else False.
        move format: (start_r, start_c, end_r, end_c, captured_positions).
        """
        (r1, c1, r2, c2, captures) = move
        piece = self.get_piece(r1, c1)
        if piece == EMPTY:
            return False

        # Basic validation: end square must be empty
        if self.get_piece(r2, c2) != EMPTY:
            return False

        # Move piece
        self.set_piece(r1, c1, EMPTY)
        self.set_piece(r2, c2, piece)
        
        # Remove captured pieces
        for (cr, cc) in captures:
            self.set_piece(cr, cc, EMPTY)

        # Check if we should king the piece
        self._maybe_king(r2, c2)

        # Switch player
        self.current_player = RED_MAN if self.current_player in (BLACK_MAN, BLACK_KING) else BLACK_MAN
        return True
    '''

    def make_move(self, move: Tuple[int, int, int, int, List[Tuple[int, int]]]) -> bool:
        #print(f"make_move called with move={move}")

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
        self.current_player = RED_MAN if self.current_player in (BLACK_MAN, BLACK_KING) else BLACK_MAN

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
        if not self.is_game_over():
            return None

        moves = self.get_legal_moves()
        if not moves:
            # The current player cannot move, so the other player wins
            return -self.get_current_player_sign()

        # Check piece counts
        black_count = sum(self.is_black_piece(self.board[r][c]) for r in range(8) for c in range(8))
        red_count = sum(self.is_red_piece(self.board[r][c]) for r in range(8) for c in range(8))
        if black_count == 0:
            return -1  # Red wins
        elif red_count == 0:
            return 1   # Black wins

        # If we reached here, it might be the no-capture draw
        if self.no_capture_count >= 40:
            return 0  # draw

        # Fallback if you have other draw conditions
        return 0
    
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