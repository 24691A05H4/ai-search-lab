"""
game.py - Tic-Tac-Toe Game Engine
Defines the core game state, rules, board representation, and win detection.
"""

from typing import List, Optional, Tuple


class TicTacToeGame:
    """
    Encapsulates the state and logic for a standard 3x3 Tic-Tac-Toe game.
    Players:
      'X' : Human player (first mover)
      'O' : AI computer player
      ' ' : Empty cell
    """

    # Winning combinations (3 rows, 3 columns, 2 diagonals)
    WINNING_COMBINATIONS = [
        (0, 1, 2),  # Row 1
        (3, 4, 5),  # Row 2
        (6, 7, 8),  # Row 3
        (0, 3, 6),  # Column 1
        (1, 4, 7),  # Column 2
        (2, 5, 8),  # Column 3
        (0, 4, 8),  # Main diagonal
        (2, 4, 6),  # Anti diagonal
    ]

    def __init__(self, board: Optional[List[str]] = None, current_turn: str = "X"):
        """Initialize game with a clean board or copy an existing board."""
        if board is not None:
            self.board = list(board)
        else:
            self.board = [" "] * 9
        self.current_turn = current_turn

    def reset(self):
        """Reset board to initial empty state."""
        self.board = [" "] * 9
        self.current_turn = "X"

    def is_valid_move(self, index: int) -> bool:
        """Check if index is within bounds and cell is unoccupied."""
        return 0 <= index < 9 and self.board[index] == " "

    def make_move(self, index: int, player: str) -> bool:
        """
        Execute a move on the board for the specified player.
        Returns True if successful, False if invalid.
        """
        if self.is_valid_move(index):
            self.board[index] = player
            self.current_turn = "O" if player == "X" else "X"
            return True
        return False

    def undo_move(self, index: int, previous_turn: Optional[str] = None):
        """Undo a move at index and restore turn."""
        if 0 <= index < 9:
            self.board[index] = " "
            if previous_turn is not None:
                self.current_turn = previous_turn
            else:
                self.current_turn = "X" if self.current_turn == "O" else "O"

    def get_available_moves(self) -> List[int]:
        """Return indices of all available empty cells."""
        return [i for i, cell in enumerate(self.board) if cell == " "]

    def check_winner(self) -> Optional[str]:
        """
        Check board for game outcome.
        Returns:
            'X'    : Human wins
            'O'    : AI wins
            'Draw' : Game ended in a tie
            None   : Game still in progress
        """
        for combo in self.WINNING_COMBINATIONS:
            a, b, c = combo
            if self.board[a] != " " and self.board[a] == self.board[b] == self.board[c]:
                return self.board[a]

        if " " not in self.board:
            return "Draw"

        return None

    def get_winning_line(self) -> Optional[Tuple[int, int, int]]:
        """Return winning cell indices if there is a winner, else None."""
        for combo in self.WINNING_COMBINATIONS:
            a, b, c = combo
            if self.board[a] != " " and self.board[a] == self.board[b] == self.board[c]:
                return combo
        return None

    def is_game_over(self) -> bool:
        """Check whether the game has concluded (win or draw)."""
        return self.check_winner() is not None

    def clone(self) -> "TicTacToeGame":
        """Return an independent duplicate of the game state."""
        return TicTacToeGame(board=self.board, current_turn=self.current_turn)

    def to_matrix(self) -> List[List[str]]:
        """Return 3x3 2D list representation of the board."""
        return [self.board[i : i + 3] for i in range(0, 9, 3)]

    def count_pieces(self) -> Tuple[int, int]:
        """Return count of ('X', 'O') on the board."""
        x_count = self.board.count("X")
        o_count = self.board.count("O")
        return x_count, o_count

