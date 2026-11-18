"""
minimax.py - Depth-Limited Minimax Algorithm with Heuristic Evaluation
Implements search tree exploration, cutoff handling, and performance metrics tracking.
"""

import time
from typing import Dict, List, Optional, Tuple
from game import TicTacToeGame


class SearchStatistics:
    """Tracks performance and exploration metrics during game tree search."""

    def __init__(self):
        self.nodes_generated: int = 0
        self.nodes_expanded: int = 0
        self.cutoff_nodes: int = 0
        self.terminal_nodes: int = 0
        self.max_depth_reached: int = 0
        self.start_time: float = 0.0
        self.execution_time_ms: float = 0.0

    def start_timer(self):
        self.start_time = time.perf_counter()

    def stop_timer(self):
        self.execution_time_ms = (time.perf_counter() - self.start_time) * 1000.0

    def reset(self):
        self.nodes_generated = 0
        self.nodes_expanded = 0
        self.cutoff_nodes = 0
        self.terminal_nodes = 0
        self.max_depth_reached = 0
        self.execution_time_ms = 0.0


def evaluate_board_heuristic(game: TicTacToeGame) -> int:
    """
    Heuristic evaluation function used when depth limit is reached at a non-terminal state.
    Positive values favor AI ('O'), negative values favor Human ('X').

    Heuristic Components:
    1. Line Analysis:
       - 2 'O's and 1 empty: +10 (Imminent winning threat for AI)
       - 1 'O' and 2 empty: +1  (Potential winning path for AI)
       - 2 'X's and 1 empty: -10 (Imminent winning threat for Human)
       - 1 'X' and 2 empty: -1  (Potential winning path for Human)
       - Blocked lines (containing both 'X' and 'O'): 0 (No win possible on this line)

    2. Positional Control:
       - Center cell (index 4): +3 for 'O', -3 for 'X' (participates in 4 winning lines)
       - Corner cells (0, 2, 6, 8): +2 for 'O', -2 for 'X' (participates in 3 winning lines each)
       - Edge cells (1, 3, 5, 7): +1 for 'O', -1 for 'X' (participates in 2 winning lines each)
    """
    board = game.board
    score = 0

    # 1. Line Evaluation
    for a, b, c in TicTacToeGame.WINNING_COMBINATIONS:
        line = [board[a], board[b], board[c]]
        o_count = line.count("O")
        x_count = line.count("X")

        if o_count > 0 and x_count > 0:
            continue  # Blocked line

        if o_count == 2 and x_count == 0:
            score += 10
        elif o_count == 1 and x_count == 0:
            score += 1
        elif x_count == 2 and o_count == 0:
            score -= 10
        elif x_count == 1 and o_count == 0:
            score -= 1

    # 2. Positional Evaluation
    # Center
    if board[4] == "O":
        score += 3
    elif board[4] == "X":
        score -= 3

    # Corners: 0, 2, 6, 8
    for corner in [0, 2, 6, 8]:
        if board[corner] == "O":
            score += 2
        elif board[corner] == "X":
            score -= 2

    # Edges: 1, 3, 5, 7
    for edge in [1, 3, 5, 7]:
        if board[edge] == "O":
            score += 1
        elif board[edge] == "X":
            score -= 1

    return score


def minimax(
    game: TicTacToeGame,
    current_depth: int,
    is_maximizing: bool,
    max_depth: int,
    stats: SearchStatistics,
) -> int:
    """
    Recursive Depth-Limited Minimax algorithm.

    Parameters:
        game: Current TicTacToeGame instance
        current_depth: Current depth in the search tree (root is 0)
        is_maximizing: True if AI turn ('O'), False if Human turn ('X')
        max_depth: Depth limit selected by user (1 to 9)
        stats: SearchStatistics instance tracking generated/cutoff/terminal nodes
    """
    stats.nodes_generated += 1
    stats.max_depth_reached = max(stats.max_depth_reached, current_depth)

    winner = game.check_winner()

    # Terminal state check
    if winner is not None:
        stats.terminal_nodes += 1
        if winner == "O":
            return 100 - current_depth  # Reward faster wins
        elif winner == "X":
            return -100 + current_depth  # Penalize faster losses
        else:
            return 0  # Draw

    # Depth cutoff check
    if current_depth >= max_depth:
        stats.cutoff_nodes += 1
        return evaluate_board_heuristic(game)

    available_moves = game.get_available_moves()
    stats.nodes_expanded += 1

    if is_maximizing:
        best_score = float("-inf")
        for move in available_moves:
            game.make_move(move, "O")
            score = minimax(game, current_depth + 1, False, max_depth, stats)
            game.undo_move(move, previous_turn="O")
            best_score = max(best_score, score)
        return int(best_score)
    else:
        best_score = float("inf")
        for move in available_moves:
            game.make_move(move, "X")
            score = minimax(game, current_depth + 1, True, max_depth, stats)
            game.undo_move(move, previous_turn="X")
            best_score = min(best_score, score)
        return int(best_score)


def get_best_ai_move(
    game: TicTacToeGame, max_depth: int = 9
) -> Tuple[Optional[int], Dict]:
    """
    Determines the optimal AI move for 'O' using Depth-Limited Minimax.

    Parameters:
        game: TicTacToeGame instance
        max_depth: Search depth limit (1 to 9)

    Returns:
        (best_move_index, analysis_dict)
    """
    available_moves = game.get_available_moves()
    if not available_moves or game.is_game_over():
        return None, {}

    stats = SearchStatistics()
    stats.start_timer()

    best_score = float("-inf")
    best_move: Optional[int] = None
    move_scores: Dict[int, int] = {}

    # The root node itself is generated and expanded
    stats.nodes_generated += 1
    stats.nodes_expanded += 1

    # Explore all immediate candidate moves
    for move in available_moves:
        game.make_move(move, "O")
        # Call minimax for Human player's turn ('X') at depth 1
        score = minimax(
            game=game,
            current_depth=1,
            is_maximizing=False,
            max_depth=max_depth,
            stats=stats,
        )
        game.undo_move(move, previous_turn="O")

        move_scores[move] = score
        if score > best_score:
            best_score = score
            best_move = move

    stats.stop_timer()

    analysis = {
        "selected_move": best_move,
        "evaluation_score": best_score,
        "max_depth_setting": max_depth,
        "max_depth_reached": stats.max_depth_reached,
        "nodes_generated": stats.nodes_generated,
        "nodes_expanded": stats.nodes_expanded,
        "cutoff_nodes": stats.cutoff_nodes,
        "terminal_nodes": stats.terminal_nodes,
        "execution_time_ms": round(stats.execution_time_ms, 3),
        "move_scores": move_scores,
        "available_moves_count": len(available_moves),
    }

    return best_move, analysis

