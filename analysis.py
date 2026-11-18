"""
analysis.py - Search Analysis & Benchmarking Module
Provides unified search analysis for both 8-Puzzle and Tic-Tac-Toe.
Aggregates move telemetry, session analytics, multi-depth benchmarks,
and CSV export functionality.
"""

from dataclasses import dataclass, asdict, field
from datetime import datetime
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

from game import TicTacToeGame
from minimax import get_best_ai_move
from puzzle import EightPuzzleGame, GOAL_STATE
from puzzle_search import depth_limited_search, iterative_deepening_search, astar_search, PuzzleSearchStats


@dataclass
class UnifiedSearchResult:
    """
    Standardized search result schema shared by both 8-Puzzle and Tic-Tac-Toe.
    Guarantees consistent data representation across games and algorithms.
    """
    game_name: str
    search_algorithm: str
    search_depth: int
    nodes_generated: int
    nodes_expanded: int
    cutoff_nodes: int
    terminal_nodes: int  # goal nodes for 8-puzzle, terminal leaves for tic-tac-toe
    execution_time_ms: float
    solution_found: bool
    evaluation_score: Optional[float] = None
    solution_path: Optional[List[Any]] = None
    extra_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.solution_path is not None:
            d["solution_length"] = len(self.solution_path)
        else:
            d["solution_length"] = 0
        return d


class GameHistoryManager:
    """Manages Tic-Tac-Toe move history and match records during the user's session."""

    def __init__(self):
        self.move_history: List[Dict] = []
        self.game_records: List[Dict] = []

    def record_move(
        self,
        move_number: int,
        player: str,
        cell_index: int,
        search_stats: Optional[Dict] = None,
    ):
        """Record a single move made by human or AI."""
        row, col = cell_index // 3, cell_index % 3
        record = {
            "move_number": move_number,
            "player": player,
            "cell_index": cell_index,
            "position": f"({row}, {col})",
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "search_depth": search_stats.get("max_depth_setting", "-") if search_stats else "-",
            "nodes_generated": search_stats.get("nodes_generated", "-") if search_stats else "-",
            "nodes_expanded": search_stats.get("nodes_expanded", "-") if search_stats else "-",
            "cutoff_nodes": search_stats.get("cutoff_nodes", "-") if search_stats else "-",
            "terminal_nodes": search_stats.get("terminal_nodes", "-") if search_stats else "-",
            "execution_time_ms": search_stats.get("execution_time_ms", "-") if search_stats else "-",
            "score": search_stats.get("evaluation_score", "-") if search_stats else "-",
        }
        self.move_history.append(record)

    def record_game_completion(
        self,
        game_id: int,
        winner: str,
        total_moves: int,
        search_depth: int,
        ai_moves_count: int,
        total_nodes: int,
    ):
        """Record the final outcome and aggregate stats of a finished match."""
        outcome = "AI Won (O)" if winner == "O" else ("Human Won (X)" if winner == "X" else "Tie / Draw")
        record = {
            "game_id": game_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "outcome": outcome,
            "winner": winner,
            "total_moves": total_moves,
            "search_depth": search_depth,
            "ai_moves": ai_moves_count,
            "total_nodes_explored": total_nodes,
        }
        self.game_records.append(record)

    def clear(self):
        """Reset all move and game histories."""
        self.move_history.clear()
        self.game_records.clear()

    def get_moves_df(self) -> pd.DataFrame:
        """Return moves as a pandas DataFrame."""
        if not self.move_history:
            return pd.DataFrame(
                columns=[
                    "move_number",
                    "player",
                    "position",
                    "timestamp",
                    "search_depth",
                    "nodes_generated",
                    "nodes_expanded",
                    "cutoff_nodes",
                    "terminal_nodes",
                    "execution_time_ms",
                    "score",
                ]
            )
        return pd.DataFrame(self.move_history)

    def get_games_df(self) -> pd.DataFrame:
        """Return completed games as a pandas DataFrame."""
        if not self.game_records:
            return pd.DataFrame(
                columns=[
                    "game_id",
                    "timestamp",
                    "outcome",
                    "total_moves",
                    "search_depth",
                    "ai_moves",
                    "total_nodes_explored",
                ]
            )
        return pd.DataFrame(self.game_records)


def run_reusable_search_analysis(
    game_name: str,
    search_algorithm: str,
    initial_state: Any,
    max_search_depth: int,
) -> Dict[str, Any]:
    """
    Unified analysis tool accepting:
        - Game name ('8-Puzzle' or 'Tic-Tac-Toe')
        - Search algorithm ('DLS', 'IDDFS', 'A*', 'Minimax', 'Depth-Limited Minimax')
        - Initial state (board tuple for 8-puzzle, TicTacToeGame for tic-tac-toe)
        - Maximum search depth (integer)

    Returns standardized dictionary with:
        - Nodes generated
        - Nodes expanded
        - Cutoff nodes
        - Goal or terminal nodes
        - Execution time
        - Search depth
    """
    g_lower = game_name.lower()
    algo_lower = search_algorithm.lower()

    if "puzzle" in g_lower:
        # Convert state to tuple if needed
        if isinstance(initial_state, EightPuzzleGame):
            state = initial_state.state
        elif isinstance(initial_state, (list, tuple)):
            state = tuple(initial_state)
        else:
            state = GOAL_STATE

        if "iterative" in algo_lower or "iddfs" in algo_lower:
            raw = iterative_deepening_search(state, max_depth_limit=max_search_depth)
        elif "a*" in algo_lower or "manhattan" in algo_lower or "informed" in algo_lower:
            raw = astar_search(state)
        else:  # Default to DLS
            raw = depth_limited_search(state, max_depth=max_search_depth)

        result = UnifiedSearchResult(
            game_name="8-Puzzle",
            search_algorithm=raw.get("search_algorithm", "Depth-Limited Search (DLS)"),
            search_depth=raw.get("search_depth", max_search_depth),
            nodes_generated=raw.get("nodes_generated", 0),
            nodes_expanded=raw.get("nodes_expanded", 0),
            cutoff_nodes=raw.get("cutoff_nodes", 0),
            terminal_nodes=raw.get("goal_nodes", 0),
            execution_time_ms=raw.get("execution_time_ms", 0.0),
            solution_found=raw.get("solution_found", False),
            solution_path=raw.get("solution_path"),
            extra_info={"initial_state": state},
        )
        return result.to_dict()

    elif "tic" in g_lower:
        if isinstance(initial_state, TicTacToeGame):
            game_obj = initial_state.clone()
        elif isinstance(initial_state, list):
            game_obj = TicTacToeGame(board=initial_state)
        else:
            game_obj = TicTacToeGame()

        best_move, stats = get_best_ai_move(game_obj, max_depth=max_search_depth)

        result = UnifiedSearchResult(
            game_name="Tic-Tac-Toe",
            search_algorithm="Depth-Limited Minimax",
            search_depth=max_search_depth,
            nodes_generated=stats.get("nodes_generated", 0),
            nodes_expanded=stats.get("nodes_expanded", 0),
            cutoff_nodes=stats.get("cutoff_nodes", 0),
            terminal_nodes=stats.get("terminal_nodes", 0),
            execution_time_ms=stats.get("execution_time_ms", 0.0),
            solution_found=(best_move is not None),
            evaluation_score=stats.get("evaluation_score"),
            extra_info={
                "selected_move": best_move,
                "max_depth_reached": stats.get("max_depth_reached", 0),
            },
        )
        return result.to_dict()

    else:
        raise ValueError(f"Unknown game name: {game_name}. Choose '8-Puzzle' or 'Tic-Tac-Toe'.")


def run_depth_benchmark(
    game: TicTacToeGame, depths: Optional[List[int]] = None
) -> List[Dict[str, Any]]:
    """
    Executes depth-limited minimax for depths 1 through 9 on the EXACT same Tic-Tac-Toe board state.
    Provides rigorous empirical comparison of nodes, cutoffs, terminals, and latency.
    """
    if depths is None:
        depths = list(range(1, 10))

    results = []
    for d in depths:
        clone = game.clone()
        best_move, stats = get_best_ai_move(clone, max_depth=d)

        results.append(
            {
                "game_name": "Tic-Tac-Toe",
                "search_algorithm": "Depth-Limited Minimax",
                "depth": d,
                "nodes_generated": stats.get("nodes_generated", 0),
                "nodes_expanded": stats.get("nodes_expanded", 0),
                "cutoff_nodes": stats.get("cutoff_nodes", 0),
                "terminal_nodes": stats.get("terminal_nodes", 0),
                "execution_time_ms": stats.get("execution_time_ms", 0.0),
                "selected_move": best_move,
                "selected_pos": f"({best_move // 3}, {best_move % 3})" if best_move is not None else "N/A",
                "evaluation_score": stats.get("evaluation_score", 0),
                "max_depth_reached": stats.get("max_depth_reached", 0),
            }
        )

    return results


def run_combined_depth_benchmark(
    puzzle_state: Tuple[int, ...],
    ttt_game: TicTacToeGame,
    depths: Optional[List[int]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Runs benchmark across common depths (default: 1 through 9) for BOTH 8-Puzzle and Tic-Tac-Toe.
    Enables side-by-side comparative analysis of search tree expansion and cutoff behaviors.
    """
    if depths is None:
        depths = list(range(1, 10))

    puzzle_results = []
    ttt_results = []

    for d in depths:
        p_row = depth_limited_search(puzzle_state, max_depth=d)
        puzzle_results.append(p_row)

        clone = ttt_game.clone()
        best_move, t_stats = get_best_ai_move(clone, max_depth=d)
        t_row = {
            "game_name": "Tic-Tac-Toe",
            "search_algorithm": "Depth-Limited Minimax",
            "search_depth": d,
            "nodes_generated": t_stats.get("nodes_generated", 0),
            "nodes_expanded": t_stats.get("nodes_expanded", 0),
            "cutoff_nodes": t_stats.get("cutoff_nodes", 0),
            "terminal_nodes": t_stats.get("terminal_nodes", 0),
            "execution_time_ms": t_stats.get("execution_time_ms", 0.0),
            "selected_move": best_move,
            "evaluation_score": t_stats.get("evaluation_score", 0),
        }
        ttt_results.append(t_row)

    return {
        "depths": depths,
        "puzzle": puzzle_results,
        "tictactoe": ttt_results,
    }


def calculate_dashboard_metrics(all_ai_stats: List[Dict]) -> Dict[str, Any]:
    """Computes overall summary KPIs across all Tic-Tac-Toe AI moves made in session."""
    if not all_ai_stats:
        return {
            "total_ai_moves": 0,
            "total_nodes_generated": 0,
            "total_nodes_expanded": 0,
            "total_cutoff_nodes": 0,
            "total_terminal_nodes": 0,
            "avg_nodes_per_move": 0.0,
            "avg_execution_time_ms": 0.0,
            "max_execution_time_ms": 0.0,
        }

    total_moves = len(all_ai_stats)
    total_nodes = sum(s.get("nodes_generated", 0) for s in all_ai_stats)
    total_expanded = sum(s.get("nodes_expanded", 0) for s in all_ai_stats)
    total_cutoffs = sum(s.get("cutoff_nodes", 0) for s in all_ai_stats)
    total_terminals = sum(s.get("terminal_nodes", 0) for s in all_ai_stats)
    times = [s.get("execution_time_ms", 0.0) for s in all_ai_stats]

    return {
        "total_ai_moves": total_moves,
        "total_nodes_generated": total_nodes,
        "total_nodes_expanded": total_expanded,
        "total_cutoff_nodes": total_cutoffs,
        "total_terminal_nodes": total_terminals,
        "avg_nodes_per_move": round(total_nodes / total_moves, 1) if total_moves else 0.0,
        "avg_execution_time_ms": round(sum(times) / total_moves, 2) if total_moves else 0.0,
        "max_execution_time_ms": round(max(times), 2) if times else 0.0,
    }


def calculate_puzzle_dashboard_metrics(all_puzzle_searches: List[Dict]) -> Dict[str, Any]:
    """Computes overall summary KPIs across all 8-Puzzle searches executed in session."""
    if not all_puzzle_searches:
        return {
            "total_searches": 0,
            "total_nodes_generated": 0,
            "total_nodes_expanded": 0,
            "total_cutoff_nodes": 0,
            "total_goal_nodes": 0,
            "avg_nodes_per_search": 0.0,
            "avg_execution_time_ms": 0.0,
            "max_search_depth": 0,
        }

    total_runs = len(all_puzzle_searches)
    total_nodes = sum(s.get("nodes_generated", 0) for s in all_puzzle_searches)
    total_expanded = sum(s.get("nodes_expanded", 0) for s in all_puzzle_searches)
    total_cutoffs = sum(s.get("cutoff_nodes", 0) for s in all_puzzle_searches)
    total_goals = sum(s.get("goal_nodes", 0) for s in all_puzzle_searches)
    times = [s.get("execution_time_ms", 0.0) for s in all_puzzle_searches]
    depths = [s.get("search_depth", 0) for s in all_puzzle_searches]

    return {
        "total_searches": total_runs,
        "total_nodes_generated": total_nodes,
        "total_nodes_expanded": total_expanded,
        "total_cutoff_nodes": total_cutoffs,
        "total_goal_nodes": total_goals,
        "avg_nodes_per_search": round(total_nodes / total_runs, 1) if total_runs else 0.0,
        "avg_execution_time_ms": round(sum(times) / total_runs, 2) if total_runs else 0.0,
        "max_search_depth": max(depths) if depths else 0,
    }


def export_results_to_csv(records: List[Dict[str, Any]]) -> str:
    """
    Converts list of search benchmark dictionaries to a clean CSV string.
    Ready for download in Streamlit via st.download_button.
    """
    if not records:
        return "game_name,search_algorithm,search_depth,nodes_generated,nodes_expanded,cutoff_nodes,terminal_nodes,execution_time_ms\n"

    # Filter out complex nested fields like solution_path for clean tabular CSV
    clean_records = []
    for r in records:
        clean_r = {}
        for k, v in r.items():
            if k in ("solution_path", "per_depth_results", "extra_info", "move_scores"):
                continue
            clean_r[k] = v
        clean_records.append(clean_r)

    df = pd.DataFrame(clean_records)
    return df.to_csv(index=False)
