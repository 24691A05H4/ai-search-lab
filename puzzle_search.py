"""
puzzle_search.py - 8-Puzzle Search Algorithms & Profiler
Implements Depth-Limited Search (DLS), Iterative Deepening Search (IDDFS),
and A* Search with Manhattan Distance heuristic.
Tracks empirical search statistics: nodes generated, expanded, cutoff, and latency.
"""

import heapq
import time
from typing import Dict, List, Optional, Set, Tuple, Any
from puzzle import EightPuzzleGame, GOAL_STATE


class PuzzleSearchStats:
    """Telemetry collector for 8-puzzle search algorithms."""

    def __init__(self):
        self.nodes_generated: int = 0
        self.nodes_expanded: int = 0
        self.cutoff_nodes: int = 0
        self.goal_nodes: int = 0
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
        self.goal_nodes = 0
        self.max_depth_reached = 0
        self.execution_time_ms = 0.0


def depth_limited_search(
    initial_state: Tuple[int, ...],
    max_depth: int,
    stats: Optional[PuzzleSearchStats] = None,
) -> Dict[str, Any]:
    """
    Executes Depth-Limited Search (DLS) on the 8-puzzle.
    Uses path-based cycle detection (ancestor state set) to avoid infinite loops.

    Definition of Cutoff Node:
        A search node where further expansion is prevented solely by the configured
        depth limit, excluding goal states.

    Parameters:
        initial_state: Starting 9-tuple of tiles (0 represents blank)
        max_depth: Maximum search depth limit
        stats: Optional existing PuzzleSearchStats accumulator

    Returns:
        Dictionary containing empirical metrics, solution path, and status.
    """
    if stats is None:
        stats = PuzzleSearchStats()
        stats.reset()

    stats.start_timer()
    
    # Root node is generated
    stats.nodes_generated += 1
    
    cutoff_occurred = False
    solution_path: Optional[List[Dict[str, Any]]] = None

    def _dls_recursive(
        state: Tuple[int, ...],
        current_depth: int,
        path_states: Set[Tuple[int, ...]],
        action_history: List[Tuple[str, int, Tuple[int, ...]]],
    ) -> Optional[str]:
        """
        Recursive helper for DLS.
        Returns:
            "SOLVED" if goal found
            "CUTOFF" if cutoff occurred
            "FAILURE" if exhausted without cutoff
        """
        nonlocal cutoff_occurred, solution_path
        stats.max_depth_reached = max(stats.max_depth_reached, current_depth)

        # 1. Goal Check (Terminal Success)
        if state == GOAL_STATE:
            stats.goal_nodes += 1
            solution_path = [
                {"action": act, "tile": tile, "state": st}
                for act, tile, st in action_history
            ]
            return "SOLVED"

        # 2. Depth Cutoff Check
        # Excludes goal states (already verified above)
        if current_depth >= max_depth:
            stats.cutoff_nodes += 1
            cutoff_occurred = True
            return "CUTOFF"

        # 3. Node Expansion
        stats.nodes_expanded += 1
        any_cutoff = False

        for action_name, next_state, moved_tile in EightPuzzleGame.get_successors(state):
            # Path-based cycle detection: ignore states already in current branch path
            if next_state in path_states:
                continue

            stats.nodes_generated += 1
            path_states.add(next_state)
            action_history.append((action_name, moved_tile, next_state))

            result = _dls_recursive(
                state=next_state,
                current_depth=current_depth + 1,
                path_states=path_states,
                action_history=action_history,
            )

            # Backtrack
            action_history.pop()
            path_states.remove(next_state)

            if result == "SOLVED":
                return "SOLVED"
            elif result == "CUTOFF":
                any_cutoff = True

        return "CUTOFF" if any_cutoff else "FAILURE"

    # Kickoff search with root state in ancestor path
    init_path_states = {initial_state}
    init_history: List[Tuple[str, int, Tuple[int, ...]]] = []

    result_status = _dls_recursive(
        state=initial_state,
        current_depth=0,
        path_states=init_path_states,
        action_history=init_history,
    )

    stats.stop_timer()

    is_solved = (result_status == "SOLVED")

    return {
        "game_name": "8-Puzzle",
        "search_algorithm": "Depth-Limited Search (DLS)",
        "search_depth": max_depth,
        "max_depth_reached": stats.max_depth_reached,
        "nodes_generated": stats.nodes_generated,
        "nodes_expanded": stats.nodes_expanded,
        "cutoff_nodes": stats.cutoff_nodes,
        "goal_nodes": stats.goal_nodes,
        "terminal_nodes": stats.goal_nodes,  # symmetric naming with minimax
        "execution_time_ms": round(stats.execution_time_ms, 3),
        "solution_found": is_solved,
        "solution_path": solution_path,
        "solution_length": len(solution_path) if solution_path is not None else 0,
        "cutoff_occurred": cutoff_occurred,
        "initial_state": initial_state,
        "manhattan_distance": EightPuzzleGame.manhattan_distance(initial_state),
    }


def iterative_deepening_search(
    initial_state: Tuple[int, ...],
    max_depth_limit: int = 20,
) -> Dict[str, Any]:
    """
    Executes Iterative Deepening Depth-First Search (IDDFS) for 8-puzzle.
    Sequentially invokes DLS from depth 0 up to max_depth_limit.

    Returns:
        Comprehensive dictionary containing aggregated metrics and per-depth progression.
    """
    overall_start = time.perf_counter()
    per_depth_results = []
    
    total_nodes_generated = 0
    total_nodes_expanded = 0
    total_cutoff_nodes = 0
    total_goal_nodes = 0
    solution_found = False
    solution_path = None
    solution_depth = -1

    for d in range(0, max_depth_limit + 1):
        step_stats = PuzzleSearchStats()
        step_result = depth_limited_search(initial_state, max_depth=d, stats=step_stats)
        
        per_depth_results.append(step_result)
        total_nodes_generated += step_result["nodes_generated"]
        total_nodes_expanded += step_result["nodes_expanded"]
        total_cutoff_nodes += step_result["cutoff_nodes"]
        total_goal_nodes += step_result["goal_nodes"]

        if step_result["solution_found"]:
            solution_found = True
            solution_path = step_result["solution_path"]
            solution_depth = d
            break

    total_time_ms = (time.perf_counter() - overall_start) * 1000.0

    return {
        "game_name": "8-Puzzle",
        "search_algorithm": "Iterative Deepening Search (IDDFS)",
        "search_depth": solution_depth if solution_found else max_depth_limit,
        "max_depth_limit": max_depth_limit,
        "nodes_generated": total_nodes_generated,
        "nodes_expanded": total_nodes_expanded,
        "cutoff_nodes": total_cutoff_nodes,
        "goal_nodes": total_goal_nodes,
        "terminal_nodes": total_goal_nodes,
        "execution_time_ms": round(total_time_ms, 3),
        "solution_found": solution_found,
        "solution_path": solution_path,
        "solution_length": len(solution_path) if solution_path is not None else 0,
        "per_depth_results": per_depth_results,
        "initial_state": initial_state,
    }


def astar_search(
    initial_state: Tuple[int, ...],
    max_expanded: int = 50000,
) -> Dict[str, Any]:
    """
    Executes A* Search with Manhattan Distance heuristic.
    Guarantees optimal shortest solution path for solvable 8-puzzles.
    """
    start_time = time.perf_counter()

    if initial_state == GOAL_STATE:
        return {
            "game_name": "8-Puzzle",
            "search_algorithm": "A* Search (Manhattan)",
            "search_depth": 0,
            "nodes_generated": 1,
            "nodes_expanded": 0,
            "cutoff_nodes": 0,
            "goal_nodes": 1,
            "terminal_nodes": 1,
            "execution_time_ms": 0.05,
            "solution_found": True,
            "solution_path": [],
            "solution_length": 0,
        }

    # Priority Queue holds: (f_score, g_score, counter, state, action_history)
    counter = 0
    h_init = EightPuzzleGame.manhattan_distance(initial_state)
    frontier = [(h_init, 0, counter, initial_state, [])]
    
    # Cost so far: state -> g_score
    cost_so_far: Dict[Tuple[int, ...], int] = {initial_state: 0}
    
    nodes_generated = 1
    nodes_expanded = 0
    max_depth_reached = 0

    while frontier:
        f, g, _, current_state, history = heapq.heappop(frontier)

        if current_state == GOAL_STATE:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "game_name": "8-Puzzle",
                "search_algorithm": "A* Search (Manhattan)",
                "search_depth": g,
                "max_depth_reached": max_depth_reached,
                "nodes_generated": nodes_generated,
                "nodes_expanded": nodes_expanded,
                "cutoff_nodes": 0,
                "goal_nodes": 1,
                "terminal_nodes": 1,
                "execution_time_ms": round(elapsed_ms, 3),
                "solution_found": True,
                "solution_path": history,
                "solution_length": len(history),
            }

        if g > cost_so_far.get(current_state, float("inf")):
            continue

        nodes_expanded += 1
        if nodes_expanded >= max_expanded:
            break

        for action_name, next_state, moved_tile in EightPuzzleGame.get_successors(current_state):
            new_g = g + 1
            if next_state not in cost_so_far or new_g < cost_so_far[next_state]:
                cost_so_far[next_state] = new_g
                h = EightPuzzleGame.manhattan_distance(next_state)
                nodes_generated += 1
                counter += 1
                max_depth_reached = max(max_depth_reached, new_g)
                next_history = history + [{"action": action_name, "tile": moved_tile, "state": next_state}]
                heapq.heappush(frontier, (new_g + h, new_g, counter, next_state, next_history))

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return {
        "game_name": "8-Puzzle",
        "search_algorithm": "A* Search (Manhattan)",
        "search_depth": max_depth_reached,
        "max_depth_reached": max_depth_reached,
        "nodes_generated": nodes_generated,
        "nodes_expanded": nodes_expanded,
        "cutoff_nodes": 0,
        "goal_nodes": 0,
        "terminal_nodes": 0,
        "execution_time_ms": round(elapsed_ms, 3),
        "solution_found": False,
        "solution_path": None,
        "solution_length": 0,
    }


def run_puzzle_depth_benchmark(
    initial_state: Tuple[int, ...],
    depths: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Evaluates Depth-Limited Search (DLS) on the exact same initial 8-puzzle state
    across depths (default: 1 through 10).
    Guarantees consistent, fair empirical comparison.
    """
    if depths is None:
        depths = list(range(1, 11))

    benchmark_rows = []
    for d in depths:
        stats = PuzzleSearchStats()
        row = depth_limited_search(initial_state, max_depth=d, stats=stats)
        benchmark_rows.append(row)

    return benchmark_rows


# Preset benchmark configurations of verified optimal solution depths
PUZZLE_PRESETS: Dict[str, Tuple[int, ...]] = {
    "1. Very Easy (1 Move to Goal)": (1, 2, 3, 4, 5, 0, 7, 8, 6),
    "2. Easy (2 Moves to Goal)": (1, 2, 3, 4, 0, 5, 7, 8, 6),
    "3. Moderate (4 Moves to Goal)": (0, 1, 3, 4, 2, 5, 7, 8, 6),
    "4. Tactical (6 Moves to Goal)": (4, 1, 3, 7, 2, 5, 0, 8, 6),
    "5. Deep Challenge (8 Moves to Goal)": (4, 1, 3, 7, 2, 5, 8, 6, 0),
}
