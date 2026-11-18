"""
visualizer_engine.py - Interactive Step-by-Step AI Search Visualizer Engine
Provides a unified event-recording system, step-by-step telemetry,
active branch tracking, and tree layout generator for both 8-Puzzle and Tic-Tac-Toe.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import copy
from puzzle import EightPuzzleGame, GOAL_STATE
from game import TicTacToeGame
from minimax import evaluate_board_heuristic


class EventType(str, Enum):
    NODE_GENERATED = "NODE_GENERATED"
    NODE_SELECTED = "NODE_SELECTED"
    NODE_EXPANDED = "NODE_EXPANDED"
    NODE_EVALUATED = "NODE_EVALUATED"
    NODE_CUTOFF = "NODE_CUTOFF"
    NODE_TERMINAL = "NODE_TERMINAL"
    NODE_PRUNED = "NODE_PRUNED"
    GOAL_FOUND = "GOAL_FOUND"
    BEST_MOVE_SELECTED = "BEST_MOVE_SELECTED"


@dataclass
class SearchNode:
    """Represents a node in the visual search tree."""
    node_id: int
    parent_id: Optional[int]
    depth: int
    state: Any  # Tuple for 8-puzzle, List[str] for TicTacToe
    action: Optional[str] = None
    player: Optional[str] = None  # 'MAX' (AI) or 'MIN' (Human)
    score: Optional[float] = None
    status: str = "UNEXPLORED"  # 'UNEXPLORED', 'EXPANDED', 'CUTOFF', 'TERMINAL', 'PRUNED', 'GOAL'
    is_terminal: bool = False
    terminal_outcome: Optional[str] = None
    children_ids: List[int] = field(default_factory=list)


@dataclass
class VisualizationStep:
    """Represents a single discrete step in the search algorithm's execution."""
    step_number: int
    event_type: str
    node_id: int
    parent_id: Optional[int]
    current_depth: int
    state: Any
    action_taken: Optional[str]
    score: Optional[float]
    nodes_generated_count: int
    nodes_expanded_count: int
    cutoff_count: int
    terminal_count: int
    explanation: str
    active_path: List[int]  # List of node IDs from root to current node
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SearchVisualizerLog:
    """Manages recorded events and provides replay/navigation methods."""

    def __init__(self, game_name: str, initial_state: Any, max_depth: int):
        self.game_name = game_name
        self.initial_state = initial_state
        self.max_depth = max_depth
        self.steps: List[VisualizationStep] = []
        self.nodes: Dict[int, SearchNode] = {}
        self.total_steps: int = 0

    def add_node(self, node: SearchNode):
        self.nodes[node.node_id] = node
        if node.parent_id is not None and node.parent_id in self.nodes:
            if node.node_id not in self.nodes[node.parent_id].children_ids:
                self.nodes[node.parent_id].children_ids.append(node.node_id)

    def add_step(self, step: VisualizationStep):
        self.steps.append(step)
        self.total_steps = len(self.steps)

    def get_step(self, step_index: int) -> Optional[VisualizationStep]:
        if 0 <= step_index < len(self.steps):
            return self.steps[step_index]
        return None

    def get_path_to_node(self, node_id: int) -> List[int]:
        """Returns the list of node IDs forming the ancestor path from root to node_id."""
        path = []
        curr = node_id
        while curr is not None and curr in self.nodes:
            path.append(curr)
            curr = self.nodes[curr].parent_id
        return list(reversed(path))

    def get_nodes_snapshot_at_step(self, step_index: int) -> Dict[int, SearchNode]:
        """
        Reconstructs the exact state of the search tree up to `step_index`.
        Guarantees nodes only appear after NODE_GENERATED and reflect live status/score updates!
        """
        snapshot: Dict[int, SearchNode] = {}
        limit = min(step_index + 1, len(self.steps))

        for i in range(limit):
            s = self.steps[i]
            nid = s.node_id
            orig = self.nodes.get(nid)
            if orig is None:
                continue

            if nid not in snapshot:
                snapshot[nid] = SearchNode(
                    node_id=orig.node_id,
                    parent_id=orig.parent_id,
                    depth=orig.depth,
                    state=orig.state,
                    action=orig.action,
                    player=orig.player,
                    score=None,
                    status="UNEXPLORED",
                    is_terminal=False,
                    terminal_outcome=None,
                    children_ids=[],
                )
                if orig.parent_id is not None and orig.parent_id in snapshot:
                    if nid not in snapshot[orig.parent_id].children_ids:
                        snapshot[orig.parent_id].children_ids.append(nid)

            snap_node = snapshot[nid]
            ev = s.event_type
            if ev == EventType.NODE_EXPANDED.value:
                snap_node.status = "EXPANDED"
            elif ev == EventType.NODE_CUTOFF.value:
                snap_node.status = "CUTOFF"
                snap_node.score = s.score
            elif ev == EventType.NODE_PRUNED.value:
                snap_node.status = "PRUNED"
            elif ev == EventType.NODE_TERMINAL.value:
                snap_node.status = "TERMINAL"
                snap_node.is_terminal = True
                snap_node.score = s.score
                snap_node.terminal_outcome = orig.terminal_outcome
            elif ev == EventType.GOAL_FOUND.value:
                snap_node.status = "GOAL"
                snap_node.is_terminal = True
                snap_node.score = s.score
            elif ev == EventType.NODE_EVALUATED.value:
                snap_node.score = s.score
            elif ev == EventType.BEST_MOVE_SELECTED.value:
                snap_node.score = s.score

        return snapshot


# =========================================================
# 8-PUZZLE VISUAL SEARCH RECORDER
# =========================================================

def record_8puzzle_search_events(
    initial_state: Tuple[int, ...],
    max_depth: int,
    max_steps_cap: int = 1500,
) -> SearchVisualizerLog:
    """
    Executes Depth-Limited Search (DLS) on the 8-puzzle while capturing
    every single algorithmic event for step-by-step playback.
    """
    log = SearchVisualizerLog("8-Puzzle", initial_state, max_depth)

    node_counter = 0
    nodes_generated = 0
    nodes_expanded = 0
    cutoff_nodes = 0
    goal_nodes = 0
    step_num = 0

    # Create root node
    root_node = SearchNode(
        node_id=0,
        parent_id=None,
        depth=0,
        state=initial_state,
        action="ROOT_START",
        status="SELECTED",
    )
    log.add_node(root_node)
    nodes_generated += 1

    log.add_step(
        VisualizationStep(
            step_number=step_num,
            event_type=EventType.NODE_GENERATED.value,
            node_id=0,
            parent_id=None,
            current_depth=0,
            state=initial_state,
            action_taken="Initialize Search at Root State",
            score=None,
            nodes_generated_count=nodes_generated,
            nodes_expanded_count=nodes_expanded,
            cutoff_count=cutoff_nodes,
            terminal_count=goal_nodes,
            explanation=f"Root node 0 created at Depth 0 with initial state. Manhattan Distance: {EightPuzzleGame.manhattan_distance(initial_state)}.",
            active_path=[0],
        )
    )
    step_num += 1

    solution_found = False

    def _dls_record(
        curr_node_id: int,
        path_states: Set[Tuple[int, ...]],
    ):
        nonlocal step_num, nodes_generated, nodes_expanded, cutoff_nodes, goal_nodes, node_counter, solution_found
        if solution_found or step_num >= max_steps_cap:
            return

        curr_node = log.nodes[curr_node_id]
        state = curr_node.state
        depth = curr_node.depth
        active_path = log.get_path_to_node(curr_node_id)

        # Event: NODE_SELECTED
        log.add_step(
            VisualizationStep(
                step_number=step_num,
                event_type=EventType.NODE_SELECTED.value,
                node_id=curr_node_id,
                parent_id=curr_node.parent_id,
                current_depth=depth,
                state=state,
                action_taken=f"Select Node {curr_node_id}",
                score=None,
                nodes_generated_count=nodes_generated,
                nodes_expanded_count=nodes_expanded,
                cutoff_count=cutoff_nodes,
                terminal_count=goal_nodes,
                explanation=f"Inspecting Node {curr_node_id} at Depth {depth}. Checking goal condition and depth cutoff threshold.",
                active_path=active_path,
            )
        )
        step_num += 1

        # 1. Goal Check
        if state == GOAL_STATE:
            goal_nodes += 1
            curr_node.status = "GOAL"
            curr_node.is_terminal = True
            solution_found = True

            log.add_step(
                VisualizationStep(
                    step_number=step_num,
                    event_type=EventType.GOAL_FOUND.value,
                    node_id=curr_node_id,
                    parent_id=curr_node.parent_id,
                    current_depth=depth,
                    state=state,
                    action_taken="GOAL REACHED! 🎉",
                    score=0,
                    nodes_generated_count=nodes_generated,
                    nodes_expanded_count=nodes_expanded,
                    cutoff_count=cutoff_nodes,
                    terminal_count=goal_nodes,
                    explanation=f"🎯 GOAL STATE FOUND at Node {curr_node_id} (Depth {depth})! Tiles match [1, 2, 3, 4, 5, 6, 7, 8, 0]. Solution path confirmed.",
                    active_path=active_path,
                )
            )
            step_num += 1
            return

        # 2. Depth Limit Check (Cutoff)
        if depth >= max_depth:
            cutoff_nodes += 1
            curr_node.status = "CUTOFF"

            log.add_step(
                VisualizationStep(
                    step_number=step_num,
                    event_type=EventType.NODE_CUTOFF.value,
                    node_id=curr_node_id,
                    parent_id=curr_node.parent_id,
                    current_depth=depth,
                    state=state,
                    action_taken="Depth Limit Cutoff",
                    score=None,
                    nodes_generated_count=nodes_generated,
                    nodes_expanded_count=nodes_expanded,
                    cutoff_count=cutoff_nodes,
                    terminal_count=goal_nodes,
                    explanation=f"✂️ Depth limit ({max_depth}) reached at Node {curr_node_id}. Further branch expansion is halted; backtracking to ancestor.",
                    active_path=active_path,
                )
            )
            step_num += 1
            return

        # 3. Node Expansion
        nodes_expanded += 1
        curr_node.status = "EXPANDED"

        successors = EightPuzzleGame.get_successors(state)
        log.add_step(
            VisualizationStep(
                step_number=step_num,
                event_type=EventType.NODE_EXPANDED.value,
                node_id=curr_node_id,
                parent_id=curr_node.parent_id,
                current_depth=depth,
                state=state,
                action_taken=f"Expand Node {curr_node_id}",
                score=None,
                nodes_generated_count=nodes_generated,
                nodes_expanded_count=nodes_expanded,
                cutoff_count=cutoff_nodes,
                terminal_count=goal_nodes,
                explanation=f"Expanding Node {curr_node_id}: Found {len(successors)} legal tile transitions into empty space.",
                active_path=active_path,
                extra={"successor_count": len(successors)},
            )
        )
        step_num += 1

        for action_name, next_state, moved_tile in successors:
            if solution_found or step_num >= max_steps_cap:
                break

            node_counter += 1
            child_id = node_counter
            nodes_generated += 1

            # Cycle Check
            if next_state in path_states:
                child_node = SearchNode(
                    node_id=child_id,
                    parent_id=curr_node_id,
                    depth=depth + 1,
                    state=next_state,
                    action=f"{action_name} (Tile {moved_tile})",
                    status="PRUNED",
                )
                log.add_node(child_node)

                log.add_step(
                    VisualizationStep(
                        step_number=step_num,
                        event_type=EventType.NODE_PRUNED.value,
                        node_id=child_id,
                        parent_id=curr_node_id,
                        current_depth=depth + 1,
                        state=next_state,
                        action_taken=f"Prune Cycle: {action_name}",
                        score=None,
                        nodes_generated_count=nodes_generated,
                        nodes_expanded_count=nodes_expanded,
                        cutoff_count=cutoff_nodes,
                        terminal_count=goal_nodes,
                        explanation=f"🚫 Cycle detected! State from moving Tile {moved_tile} already exists on the active branch path. Node {child_id} pruned to prevent infinite loops.",
                        active_path=active_path + [child_id],
                    )
                )
                step_num += 1
                continue

            # Valid Child Node
            child_node = SearchNode(
                node_id=child_id,
                parent_id=curr_node_id,
                depth=depth + 1,
                state=next_state,
                action=f"{action_name} (Tile {moved_tile})",
                status="UNEXPLORED",
            )
            log.add_node(child_node)

            log.add_step(
                VisualizationStep(
                    step_number=step_num,
                    event_type=EventType.NODE_GENERATED.value,
                    node_id=child_id,
                    parent_id=curr_node_id,
                    current_depth=depth + 1,
                    state=next_state,
                    action_taken=f"Generate Child: {action_name} (Tile {moved_tile})",
                    score=None,
                    nodes_generated_count=nodes_generated,
                    nodes_expanded_count=nodes_expanded,
                    cutoff_count=cutoff_nodes,
                    terminal_count=goal_nodes,
                    explanation=f"Generated Node {child_id} at Depth {depth + 1} by sliding Tile {moved_tile} into blank space.",
                    active_path=active_path + [child_id],
                )
            )
            step_num += 1

            # Recurse
            path_states.add(next_state)
            _dls_record(child_id, path_states)
            path_states.remove(next_state)

    _dls_record(0, {initial_state})
    return log


# =========================================================
# TIC-TAC-TOE VISUAL MINIMAX RECORDER
# =========================================================

def record_minimax_search_events(
    game: TicTacToeGame,
    max_depth: int,
    max_steps_cap: int = 1500,
) -> SearchVisualizerLog:
    """
    Executes Depth-Limited Minimax while recording every decision step:
    MAX/MIN alternation, terminal state detections, heuristic cutoffs,
    and bottom-up score propagation.
    """
    initial_board = list(game.board)
    log = SearchVisualizerLog("Tic-Tac-Toe", initial_board, max_depth)

    node_counter = 0
    nodes_generated = 0
    nodes_expanded = 0
    cutoff_nodes = 0
    terminal_nodes = 0
    step_num = 0

    root_node = SearchNode(
        node_id=0,
        parent_id=None,
        depth=0,
        state=initial_board,
        action="ROOT_START",
        player="MAX",
        status="SELECTED",
    )
    log.add_node(root_node)
    nodes_generated += 1

    log.add_step(
        VisualizationStep(
            step_number=step_num,
            event_type=EventType.NODE_GENERATED.value,
            node_id=0,
            parent_id=None,
            current_depth=0,
            state=initial_board,
            action_taken="Root Minimax Evaluation (AI 'O')",
            score=None,
            nodes_generated_count=nodes_generated,
            nodes_expanded_count=nodes_expanded,
            cutoff_count=cutoff_nodes,
            terminal_count=terminal_nodes,
            explanation=f"Root node 0 initialized. AI ('O') is the MAXIMIZING player, seeking the move that yields highest utility.",
            active_path=[0],
        )
    )
    step_num += 1

    def _minimax_record(
        curr_node_id: int,
        active_game: TicTacToeGame,
        current_depth: int,
        is_maximizing: bool,
    ) -> int:
        nonlocal step_num, nodes_generated, nodes_expanded, cutoff_nodes, terminal_nodes, node_counter

        curr_node = log.nodes[curr_node_id]
        active_path = log.get_path_to_node(curr_node_id)
        player_role = "MAX (AI 'O')" if is_maximizing else "MIN (Human 'X')"

        # Event: NODE_SELECTED
        log.add_step(
            VisualizationStep(
                step_number=step_num,
                event_type=EventType.NODE_SELECTED.value,
                node_id=curr_node_id,
                parent_id=curr_node.parent_id,
                current_depth=current_depth,
                state=list(active_game.board),
                action_taken=f"Select Node {curr_node_id} ({player_role})",
                score=None,
                nodes_generated_count=nodes_generated,
                nodes_expanded_count=nodes_expanded,
                cutoff_count=cutoff_nodes,
                terminal_count=terminal_nodes,
                explanation=f"Inspecting Node {curr_node_id} at Depth {current_depth} for {player_role}. Testing terminal win/draw or depth limit.",
                active_path=active_path,
            )
        )
        step_num += 1

        # 1. Terminal Check
        winner = active_game.check_winner()
        if winner is not None:
            terminal_nodes += 1
            curr_node.is_terminal = True
            curr_node.status = "TERMINAL"
            curr_node.terminal_outcome = winner

            if winner == "O":
                score = 100 - current_depth
                desc = f"Terminal AI Win ('O')! Utility score = +{score} (incentivizes fast win)."
            elif winner == "X":
                score = -100 + current_depth
                desc = f"Terminal Human Win ('X')! Utility score = {score} (penalizes fast loss)."
            else:
                score = 0
                desc = "Terminal Stalemate (Draw)! Utility score = 0."

            curr_node.score = score
            log.add_step(
                VisualizationStep(
                    step_number=step_num,
                    event_type=EventType.NODE_TERMINAL.value,
                    node_id=curr_node_id,
                    parent_id=curr_node.parent_id,
                    current_depth=current_depth,
                    state=list(active_game.board),
                    action_taken=f"Terminal Outcome: {winner}",
                    score=score,
                    nodes_generated_count=nodes_generated,
                    nodes_expanded_count=nodes_expanded,
                    cutoff_count=cutoff_nodes,
                    terminal_count=terminal_nodes,
                    explanation=f"🏁 {desc}",
                    active_path=active_path,
                )
            )
            step_num += 1
            return score

        # 2. Depth Cutoff Check
        if current_depth >= max_depth:
            cutoff_nodes += 1
            curr_node.status = "CUTOFF"
            score = evaluate_board_heuristic(active_game)
            curr_node.score = score

            log.add_step(
                VisualizationStep(
                    step_number=step_num,
                    event_type=EventType.NODE_CUTOFF.value,
                    node_id=curr_node_id,
                    parent_id=curr_node.parent_id,
                    current_depth=current_depth,
                    state=list(active_game.board),
                    action_taken="Depth Limit Cutoff",
                    score=score,
                    nodes_generated_count=nodes_generated,
                    nodes_expanded_count=nodes_expanded,
                    cutoff_count=cutoff_nodes,
                    terminal_count=terminal_nodes,
                    explanation=f"✂️ Search depth limit ({max_depth}) reached at Node {curr_node_id}. Invoking domain heuristic evaluation: Score = {score}.",
                    active_path=active_path,
                )
            )
            step_num += 1
            return score

        # 3. Node Expansion
        nodes_expanded += 1
        curr_node.status = "EXPANDED"
        available_moves = active_game.get_available_moves()

        log.add_step(
            VisualizationStep(
                step_number=step_num,
                event_type=EventType.NODE_EXPANDED.value,
                node_id=curr_node_id,
                parent_id=curr_node.parent_id,
                current_depth=current_depth,
                state=list(active_game.board),
                action_taken=f"Expand {len(available_moves)} Legal Moves",
                score=None,
                nodes_generated_count=nodes_generated,
                nodes_expanded_count=nodes_expanded,
                cutoff_count=cutoff_nodes,
                terminal_count=terminal_nodes,
                explanation=f"Expanding Node {curr_node_id}: {len(available_moves)} empty cells available for {player_role}.",
                active_path=active_path,
            )
        )
        step_num += 1

        if is_maximizing:
            best_score = float("-inf")
            for move in available_moves:
                if step_num >= max_steps_cap:
                    break
                node_counter += 1
                child_id = node_counter
                nodes_generated += 1

                active_game.make_move(move, "O")
                child_node = SearchNode(
                    node_id=child_id,
                    parent_id=curr_node_id,
                    depth=current_depth + 1,
                    state=list(active_game.board),
                    action=f"AI plays Cell {move} ({move//3},{move%3})",
                    player="MIN",
                    status="UNEXPLORED",
                )
                log.add_node(child_node)

                log.add_step(
                    VisualizationStep(
                        step_number=step_num,
                        event_type=EventType.NODE_GENERATED.value,
                        node_id=child_id,
                        parent_id=curr_node_id,
                        current_depth=current_depth + 1,
                        state=list(active_game.board),
                        action_taken=f"Generate Move: O at {move}",
                        score=None,
                        nodes_generated_count=nodes_generated,
                        nodes_expanded_count=nodes_expanded,
                        cutoff_count=cutoff_nodes,
                        terminal_count=terminal_nodes,
                        explanation=f"Generated child Node {child_id} (Depth {current_depth + 1}) where AI ('O') plays at Cell {move}.",
                        active_path=active_path + [child_id],
                    )
                )
                step_num += 1

                child_score = _minimax_record(child_id, active_game, current_depth + 1, False)
                active_game.undo_move(move, previous_turn="O")

                if child_score > best_score:
                    best_score = child_score

            curr_node.score = best_score
            # Event: Score Propagated
            log.add_step(
                VisualizationStep(
                    step_number=step_num,
                    event_type=EventType.NODE_EVALUATED.value,
                    node_id=curr_node_id,
                    parent_id=curr_node.parent_id,
                    current_depth=current_depth,
                    state=list(active_game.board),
                    action_taken="Propagate MAX Score",
                    score=best_score,
                    nodes_generated_count=nodes_generated,
                    nodes_expanded_count=nodes_expanded,
                    cutoff_count=cutoff_nodes,
                    terminal_count=terminal_nodes,
                    explanation=f"Backtracking: MAX Node {curr_node_id} evaluated with maximum child utility = {best_score}.",
                    active_path=active_path,
                )
            )
            step_num += 1
            return int(best_score)

        else:
            best_score = float("inf")
            for move in available_moves:
                if step_num >= max_steps_cap:
                    break
                node_counter += 1
                child_id = node_counter
                nodes_generated += 1

                active_game.make_move(move, "X")
                child_node = SearchNode(
                    node_id=child_id,
                    parent_id=curr_node_id,
                    depth=current_depth + 1,
                    state=list(active_game.board),
                    action=f"Human plays Cell {move} ({move//3},{move%3})",
                    player="MAX",
                    status="UNEXPLORED",
                )
                log.add_node(child_node)

                log.add_step(
                    VisualizationStep(
                        step_number=step_num,
                        event_type=EventType.NODE_GENERATED.value,
                        node_id=child_id,
                        parent_id=curr_node_id,
                        current_depth=current_depth + 1,
                        state=list(active_game.board),
                        action_taken=f"Generate Move: X at {move}",
                        score=None,
                        nodes_generated_count=nodes_generated,
                        nodes_expanded_count=nodes_expanded,
                        cutoff_count=cutoff_nodes,
                        terminal_count=terminal_nodes,
                        explanation=f"Generated child Node {child_id} (Depth {current_depth + 1}) where Human ('X') plays at Cell {move}.",
                        active_path=active_path + [child_id],
                    )
                )
                step_num += 1

                child_score = _minimax_record(child_id, active_game, current_depth + 1, True)
                active_game.undo_move(move, previous_turn="X")

                if child_score < best_score:
                    best_score = child_score

            curr_node.score = best_score
            log.add_step(
                VisualizationStep(
                    step_number=step_num,
                    event_type=EventType.NODE_EVALUATED.value,
                    node_id=curr_node_id,
                    parent_id=curr_node.parent_id,
                    current_depth=current_depth,
                    state=list(active_game.board),
                    action_taken="Propagate MIN Score",
                    score=best_score,
                    nodes_generated_count=nodes_generated,
                    nodes_expanded_count=nodes_expanded,
                    cutoff_count=cutoff_nodes,
                    terminal_count=terminal_nodes,
                    explanation=f"Backtracking: MIN Node {curr_node_id} evaluated with minimum child utility = {best_score}.",
                    active_path=active_path,
                )
            )
            step_num += 1
            return int(best_score)

    sim_game = game.clone()
    final_score = _minimax_record(0, sim_game, 0, True)

    # Event: BEST_MOVE_SELECTED
    log.add_step(
        VisualizationStep(
            step_number=step_num,
            event_type=EventType.BEST_MOVE_SELECTED.value,
            node_id=0,
            parent_id=None,
            current_depth=0,
            state=initial_board,
            action_taken=f"Optimal Decision Finalized (Score: {final_score})",
            score=final_score,
            nodes_generated_count=nodes_generated,
            nodes_expanded_count=nodes_expanded,
            cutoff_count=cutoff_nodes,
            terminal_count=terminal_nodes,
            explanation=f"🏆 Minimax decision complete! The AI selects the move yielding optimal propagated score {final_score}.",
            active_path=[0],
        )
    )

    return log
