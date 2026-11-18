"""
puzzle.py - 8-Puzzle Game Engine
Defines the 3x3 sliding tile puzzle state, rules, moves, solvability parity,
and heuristic distance functions.
"""

import random
from typing import List, Optional, Tuple, Set


GOAL_STATE: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 0)


class EightPuzzleGame:
    """
    Represents an 8-Puzzle (3x3 sliding tile) game.
    Tiles are represented by integers 1 to 8, and 0 represents the empty (blank) space.
    Grid indices:
        0 1 2
        3 4 5
        6 7 8
    Goal state:
        1 2 3
        4 5 6
        7 8 0 (EMPTY)
    """

    # Precomputed adjacent moves for each blank position: index -> list of adjacent indices
    ADJACENCY = {
        0: [1, 3],        # Top-left can move right (1), down (3)
        1: [0, 2, 4],     # Top-center can move left (0), right (2), down (4)
        2: [1, 5],        # Top-right can move left (1), down (5)
        3: [0, 4, 6],     # Mid-left
        4: [1, 3, 5, 7],  # Center
        5: [2, 4, 8],     # Mid-right
        6: [3, 7],        # Bottom-left
        7: [4, 6, 8],     # Bottom-center
        8: [5, 7],        # Bottom-right
    }

    # Action names relative to blank tile movement: (delta_row, delta_col)
    MOVE_DIRECTIONS = {
        (-1, 0): "UP",
        (1, 0): "DOWN",
        (0, -1): "LEFT",
        (0, 1): "RIGHT",
    }

    def __init__(self, state: Optional[Tuple[int, ...]] = None):
        """Initialize with given state or standard goal state."""
        if state is not None:
            if len(state) != 9 or set(state) != set(range(9)):
                raise ValueError("State must be a permutation of integers 0 to 8.")
            self.state: Tuple[int, ...] = tuple(state)
        else:
            self.state = GOAL_STATE
        
        self.initial_state: Tuple[int, ...] = self.state
        self.move_count: int = 0
        self.move_history: List[Tuple[int, ...]] = [self.state]

    @property
    def blank_index(self) -> int:
        """Return the index (0..8) of the empty cell (0)."""
        return self.state.index(0)

    def is_solved(self) -> bool:
        """Check if current state matches GOAL_STATE."""
        return self.state == GOAL_STATE

    def get_valid_tile_moves(self) -> List[int]:
        """Return indices of tiles that can legally slide into the blank cell."""
        return self.ADJACENCY[self.blank_index]

    def can_move_tile(self, tile_index: int) -> bool:
        """Check if a tile at tile_index can slide into the blank space."""
        return tile_index in self.ADJACENCY[self.blank_index]

    def move_tile(self, tile_index: int) -> bool:
        """
        Slide the tile at tile_index into the blank space.
        Returns True if move succeeded, False otherwise.
        """
        blank_idx = self.blank_index
        if tile_index in self.ADJACENCY[blank_idx]:
            state_list = list(self.state)
            state_list[blank_idx], state_list[tile_index] = state_list[tile_index], state_list[blank_idx]
            self.state = tuple(state_list)
            self.move_count += 1
            self.move_history.append(self.state)
            return True
        return False

    def move_blank_direction(self, direction: str) -> bool:
        """
        Move the blank tile in direction: 'UP', 'DOWN', 'LEFT', or 'RIGHT'.
        """
        b_idx = self.blank_index
        b_r, b_c = b_idx // 3, b_idx % 3

        dir_map = {
            "UP": (-1, 0),
            "DOWN": (1, 0),
            "LEFT": (0, -1),
            "RIGHT": (0, 1),
        }
        if direction not in dir_map:
            return False

        dr, dc = dir_map[direction]
        nr, nc = b_r + dr, b_c + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            target_idx = nr * 3 + nc
            return self.move_tile(target_idx)
        return False

    def reset_to_initial(self):
        """Reset puzzle back to its starting state."""
        self.state = self.initial_state
        self.move_count = 0
        self.move_history = [self.state]

    def reset_to_goal(self):
        """Reset puzzle to solved goal state."""
        self.state = GOAL_STATE
        self.initial_state = GOAL_STATE
        self.move_count = 0
        self.move_history = [self.state]

    def set_state(self, new_state: Tuple[int, ...]):
        """Manually override board state."""
        if len(new_state) == 9 and set(new_state) == set(range(9)):
            self.state = tuple(new_state)
            self.initial_state = self.state
            self.move_count = 0
            self.move_history = [self.state]

    def clone(self) -> "EightPuzzleGame":
        """Return an independent copy of current puzzle state."""
        new_puzzle = EightPuzzleGame(self.state)
        new_puzzle.initial_state = self.initial_state
        new_puzzle.move_count = self.move_count
        new_puzzle.move_history = list(self.move_history)
        return new_puzzle

    def shuffle(self, steps: int = 25, seed: Optional[int] = None) -> Tuple[int, ...]:
        """
        Shuffles the puzzle by making random valid moves starting from GOAL_STATE.
        This guarantees 100% solvability parity!
        """
        if seed is not None:
            random.seed(seed)
        
        current_state = list(GOAL_STATE)
        last_blank = -1
        
        for _ in range(steps):
            b_idx = current_state.index(0)
            neighbors = [n for n in self.ADJACENCY[b_idx] if n != last_blank]
            if not neighbors:
                neighbors = self.ADJACENCY[b_idx]
            chosen = random.choice(neighbors)
            current_state[b_idx], current_state[chosen] = current_state[chosen], current_state[b_idx]
            last_blank = b_idx

        self.state = tuple(current_state)
        self.initial_state = self.state
        self.move_count = 0
        self.move_history = [self.state]
        return self.state

    @staticmethod
    def count_inversions(state: Tuple[int, ...]) -> int:
        """
        Counts inversions among tiles 1 to 8 (ignoring the blank tile 0).
        An inversion is a pair (a, b) where a > b and a appears before b.
        """
        tiles = [x for x in state if x != 0]
        inversions = 0
        for i in range(len(tiles)):
            for j in range(i + 1, len(tiles)):
                if tiles[i] > tiles[j]:
                    inversions += 1
        return inversions

    @staticmethod
    def is_solvable(state: Tuple[int, ...]) -> bool:
        """
        For a 3x3 grid (odd width), an 8-puzzle state is solvable
        if and only if the number of inversions is EVEN.
        """
        return EightPuzzleGame.count_inversions(state) % 2 == 0

    @staticmethod
    def manhattan_distance(state: Tuple[int, ...]) -> int:
        """
        Computes sum of Manhattan distances of tiles 1..8 from their goal positions.
        Admissible and consistent heuristic for 8-Puzzle.
        """
        total_dist = 0
        for idx, val in enumerate(state):
            if val != 0:
                # Goal index for tile `val` is val - 1
                goal_idx = val - 1
                curr_r, curr_c = idx // 3, idx % 3
                goal_r, goal_c = goal_idx // 3, goal_idx % 3
                total_dist += abs(curr_r - goal_r) + abs(curr_c - goal_c)
        return total_dist

    @staticmethod
    def misplaced_tiles(state: Tuple[int, ...]) -> int:
        """
        Counts number of tiles (1..8) not in their goal position.
        Admissible heuristic for 8-Puzzle.
        """
        count = 0
        for idx, val in enumerate(state):
            if val != 0 and val != GOAL_STATE[idx]:
                count += 1
        return count

    @staticmethod
    def get_successors(state: Tuple[int, ...]) -> List[Tuple[str, Tuple[int, ...], int]]:
        """
        Generates legal successor states from state.
        Returns list of (action_name, next_state, moved_tile_value).
        """
        b_idx = state.index(0)
        b_r, b_c = b_idx // 3, b_idx % 3
        successors = []

        for adj_idx in EightPuzzleGame.ADJACENCY[b_idx]:
            adj_r, adj_c = adj_idx // 3, adj_idx % 3
            dr, dc = adj_r - b_r, adj_c - b_c
            # Name direction based on tile slide
            moved_tile = state[adj_idx]
            
            # Action name
            if dr == -1 and dc == 0:
                act = "SLIDE_DOWN" # tile moves down into blank
            elif dr == 1 and dc == 0:
                act = "SLIDE_UP"
            elif dr == 0 and dc == -1:
                act = "SLIDE_RIGHT"
            else:
                act = "SLIDE_LEFT"

            state_list = list(state)
            state_list[b_idx], state_list[adj_idx] = state_list[adj_idx], state_list[b_idx]
            next_state = tuple(state_list)
            successors.append((act, next_state, moved_tile))

        return successors

    def to_matrix(self) -> List[List[int]]:
        """Return 3x3 2D list of the board."""
        return [list(self.state[i : i + 3]) for i in range(0, 9, 3)]

