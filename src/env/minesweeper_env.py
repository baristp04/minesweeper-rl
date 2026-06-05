import numpy as np
from typing import Any, Optional
from collections import deque
from typing import Any
from src.env.validator import SolvabilityValidator
from src.core.base_interfaces import BaseEnvironment


class MinesweeperEnv(BaseEnvironment):
    """
    Core environment logic for the Minesweeper RL Agent.
    Focuses on the step mechanism and the BFS flood-fill algorithm.
    """

    def get_valid_actions(self) -> list[tuple[int, int]]:
        """Returns a list of coordinates for all hidden cells."""
        valid_indices = np.argwhere(self.observation_board == -1)
        return [(int(r), int(c)) for r, c in valid_indices]

    def __init__(self, rows: int = 9, cols: int = 9, num_mines=None):
        self.rows = rows
        self.cols = cols
        
        total_cells = self.rows * self.cols

        # --- YENİ EKLENEN DİNAMİK HESAPLAMA ---
        # Eğer dışarıdan özel bir mayın sayısı girilmediyse, tahtanın %15'ini mayın yap
        if num_mines is None:
            # 10x10'da 15 mayın, 20x20'de 60 mayın üretir. En az 1 mayın olmasını garantiler.
            self.num_mines = max(1, int(total_cells * 0.15))
        else:
            self.num_mines = num_mines

        # --- ARKADAŞLARININ YAZDIĞI ORİJİNAL GÜVENLİK KONTROLÜ (Aynen korundu) ---
        # Safety Check: Prevent placing more mines than available cells
        # Cap the mines at total_cells - 1 (leave at least one safe cell)
        self.num_mines = min(self.num_mines, total_cells - 1)

        # internal_board: Keeps track of actual mine locations (-1) and neighbor counts (0-8)
        self.internal_board = np.zeros((self.rows, self.cols), dtype=np.int8)

        # observation_board: What the RL agent/GUI actually sees. (-1 means hidden)
        self.observation_board = np.full((self.rows, self.cols), -1, dtype=np.int8)
        self.done = False

    def reset(self, seed: Optional[int] = None, make_solvable: bool = True, max_attempts: int = 100,
              custom_board: Optional[np.ndarray] = None) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Resets the environment to a new game state.

        Args:
            seed (Optional[int]): Seed for the random number generator to ensure reproducibility.
            make_solvable (bool): If True, rerolls the random board until a logically solvable one is generated.
            max_attempts (int): Maximum rerolls before giving up (prevents infinite loops on impossible densities).
            custom_board (Optional[np.ndarray]): A predefined board (e.g., from the Adversarial Generator).
                                                 If provided, random generation is skipped.

        Returns:
            tuple[np.ndarray, dict]: The initial observation board (all -1s) and an info dictionary.
        """
        # 1. Handle Reproducibility
        if seed is not None:
            np.random.seed(seed)

        self.done = False
        self.observation_board = np.full((self.rows, self.cols), -1, dtype=np.int8)

        attempts = 0
        is_fair_board = False

        # 2. Adversarial Generator Override
        if custom_board is not None:
            self.internal_board = custom_board.copy()

            # Find where the generator placed the mines to calculate neighbors
            mine_coords = np.where(self.internal_board == -1)

            # Calculate Neighbor Counts for the injected board
            for r, c in zip(mine_coords[0], mine_coords[1]):
                r_start, r_end = max(0, r - 1), min(self.rows, r + 2)
                c_start, c_end = max(0, c - 1), min(self.cols, c + 2)

                neighborhood = self.internal_board[r_start:r_end, c_start:c_end]
                self.internal_board[r_start:r_end, c_start:c_end] = np.where(
                    neighborhood != -1,
                    neighborhood + 1,
                    neighborhood
                )

            is_fair_board = SolvabilityValidator.is_solvable(self.internal_board)
            return self.observation_board.copy(), {
                "status": "adversarial_reset",
                "is_solvable": is_fair_board,
                "generation_attempts": 1
            }

        # 3. Random Generation Loop (Default Mode)
        while attempts < max_attempts:
            attempts += 1
            self.internal_board = np.zeros((self.rows, self.cols), dtype=np.int8)

            # Scatter the Mines
            total_cells = self.rows * self.cols
            mine_indices = np.random.choice(total_cells, self.num_mines, replace=False)
            mine_coords = np.unravel_index(mine_indices, (self.rows, self.cols))

            self.internal_board[mine_coords] = -1

            # Calculate Neighbor Counts (Efficient Vectorized Approach)
            for r, c in zip(mine_coords[0], mine_coords[1]):
                r_start, r_end = max(0, r - 1), min(self.rows, r + 2)
                c_start, c_end = max(0, c - 1), min(self.cols, c + 2)

                neighborhood = self.internal_board[r_start:r_end, c_start:c_end]
                self.internal_board[r_start:r_end, c_start:c_end] = np.where(
                    neighborhood != -1,
                    neighborhood + 1,
                    neighborhood
                )

            # Validation Check
            if not make_solvable:
                is_fair_board = SolvabilityValidator.is_solvable(self.internal_board)
                break  # Accept the first board regardless of solvability

            is_fair_board = SolvabilityValidator.is_solvable(self.internal_board)
            if is_fair_board:
                break  # We found a perfectly solvable board!

        # 4. Return State and Metadata
        info = {
            "status": "random_reset_complete",
            "is_solvable": is_fair_board,
            "generation_attempts": attempts
        }

        return self.observation_board.copy(), info


    def step(self, action: tuple[int, int]) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Executes the agent's action and updates the environment.
        """
        if self.done:
            return self.observation_board.copy(), 0.0, self.done, False, {"error": "Game over. Call reset()."}

        r, c = action

        # 1. Invalid Move (clicking an already opened cell)
        if self.observation_board[r, c] != -1:
            return self.observation_board.copy(), -0.5, False, False, {"status": "invalid_move"}

        # 2. Fatal Move (Stepping on a mine)
        if self.internal_board[r, c] == -1:
            self.observation_board[r, c] = -2  # Let's use -2 to indicate exploded mine for the GUI
            self.done = True
            return self.observation_board.copy(), -1.0, self.done, False, {"status": "boom"}

        # 3. Safe Move (Zero or Numbered Cell)
        # We trigger the BFS flood fill. If it's a number (>0), it only opens that single cell.
        # If it's a 0, it opens the neighbors as well.
        self._flood_fill(r, c)

        # 4. Check Win Condition
        # If the number of hidden cells (-1) equals the number of total mines, the agent wins.
        hidden_cells = np.count_nonzero(self.observation_board == -1)
        if hidden_cells == self.num_mines:
            self.done = True
            return self.observation_board.copy(), 1.0, self.done, False, {"status": "win"}

        # Standard reward for a safe, non-winning move
        return self.observation_board.copy(), 0.1, False, False, {"status": "safe"}

    def _flood_fill(self, start_r: int, start_c: int) -> None:
        """
        Breadth-First Search (BFS) to reveal cells.
        If a cell has 0 adjacent mines, it reveals all its neighbors.
        """
        queue = deque([(start_r, start_c)])

        # Keep track of visited cells in this iteration to prevent infinite loops
        visited = set()
        visited.add((start_r, start_c))

        while queue:
            r, c = queue.popleft()

            # Reveal the cell on the agent's observation board
            self.observation_board[r, c] = self.internal_board[r, c]

            # If the current cell is exactly 0, we must explore its 8 neighbors
            if self.internal_board[r, c] == 0:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue  # Skip the cell itself

                        nr, nc = r + dr, c + dc

                        # Boundary check: Ensure we don't go off the board
                        if 0 <= nr < self.rows and 0 <= nc < self.cols:
                            # Only add to queue if it's hidden and hasn't been visited yet
                            if (nr, nc) not in visited and self.observation_board[nr, nc] == -1:
                                visited.add((nr, nc))
                                queue.append((nr, nc))