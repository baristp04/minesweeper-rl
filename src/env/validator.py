import numpy as np
from collections import deque


class SolvabilityValidator:
    """
    Validates if a Minesweeper board can be solved using pure first-order logic.
    This ensures the RL agent is not punished for impossible guessing scenarios.
    """

    @staticmethod
    def is_solvable(internal_board: np.ndarray) -> bool:
        """
        Simulates a perfect logical player.
        Returns True if the board can be cleared without guessing.
        """
        rows, cols = internal_board.shape

        # State tracking matrices for our "simulated player"
        revealed = np.zeros((rows, cols), dtype=bool)
        flagged = np.zeros((rows, cols), dtype=bool)

        # 1. Find a safe starting point
        # A fair game should guarantee at least one safe '0' to start the flood-fill
        zeros = np.argwhere(internal_board == 0)
        if len(zeros) == 0:
            return False  # Inherently requires guessing on the first click

        # Simulate the first click on the first available zero
        start_r, start_c = zeros[0]
        SolvabilityValidator._simulate_click(start_r, start_c, internal_board, revealed)

        # 2. The Logic Loop
        progress = True
        while progress:
            progress = False

            # Scan the board for revealed numbers
            for r in range(rows):
                for c in range(cols):
                    if revealed[r, c] and internal_board[r, c] > 0:
                        cell_value = internal_board[r, c]
                        neighbors = SolvabilityValidator._get_neighbors(r, c, rows, cols)

                        # Categorize neighbors
                        hidden_neighbors = [n for n in neighbors if not revealed[n] and not flagged[n]]
                        flagged_neighbors = sum(1 for n in neighbors if flagged[n])

                        # Rule 1: We found all the mines for this number
                        if len(hidden_neighbors) > 0 and len(hidden_neighbors) == (cell_value - flagged_neighbors):
                            for hr, hc in hidden_neighbors:
                                flagged[hr, hc] = True
                                progress = True  # We learned something, keep looping!

                        # Rule 2: We found all the safe cells for this number
                        elif len(hidden_neighbors) > 0 and flagged_neighbors == cell_value:
                            for hr, hc in hidden_neighbors:
                                SolvabilityValidator._simulate_click(hr, hc, internal_board, revealed)
                                progress = True  # We opened new cells, keep looping!

        # 3. Check Win Condition
        # If the number of revealed cells equals the total number of safe cells, the board is solvable.
        total_safe_cells = np.count_nonzero(internal_board != -1)
        total_revealed = np.count_nonzero(revealed)

        return total_revealed == total_safe_cells

    @staticmethod
    def _get_neighbors(r: int, c: int, rows: int, cols: int) -> list[tuple[int, int]]:
        """Returns valid coordinates for the 8 neighboring cells."""
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    neighbors.append((nr, nc))
        return neighbors

    @staticmethod
    def _simulate_click(start_r: int, start_c: int, internal_board: np.ndarray, revealed: np.ndarray):
        """Simulates clicking a cell and applying the flood-fill if it's a 0."""
        if revealed[start_r, start_c]:
            return

        queue = deque([(start_r, start_c)])
        revealed[start_r, start_c] = True

        while queue:
            r, c = queue.popleft()

            # If we revealed a 0, we must automatically reveal its neighbors
            if internal_board[r, c] == 0:
                for nr, nc in SolvabilityValidator._get_neighbors(r, c, internal_board.shape[0],
                                                                  internal_board.shape[1]):
                    if not revealed[nr, nc]:
                        revealed[nr, nc] = True
                        queue.append((nr, nc))