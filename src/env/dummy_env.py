import numpy as np
from typing import Any
from src.core.base_interfaces import BaseEnvironment

class DummyMinesweeperEnv(BaseEnvironment):
    """
    GUI geliştirimi için Gymnasium benzeri rastgele hamle yanıtları üreten kukla ortam.
    -1: Kapalı kutu
     0-8: Çevredeki mayın sayısı
    -2: Patlamış mayın
    """

    def __init__(self, rows: int = 9, cols: int = 9):
        self.rows = rows
        self.cols = cols
        self.board = np.full((self.rows, self.cols), -1, dtype=np.int8)
        self.done = False

    def reset(self) -> tuple[np.ndarray, dict[str, Any]]:
        self.board = np.full((self.rows, self.cols), -1, dtype=np.int8)
        self.done = False
        return self.board.copy(), {"message": "Dummy Env Reset"}

    def step(self, action: tuple[int, int]) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.done:
            return self.board.copy(), 0.0, self.done, False, {"error": "Game over, reset needed."}

        r, c = action
        
        # %10 ihtimalle mayına basma simülasyonu
        if np.random.rand() < 0.10:
            self.board[r, c] = -2  # Mayın kodu
            self.done = True
            reward = -1.0
            info = {"status": "boom"}
        else:
            # Güvenli kare, 0-8 arası rastgele ipucu değeri ata
            self.board[r, c] = np.random.randint(0, 9)
            reward = 0.1
            info = {"status": "safe"}

            # Tüm alanlar açıldıysa kazanma simülasyonu
            if len(self.get_valid_actions()) == 0:
                self.done = True
                reward = 1.0
                info["status"] = "win"

        return self.board.copy(), reward, self.done, False, info

    def get_valid_actions(self) -> list[tuple[int, int]]:
        # Kapalı karelerin (-1) koordinatlarını bul
        valid_indices = np.argwhere(self.board == -1)
        return [(int(r), int(c)) for r, c in valid_indices]