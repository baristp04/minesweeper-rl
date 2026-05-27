import numpy as np
import random
from src.core.base_interfaces import BaseAgent, BaseXAI

class DummyRandomAgent(BaseAgent):
    """
    Rastgele yasal (açılmamış kareler) hamleler yapan kukla ajan.
    """

    def select_action(self, obs: np.ndarray, valid_actions: list[tuple[int, int]]) -> tuple[int, int]:
        if not valid_actions:
            raise ValueError("Yapılacak geçerli hamle kalmadı.")
        
        # Tamamen rastgele bir geçerli hücre seç
        return random.choice(valid_actions)


class DummyXAI(BaseXAI):
    """
    Arayüz ekibinin renk paletlerini test edebilmesi için 
    [0.0, 1.0] aralığında rastgele risk skorları üreten kukla XAI.
    """

    def generate_risk_heatmap(self, obs: np.ndarray) -> np.ndarray:
        rows, cols = obs.shape
        # Tüm tahta için rastgele risk matrisi oluştur
        heatmap = np.random.rand(rows, cols)
        
        # Mantıksal tutarlılık: Eğer hücre zaten açılmışsa (>=0 veya ==-2), risk 0 olsun.
        # Bu, arayüzde mantıklı görünmesini sağlar.
        opened_mask = obs != -1
        heatmap[opened_mask] = 0.0
        
        return heatmap