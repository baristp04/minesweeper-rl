import abc
from typing import Any
import numpy as np

class BaseEnvironment(abc.ABC):
    """
    Mayın tarlası oyun ortamı için temel sözleşme.
    Gymnasium API standartlarına (step, reset) uygun tasarlanmıştır.
    """

    @abc.abstractmethod
    def reset(self) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Ortamı sıfırlar ve başlangıç durumunu döndürür.
        
        Returns:
            obs (np.ndarray): Tahtanın başlangıç matrisi (ör. gizli kareler -1).
            info (dict): Ekstra bilgileri barındıran sözlük.
        """
        pass

    @abc.abstractmethod
    def step(self, action: tuple[int, int]) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Ajanın hamlesini ortama uygular.
        
        Args:
            action (tuple[int, int]): Tıklanacak hücrenin (satır, sütun) koordinatları.
            
        Returns:
            obs (np.ndarray): Güncel oyun tahtası.
            reward (float): Hamle sonucu alınan ödül/ceza.
            terminated (bool): Oyun kurallar gereği bitti mi? (Kazanma/Mayına basma).
            truncated (bool): Zaman veya adım sınırı aşıldı mı?
            info (dict): Ek bilgiler (ör. ortamın iç durumu, hile haritası).
        """
        pass

    @abc.abstractmethod
    def get_valid_actions(self) -> list[tuple[int, int]]:
        """
        Şu anki durumda tıklanabilecek geçerli (henüz açılmamış) hücreleri döndürür.
        """
        pass


class BaseAgent(abc.ABC):
    """
    PPO/DQN ajanı için temel sözleşme.
    """

    @abc.abstractmethod
    def select_action(self, obs: np.ndarray, valid_actions: list[tuple[int, int]]) -> tuple[int, int]:
        """
        Geçerli durumu (obs) alıp kurallara uygun bir hamle seçer.
        
        Args:
            obs (np.ndarray): Oyun tahtasının mevcut durumu.
            valid_actions (list[tuple[int, int]]): İzin verilen hamleler maskesi.
            
        Returns:
            tuple[int, int]: Seçilen (satır, sütun) koordinatı.
        """
        pass


class BaseXAI(abc.ABC):
    """
    Model kararlarının açıklanabilirliğini (Risk Isı Haritası) sağlayacak sözleşme.
    """

    @abc.abstractmethod
    def generate_risk_heatmap(self, obs: np.ndarray) -> np.ndarray:
        """
        Mevcut tahta durumunu analiz edip her kare için risk skoru üretir.
        
        Args:
            obs (np.ndarray): Oyun tahtası matrisi.
            
        Returns:
            np.ndarray: Orijinal tahta ile aynı boyutta, [0.0, 1.0] aralığında float matris.
                        (0.0: Güvenli, 1.0: Kesin Mayın)
        """
        pass