import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

class FullyConvGenerator(nn.Module):
    """
    Tamamen Evrişimli (Fully Convolutional) Çekişmeli Harita Üreteci.
    Artık 'rows' ve 'cols' parametrelerine init aşamasında ihtiyaç duymaz.
    İstenilen her boyutta dinamik olarak mayın tarlası üretebilir.
    """

    def __init__(self, latent_dim: int = 16, hidden_channels: int = 64):
        super().__init__()
        self.latent_dim = latent_dim

        # Lineer katmanlar (MLP) çöpe atıldı, yerine FCN geldi.
        self.net = nn.Sequential(
            nn.Conv2d(latent_dim, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid()  # Çıktıyı 0.0 (Güvenli) ile 1.0 (Mayın) arasına sıkıştırır
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z (torch.Tensor): (Batch, latent_dim, H, W) boyutunda Uzamsal Gürültü Tensörü
        Returns:
            torch.Tensor: (Batch, H, W) boyutunda Olasılık Haritası
        """
        probs = self.net(z)
        return probs.squeeze(1) # Kanal boyutunu(1) atıp (Batch, H, W) döndürür

    def sample_board(self, rows: int, cols: int, num_mines: int) -> tuple[np.ndarray, torch.Tensor]:
        """
        İstenilen boyutta (rows, cols) tahtayı dinamik olarak üretir.
        """
        # 1. Uzamsal Gürültü (Spatial Noise) Üretimi
        # 1. Uzamsal Gürültü (Spatial Noise) Üretimi
        device = next(self.parameters()).device # Modelin GPU'da mı CPU'da mı olduğunu bul
        z = torch.randn(1, self.latent_dim, rows, cols, device=device)
        
        # Olasılıkları hesapla -> Shape: (rows, cols)
        probs = self.forward(z).squeeze(0)

        # 2. Olasılıkları düzleştir (Flatten)
        flat_probs = probs.flatten()

        # 3. Kategorik Örnekleme (Multinomial)
        # Ağın verdiği olasılık ağırlıklarına göre 'num_mines' kadar mayın seç
        mine_indices = torch.multinomial(flat_probs, num_mines, replacement=False)

        # REINFORCE algoritması için seçilen dizilimin Log Olasılığını hesapla
        log_prob = torch.sum(torch.log(flat_probs[mine_indices]))

        # 4. Ortam (Environment) için NumPy tahtasına dönüştür
        board = np.zeros((rows, cols), dtype=np.int8)
        np_indices = mine_indices.numpy()
        coords = np.unravel_index(np_indices, (rows, cols))
        board[coords] = -1

        return board, log_prob

    @staticmethod
    def update_generator_weights(optimizer: optim.Optimizer, log_prob: torch.Tensor, agent_reward: float,
                                 max_possible_reward: float = 1.0) -> float:
        """
        Jeneratörün ağırlıklarını günceller (Asimetrik Self-Play Mantığı).
        Ajan kaybederse (düşük ödül), jeneratör yüksek ödül alır.
        """
        generator_reward = max_possible_reward - agent_reward
        loss = -log_prob * generator_reward

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], max_norm=0.5)
        optimizer.step()

        return loss.item()