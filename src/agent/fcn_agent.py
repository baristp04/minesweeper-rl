import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core.base_interfaces import BaseAgent

class MinesweeperFCN(nn.Module):
    """
    Boyuttan bağımsız (Fully Convolutional) Mayın Tarlası Sinir Ağı.
    Girdi olarak herhangi bir (H, W) boyutunda matris alır, 
    aynı (H, W) boyutunda her hücre için bir hamle puanı (logit) üretir.
    """
    def __init__(self, embedding_dim: int = 16, hidden_channels: int = 64, num_layers: int = 4):
        super().__init__()
        
        # 1. Embedding Katmanı:
        # Tahtadaki değerler (-2, -1, 0...8) sayısal büyüklük değil, "kategori"dir.
        # Bu yüzden onları 11 sınıflı (class) bir kelime dağarcığı gibi ele alıp vektörlere çeviriyoruz.
        self.embedding = nn.Embedding(num_embeddings=11, embedding_dim=embedding_dim)
        
        # 2. Evrişim (Convolution) Katmanları:
        layers = []
        in_channels = embedding_dim
        
        for _ in range(num_layers):
            # kernel_size=3 ve padding=1 kombinasyonu, matrisin en/boy oranını ASLA bozmaz.
            layers.append(nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = hidden_channels
            
        self.feature_extractor = nn.Sequential(*layers)
        
        # 3. Çıktı (Policy Head):
        # Her bir piksel (hücre) için tek bir puan (1 kanal) üretiriz.
        self.policy_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, H, W) boyutunda oyun tahtası tensörü.
        Returns:
            (Batch, H, W) boyutunda hamle puanları (logits).
        """
        # Veriyi Embedding'den geçir (Batch, H, W, Embedding_Dim)
        x = self.embedding(x)
        
        # PyTorch Conv2d katmanları (Batch, Channels, H, W) formatı ister.
        # Bu yüzden kanal boyutunu öne alıyoruz.
        x = x.permute(0, 3, 1, 2)
        
        # Özellikleri çıkar
        features = self.feature_extractor(x)
        
        # Her hücre için hamle puanı üret (Batch, 1, H, W)
        logits = self.policy_head(features)
        
        # Kanal boyutunu (1) yok et, (Batch, H, W) olarak geri döndür
        return logits.squeeze(1)


class FCNAgent(BaseAgent):
    """
    FCN modelini kullanarak kurallara uygun hamleler seçen akıllı ajan.
    """
    def __init__(self, model: MinesweeperFCN, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device

    def select_action(self, obs: np.ndarray, valid_actions: list[tuple[int, int]]) -> tuple[int, int]:
        if not valid_actions:
            raise ValueError("Yapılacak geçerli hamle kalmadı.")

        # 1. Veri Ön İşleme (Normalization/Shifting)
        # Gözlem matrisindeki değerler: -2 (Patlamış), -1 (Kapalı), 0-8 (İpuçları)
        # Embedding katmanına negatif index veremeyiz. Bu yüzden tüm değerleri +2 kaydırıyoruz.
        # Yeni haritalama: -2 -> 0, -1 -> 1, 0 -> 2 ... 8 -> 10
        obs_shifted = obs.astype(np.int64) + 2
        
        # Tensöre çevir ve Batch boyutu ekle (1, H, W)
        obs_tensor = torch.tensor(obs_shifted, dtype=torch.long, device=self.device).unsqueeze(0)

        # 2. Modelden Geçir (Logits üret)
        self.model.eval() # Çıkarım (inference) modu
        with torch.no_grad():
            logits = self.model(obs_tensor).squeeze(0) # (H, W) boyutuna indir

        # 3. Action Masking (Geçersiz Hamleleri Filtreleme)
        # Model açılmış karelere bile puan vermiş olabilir. 
        # Biz sadece geçerli (kapalı) hücrelerin puanlarını dikkate almalıyız.
        mask = torch.full_like(logits, float('-inf')) # Her şeyi eksi sonsuz yap
        
        for r, c in valid_actions:
            mask[r, c] = logits[r, c] # Sadece geçerli hücrelerin gerçek puanını koru

        # 4. Hamle Seçimi
        # Maskelenmiş matristeki en yüksek puana sahip hücreyi bul (Greedy Action)
        # Düzleştirilmiş (flatten) matristeki en büyük değerin indexini alır
        flat_argmax = torch.argmax(mask).item()
        
        # Düzleştirilmiş indexi (satır, sütun) formatına geri çevir
        rows, cols = obs.shape
        best_r = flat_argmax // cols
        best_c = flat_argmax % cols

        return (best_r, best_c)