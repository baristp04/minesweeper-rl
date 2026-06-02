import torch
import torch.nn.functional as F
import numpy as np
from captum.attr import LayerIntegratedGradients  # <-- DÜZELTİLDİ
from src.core.base_interfaces import BaseXAI
from src.agent.fcn_agent import MinesweeperActorCritic

class CaptumRiskAnalyzer(BaseXAI):
    """
    RL Modelinin kararlarını açıklayan ve Risk Isı Haritası üreten XAI Modülü.
    Captum kütüphanesini kullanarak LayerIntegratedGradients (Dikkat) haritaları çıkartır.
    """

    def __init__(self, model: MinesweeperActorCritic, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        
        # Captum, modelin doğrudan bir skaler veya 1D tensor döndürmesini bekler.
        class ActorWrapper(torch.nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model
                
            def forward(self, x):
                logits, _ = self.base_model(x)
                # En yüksek güvene sahip hamlenin skorunu 1D tensor olarak döndür (Captum formatı)
                return logits.max().unsqueeze(0)
                
        self.wrapper_model = ActorWrapper(self.model)
        
        # Embedding (Tam sayı girdileri olan) katmanları için en güvenilir metot olan
        # LayerIntegratedGradients kullanıyoruz.
        self.saliency = LayerIntegratedGradients(self.wrapper_model, self.model.embedding)

    def generate_risk_heatmap(self, obs: np.ndarray) -> np.ndarray:
        """
        Actor logits çıktısını kullanarak 0.0 (Güvenli) ile 1.0 (Riskli) 
        arasında bir risk haritası çıkartır. Açık hücreler 0.0 kabul edilir.
        """
        rows, cols = obs.shape
        
        # Veri Ön İşleme
        obs_shifted = obs.astype(np.int64) + 2
        obs_tensor = torch.tensor(obs_shifted, dtype=torch.long, device=self.device).unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(obs_tensor)
            logits = logits.squeeze(0) # (H, W)

        # Geçersiz (zaten açılmış) hamleleri maskele
        mask = torch.full_like(logits, float('-inf'))
        valid_indices = np.argwhere(obs == -1)
        for r, c in valid_indices:
            mask[r, c] = logits[r, c]

        # Logits değerlerini olasılığa çevir (Softmax)
        flat_mask = mask.flatten()
        probs = F.softmax(flat_mask, dim=0).reshape(rows, cols)

        # Risk = 1.0 - Olasılık (Modelin tıklama ihtimali düşükse, risk yüksektir)
        risk_map = 1.0 - probs.cpu().numpy()

        # Görsel tutarlılık için: Zaten açık olan yerlerin (-1 olmayanlar) riskini 0.0 yap
        opened_mask = obs != -1
        risk_map[opened_mask] = 0.0

        # Normalizasyon: Açıkta kalan kapalı hücrelerin riskini [0, 1] aralığına tam oturt
        if not np.all(risk_map[~opened_mask] == 0):
            min_val = risk_map[~opened_mask].min()
            max_val = risk_map[~opened_mask].max()
            if max_val > min_val:
                risk_map[~opened_mask] = (risk_map[~opened_mask] - min_val) / (max_val - min_val)

        return risk_map

    def generate_attention_map(self, obs: np.ndarray) -> np.ndarray:
        """
        Captum LayerIntegratedGradients kullanarak modelin karar verirken
        tahtadaki hangi sayılara odaklandığını (Attention) gösteren matrisi döndürür.
        """
        rows, cols = obs.shape
        obs_shifted = obs.astype(np.int64) + 2
        obs_tensor = torch.tensor(obs_shifted, dtype=torch.long, device=self.device).unsqueeze(0)
        
        # NOT: Tam sayıların türevi alınamayacağı için requires_grad_() kaldırıldı.
        # LayerIntegratedGradients arka planda direkt Embedding ağırlıkları üzerinden türev alır.

        # Captum Attribution Hesaplama
        # Modelin çıktısını alıp onu etkileyen en önemli (dikkat çeken) hücreleri bulur
        attributions = self.saliency.attribute(obs_tensor)
        
        # Attributions boyutu: (1, H, W, Embedding_Dim).
        # Embedding boyutu boyunca mutlak değerlerin toplamını alıyoruz.
        attention_map = attributions.squeeze(0).abs().sum(dim=-1).detach().cpu().numpy()

        # Sadece açılmış sayılara/hücrelere olan dikkati vurgula
        opened_mask = obs != -1
        attention_map = attention_map * opened_mask

        # Normalizasyon [0.0 - 1.0] aralığı
        max_attr = attention_map.max()
        if max_attr > 0:
            attention_map /= max_attr

        return attention_map