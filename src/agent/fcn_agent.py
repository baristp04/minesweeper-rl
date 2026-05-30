import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core.base_interfaces import BaseAgent
from torch.distributions import Categorical

class MinesweeperActorCritic(nn.Module):
    def __init__(self, embedding_dim: int = 16, hidden_channels: int = 64, num_layers: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=11, embedding_dim=embedding_dim)
        
        layers = []
        in_channels = embedding_dim
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = hidden_channels
            
        self.feature_extractor = nn.Sequential(*layers)
        
        # ACTOR HEAD: Hangi kareye tıklamalıyım? (H, W boyutunda logits)
        self.policy_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)

        # CRITIC HEAD: Bu tahtada kazanma ihtimalim/beklenen ödülüm nedir? (Tek bir skaler değer)
        self.value_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), # Tüm tahtayı tek bir piksele sıkıştırır
            nn.Flatten(),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(x)
        x = x.permute(0, 3, 1, 2)
        
        features = self.feature_extractor(x)
        
        logits = self.policy_head(features).squeeze(1)
        value = self.value_head(features).squeeze(1) # (Batch, 1) -> (Batch,)
        
        return logits, value


class FCNAgent(BaseAgent):
    """
    FCN modelini kullanarak kurallara uygun hamleler seçen akıllı ajan.
    """
    def __init__(self, model: MinesweeperActorCritic, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device

    def select_action(self, obs: np.ndarray, valid_actions: list[tuple[int, int]]) -> tuple[int, int]:
        if not valid_actions:
            raise ValueError("Yapılacak geçerli hamle kalmadı.")

        # 1. Veri Ön İşleme
        obs_shifted = obs.astype(np.int64) + 2
        obs_tensor = torch.tensor(obs_shifted, dtype=torch.long, device=self.device).unsqueeze(0)

        # 2. Modelden Geçir (Logits ve Value üretilir, Value çıkarımda önemsizdir)
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(obs_tensor)
            logits = logits.squeeze(0) # (H, W) boyutuna indir

        # 3. Action Masking
        mask = torch.full_like(logits, float('-inf'))
        for r, c in valid_actions:
            mask[r, c] = logits[r, c]

        # 4. En Yüksek Puanlı Hamleyi Seç (Greedy)
        flat_argmax = torch.argmax(mask).item()
        rows, cols = obs.shape
        best_r = flat_argmax // cols
        best_c = flat_argmax % cols

        return (best_r, best_c)

    def get_action_and_log_prob(self, obs: np.ndarray, valid_actions: list[tuple[int, int]]):
        # Normalize and shift observations
        obs_shifted = obs.astype(np.int64) + 2
        obs_tensor = torch.tensor(obs_shifted, dtype=torch.long, device=self.device).unsqueeze(0)

        self.model.train() 
        logits = self.model(obs_tensor).squeeze(0)
        
        # Mask invalid actions with -inf
        mask = torch.full_like(logits, float('-inf'))
        for r, c in valid_actions:
            mask[r, c] = logits[r, c]

        # Flatten the mask to apply Softmax over all cells
        flat_mask = mask.flatten()
        
        # Create a probability distribution over valid actions
        probs = torch.nn.functional.softmax(flat_mask, dim=0)
        
        # Sample an action based on probabilities
        m = Categorical(probs)
        action_idx = m.sample()
        log_prob = m.log_prob(action_idx)

        # Convert flattened index back to 2D coordinates
        rows, cols = obs.shape
        best_r = (action_idx.item() // cols)
        best_c = (action_idx.item() % cols)

        return (best_r, best_c), log_prob

    def get_action_info(self, obs: np.ndarray, valid_actions: list[tuple[int, int]]):
        obs_shifted = obs.astype(np.int64) + 2
        obs_tensor = torch.tensor(obs_shifted, dtype=torch.long, device=self.device).unsqueeze(0)

        self.model.train() 
        logits, value = self.model(obs_tensor)
        logits = logits.squeeze(0)
        
        mask = torch.full_like(logits, float('-inf'))
        for r, c in valid_actions:
            mask[r, c] = logits[r, c]

        flat_mask = mask.flatten()
        probs = torch.nn.functional.softmax(flat_mask, dim=0)
        
        m = Categorical(probs)
        action_idx = m.sample()
        log_prob = m.log_prob(action_idx)

        rows, cols = obs.shape
        best_r = (action_idx.item() // cols)
        best_c = (action_idx.item() % cols)

        return (best_r, best_c), log_prob, value