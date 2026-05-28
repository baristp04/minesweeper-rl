import numpy as np
import torch
import pytest

from src.env.generator import FullyConvGenerator
from src.agent.fcn_agent import MinesweeperFCN, FCNAgent

def test_generator_dynamic_sizes():
    """Jeneratörün farklı boyutlarda dinamik olarak tahta üretebildiğini test eder."""
    generator = FullyConvGenerator(latent_dim=16, hidden_channels=32)
    
    # 1. Test: 9x9 tahta ve 10 mayın
    board9, log_prob9 = generator.sample_board(rows=9, cols=9, num_mines=10)
    assert board9.shape == (9, 9), "Jeneratör 9x9 matris üretemedi!"
    assert np.count_nonzero(board9 == -1) == 10, "Mayın sayısı 10 olmalı!"
    assert isinstance(log_prob9, torch.Tensor), "Log_prob bir tensör olmalı (REINFORCE için)"

    # 2. Test: 12x16 tahta ve 15 mayın (Boyut Bağımsızlık Kontrolü)
    board12, log_prob12 = generator.sample_board(rows=12, cols=16, num_mines=15)
    assert board12.shape == (12, 16), "Jeneratör 12x16 matris üretemedi!"
    assert np.count_nonzero(board12 == -1) == 15, "Mayın sayısı 15 olmalı!"

def test_fcn_forward_pass():
    """FCN ağının boyut bağımsız (Size-Agnostic) çalıştığını ve doğru tipte çıktı verdiğini test eder."""
    net = MinesweeperFCN(embedding_dim=8, hidden_channels=16, num_layers=2)
    
    # Embedding +2 kaydırması yapıldığı için girdiler 0-10 arası pozitif tam sayı (Long) olmalı
    # 10x10'luk rastgele bir tensor veriyoruz (Batch=1, H=10, W=10)
    obs_10x10 = torch.randint(0, 11, (1, 10, 10), dtype=torch.long)
    
    logits = net(obs_10x10)
    
    # Çıktı matrisi girdi ile BİREBİR aynı boyutta (1, 10, 10) olmalı
    assert logits.shape == (1, 10, 10), "FCN çıktısı girdi boyutuyla eşleşmiyor!"
    # Çıktı bir float tensörü olmalı (Logits)
    assert logits.dtype == torch.float32, "Logits float formatında olmalı!"

def test_agent_action_masking():
    """Ajanın maskeleme (Action Masking) sisteminin açık karelere tıklamasını engellediğini test eder."""
    net = MinesweeperFCN()
    agent = FCNAgent(model=net)

    # 3x3'lük yapay bir oyun tahtası oluşturalım
    # Sadece (0, 2) ve (2, 2) kapalı (-1), geri kalanı açılmış ipuçları (0, 1) olsun
    obs = np.array([
        [0, 1, -1],
        [1, 1,  0],
        [0, 0, -1]
    ], dtype=np.int8)

    valid_actions = [(0, 2), (2, 2)]

    # Ajanı hamle yapmaya zorla
    action = agent.select_action(obs, valid_actions)

    # Ajanın seçtiği hamle KESİNLİKLE valid_actions listesinden biri olmalı
    # Ağ o anki rastgele ağırlıklarıyla açık bir kareye çok puan verse bile
    # action_masking bunun önüne geçmelidir.
    assert action in valid_actions, f"Ajan kural dışı hamle yaptı! Seçilen: {action}"