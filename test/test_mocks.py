import numpy as np
from src.env.minesweeper_env import MinesweeperEnv
from src.agent.dummy_agent import DummyRandomAgent, DummyXAI

def test_environment_initialization():
    env = MinesweeperEnv(rows=8, cols=8)
    obs, info = env.reset()
    
    # Tahta boyutu doğru mu?
    assert obs.shape == (8, 8)
    # Başlangıçta tüm kareler -1 (kapalı) olmalı
    assert np.all(obs == -1)

def test_valid_actions():
    env = MinesweeperEnv(rows=3, cols=3)
    env.reset()
    valid_actions = env.get_valid_actions()
    
    # 3x3 tahtada başlangıçta 9 geçerli hamle olmalı
    assert len(valid_actions) == 9
    # Format (int, int) şeklinde mi?
    assert isinstance(valid_actions[0], tuple)
    assert len(valid_actions[0]) == 2

def test_xai_heatmap_boundaries():
    env = MinesweeperEnv(rows=5, cols=5)
    xai = DummyXAI()
    obs, _ = env.reset()
    
    heatmap = xai.generate_risk_heatmap(obs)
    
    # Isı haritası [0.0, 1.0] aralığında float olmalı
    assert heatmap.shape == (5, 5)
    assert np.min(heatmap) >= 0.0
    assert np.max(heatmap) <= 1.0

def test_agent_action_is_valid():
    env = MinesweeperEnv(rows=4, cols=4)
    agent = DummyRandomAgent()
    obs, _ = env.reset()
    
    valid_actions = env.get_valid_actions()
    action = agent.select_action(obs, valid_actions)
    
    # Seçilen hamle, geçerli hamleler listesinin içinde mi?
    assert action in valid_actions