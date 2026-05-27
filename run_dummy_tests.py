from src.env.minesweeper_env import MinesweeperEnv
from src.agent.dummy_agent import DummyRandomAgent, DummyXAI

def test_game_loop():
    print("--- Mayın Tarlası Kukla Sistem Testi Başlıyor ---")
    
    # 1. Modülleri Başlat (5x5 küçük bir tahta kullanalım)
    env = MinesweeperEnv(rows=5, cols=5)
    agent = DummyRandomAgent()
    xai = DummyXAI()

    # 2. Ortamı Sıfırla
    obs, info = env.reset()
    print(f"Başlangıç Tahtası:\n{obs}")
    
    done = False
    step_count = 0

    # 3. Oyun Döngüsü
    while not done:
        step_count += 1
        print(f"\n--- Adım {step_count} ---")
        
        # Ajan için geçerli hamleleri al ve hamle seç
        valid_actions = env.get_valid_actions()
        action = agent.select_action(obs, valid_actions)
        print(f"Ajanın Seçtiği Hamle: {action}")
        
        # Hamleyi ortama uygula
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # XAI'dan risk ısı haritasını al
        heatmap = xai.generate_risk_heatmap(obs)
        
        # Çıktıları Doğrula
        print(f"Alınan Ödül: {reward}")
        print(f"Oyun Durumu: {info['status']}")
        print(f"Isı Haritası Boyutu: {heatmap.shape} | Max Risk: {heatmap.max():.2f}")
        
        # Eğer oyun bittiyse sebebi yazdır
        if done:
            print(f"\nOyun Bitti! Sebep: {info['status']}")
            print(f"Son Tahta Durumu:\n{obs}")

if __name__ == "__main__":
    test_game_loop()