import numpy as np
import torch

# Proje modüllerini içe aktarıyoruz
from src.env.minesweeper_env import MinesweeperEnv
from src.agent.fcn_agent import MinesweeperActorCritic, FCNAgent
from src.xai.captum_xai import CaptumRiskAnalyzer

def test_game_loop():
    print("--- Gerçek FCN Ajanı ve Captum XAI Testi Başlıyor ---")
    
    # 1. Cihaz Seçimi (Ekran kartı varsa CUDA kullan, yoksa CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Kullanılan İşlem Birimi: {device.upper()}")

    # 2. Ortamı Başlat (5x5 küçük bir tahta, 3 mayın kullanalım)
    env = MinesweeperEnv(rows=5, cols=5, num_mines=3)
    
    # 3. YZ Modelini Başlat
    model = MinesweeperActorCritic().to(device)
    
    # --- EKSİK OLAN VE EKLENMESİ GEREKEN KISIM ---
    try:
        model.load_state_dict(torch.load("a2c_minesweeper_agent.pt", map_location=device))
        print("✅ 50.000 bölümlük dahi ağırlıklar yüklendi!")
    except FileNotFoundError:
        print("⚠️ Uyarı: Eğitilmiş ağırlıklar bulunamadı! Rastgele modelle devam ediliyor.")
    
    model.eval() # Modeli kesinlikle Çıkarım (Test) moduna almalıyız
    # ---------------------------------------------

    # Ajanı ve XAI modülünü AYNI model referansıyla başlatıyoruz (Dependency Injection)
    agent = FCNAgent(model=model, device=device)
    xai = CaptumRiskAnalyzer(model=model, device=device)

    # 4. Ortamı Sıfırla ve Başlangıç Durumunu Al
    obs, info = env.reset()
    print(f"\nBaşlangıç Tahtası:\n{obs}")
    
    done = False
    step_count = 0

    # 5. Oyun Döngüsü
    while not done:
        step_count += 1
        print(f"\n--- Adım {step_count} ---")
        
        # A. Ajan için geçerli hamleleri al ve hamle seç
        valid_actions = env.get_valid_actions()
        
        # Eğer geçerli hamle kalmadıysa döngüyü kır (Güvenlik önlemi)
        if not valid_actions:
            print("Yapılacak geçerli hamle kalmadı!")
            break
            
        action = agent.select_action(obs, valid_actions)
        print(f"Ajanın Seçtiği Hamle Koordinatı: {action}")
        
        # B. Hamleyi ortama uygula (Environment Step)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # C. XAI'dan risk ve dikkat (attention) ısı haritalarını al
        risk_heatmap = xai.generate_risk_heatmap(obs)
        attention_map = xai.generate_attention_map(obs)
        
        # D. Çıktıları Doğrula ve Ekrana Yazdır
        print(f"Alınan Ödül: {reward}")
        print(f"Oyun Durumu: {info.get('status', 'Bilinmiyor')}")
        
        # Risk matrisinden en yüksek riskli değer
        print(f"Isı Haritası Boyutu: {risk_heatmap.shape} | Tahtadaki Max Risk Skoru: {risk_heatmap.max():.2f}")
        
        # Dikkat (Attention) matrisinde modelin en çok baktığı koordinatı bul
        max_attention_idx = np.unravel_index(np.argmax(attention_map), attention_map.shape)
        clean_idx=(int(max_attention_idx[0]), int(max_attention_idx[1]))
        print(f"Modelin En Çok Odaklandığı Kare (Attention Max): {clean_idx}")
        
        # Eğer oyun bittiyse sebebi yazdır
        if done:
            print(f"\nOyun Bitti! Sebep: {info.get('status', 'Bilinmiyor')}")
            print(f"Son Tahta Durumu:\n{obs}")

if __name__ == "__main__":
    test_game_loop()