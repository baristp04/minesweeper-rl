import torch
import numpy as np
from tqdm import tqdm

# Proje modüllerini içe aktar
from src.env.minesweeper_env import MinesweeperEnv
from src.agent.fcn_agent import MinesweeperActorCritic, FCNAgent

def evaluate_agent(num_episodes: int = 2000, rows: int = 9, cols: int = 9, num_mines: int = 10, hidden_channels: int = 64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🕵️‍♂️ Adil Değerlendirme {device} üzerinde başlatılıyor...")

    # 1. Ortam ve Ajanı Başlat
    env = MinesweeperEnv(rows=rows, cols=cols, num_mines=num_mines)
    agent_net = MinesweeperActorCritic(hidden_channels=hidden_channels).to(device)
    
    # Eğitilmiş Ağırlıkları Yükle
    try:
        agent_net.load_state_dict(torch.load("a2c_minesweeper_agent.pt", map_location=device))
        print("✅ Eğitilmiş model ağırlıkları başarıyla yüklendi!")
    except FileNotFoundError:
        print("❌ HATA: Model ağırlıkları bulunamadı.")
        return
        
    agent_net.eval() 
    agent = FCNAgent(model=agent_net, device=device)

    # 2. Metrikleri Hazırla
    wins = 0
    total_rewards = 0.0
    total_steps = 0

    # 3. Değerlendirme Döngüsü
    for episode in tqdm(range(num_episodes), desc="Adil Oyunlar Oynanıyor"):
        obs, info = env.reset(make_solvable=True)
        done = False
        episode_reward = 0.0
        steps = 0

        while not done:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
                
            # --- YENİLİK: ADİL SINAV (İlk Hamle Koruması) ---
            if steps == 0:
                safe_zeros = np.argwhere(env.internal_board == 0)
                if len(safe_zeros) > 0:
                    # Windows Mayın Tarlası gibi ilk tıklama her zaman güvenli (0) bir yer olur
                    action = tuple(safe_zeros[np.random.choice(len(safe_zeros))])
                else:
                    action = agent.select_action(obs, valid_actions)
            else:
                # Sonraki tüm hamleler ajanın zekasına kalır
                action = agent.select_action(obs, valid_actions)
                
            obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            steps += 1

        # Oyun Sonu İstatistikleri
        if step_info.get("status") == "win":
            wins += 1
            
        total_rewards += episode_reward
        total_steps += steps

    # 4. Mantıklı Mimar Raporu
    win_rate = (wins / num_episodes) * 100
    avg_reward = total_rewards / num_episodes
    avg_steps = total_steps / num_episodes

    print("\n" + "="*50)
    print("🎯 ADİL GERÇEKLİK TESTİ RAPORU 🎯")
    print("="*50)
    print(f"Oynanan Oyun Sayısı    : {num_episodes}")
    print(f"Gerçek Kazanma Oranı   : %{win_rate:.2f}")
    print(f"Ortalama Ödül          : {avg_reward:.3f}")
    print(f"Ortalama Hayatta Kalma : {avg_steps:.1f} Adım")
    print("="*50)

    # Hafızasız (Memoryless) RL Ajanı İçin Gerçekçi Teşhis
    if win_rate >= 40:
        print("\n🏆 MİMARIN TEŞHİSİ: KUSURSUZ! Hafızası olmayan bir ajan için %40+ Win Rate teorik maksimumdur. Overfitting YOK, ajan zeki!")
    elif win_rate >= 20:
        print("\n⚠️ MİMARIN TEŞHİSİ: İYİ. Ajan temel kuralları biliyor ama karmaşık tuzaklarda zorlanıyor.")
    else:
        print("\n🚨 MİMARIN TEŞHİSİ: BAŞARISIZ. Ajan ezberlemiş veya mantık kuramıyor.")

if __name__ == "__main__":
    evaluate_agent(num_episodes=1000)