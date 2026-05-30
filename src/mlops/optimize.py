import optuna
import torch
import torch.optim as optim
import numpy as np

# Projendeki modülleri içeri aktarıyoruz
from src.env.minesweeper_env import MinesweeperEnv
from src.env.generator import FullyConvGenerator
from src.agent.fcn_agent import MinesweeperActorCritic, FCNAgent
from src.mlops.train import update_actor_critic

def objective(trial):
    """
    Optuna'nın her deneme (trial) için çağıracağı test fonksiyonu.
    Amacımız: Verilen hiperparametrelerle kısa bir eğitim yapıp, ajanın performansını (ödül) döndürmek.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Optuna'nın Deneyeceği Hiperparametreler
    agent_lr = trial.suggest_float("agent_lr", 1e-5, 1e-2, log=True)
    generator_lr = trial.suggest_float("generator_lr", 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.90, 0.999)
    hidden_channels = trial.suggest_categorical("hidden_channels", [32, 64, 128])
    
    # Hızlı test için bölüm sayısını düşük tutuyoruz (örn: 500)
    num_episodes = 500
    rows, cols, num_mines = 9, 9, 10

    # 2. Ortam ve Modelleri Başlat
    env = MinesweeperEnv(rows=rows, cols=cols, num_mines=num_mines)
    
    agent_net = MinesweeperActorCritic(hidden_channels=hidden_channels).to(device)
    agent = FCNAgent(model=agent_net, device=device)
    agent_optimizer = optim.Adam(agent_net.parameters(), lr=agent_lr)
    
    generator = FullyConvGenerator(hidden_channels=hidden_channels).to(device)
    generator_optimizer = optim.Adam(generator.parameters(), lr=generator_lr)

    # Performans ölçümü için son 100 oyunun ödüllerini tutacağız
    recent_rewards = []

    # 3. Kısa Eğitim Döngüsü
    for episode in range(num_episodes):
        custom_board, gen_log_prob = generator.sample_board(rows=rows, cols=cols, num_mines=num_mines)
        obs, _ = env.reset(custom_board=custom_board)
        done = False
        
        log_probs, values, rewards = [], [], []
        episode_reward = 0.0

        while not done:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
                
            action, log_prob, value = agent.get_action_info(obs, valid_actions)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            episode_reward += reward

        # Teorik Maksimum Ödül: (Güvenli Kare Sayısı * 0.1) + Kazanma Bonusu (1.0)
        max_theoretical_reward = ((rows * cols) - num_mines) * 0.1 + 1.0

        # Ağırlıkları güncelle
        update_actor_critic(agent_optimizer, log_probs, values, rewards, gamma=gamma)
        FullyConvGenerator.update_generator_weights(
            generator_optimizer, gen_log_prob, episode_reward, max_possible_reward=max_theoretical_reward
        )

        # Son 100 bölümün ödülünü kaydet
        if episode >= num_episodes - 100:
            recent_rewards.append(episode_reward)

    # 4. Optuna'ya Başarı Puanını Döndür (Amacımız bu değeri maksimize etmek)
    avg_reward = np.mean(recent_rewards) if recent_rewards else -1.0
    return avg_reward

if __name__ == "__main__":
    print("Optuna Hiperparametre Optimizasyonu Başlıyor...")
    
    # Amacımız 'objective' fonksiyonundan dönen ortalama ödülü maksimize etmek (maximize)
    study = optuna.create_study(direction="maximize", study_name="Minesweeper-A2C-Optimization")
    
    # 30 farklı kombinasyon (trial) denemesini istiyoruz
    study.optimize(objective, n_trials=30)

    print("\n--- Optimizasyon Tamamlandı ---")
    print("En İyi Parametreler:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    print(f"En İyi Ortalama Ödül: {study.best_value:.2f}")