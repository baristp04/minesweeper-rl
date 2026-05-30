import torch
import torch.optim as optim
import torch.nn.functional as F
import wandb
import random
import numpy as np

# Proje modüllerini içe aktar
from src.env.minesweeper_env import MinesweeperEnv
from src.env.generator import FullyConvGenerator
from src.agent.fcn_agent import MinesweeperActorCritic, FCNAgent

def update_actor_critic(optimizer, log_probs, values, rewards, gamma=0.99):
    """Updates the Actor-Critic agent weights at the end of an episode."""
    if not log_probs:
        return 0.0

    returns = []
    R = 0
    # Calculate the discounted sum of rewards
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
        
    returns = torch.tensor(returns, dtype=torch.float32, device=values[0].device)
    
    # Normalize returns for training stability
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

    policy_losses = []
    value_losses = []
    
    for log_prob, value, R in zip(log_probs, values, returns):
        # Advantage: Was the action better or worse than expected?
        advantage = R - value.item()
        
        policy_losses.append(-log_prob * advantage)
        value_losses.append(F.mse_loss(value, torch.tensor([R], device=value.device)))
        
    # Combine losses, backpropagate, and step
    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss.backward()
    
    # Gradient clipping prevents exploding gradients in RL
    torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], max_norm=0.5)
    optimizer.step()
    
    return loss.item()

def train_adversarial_loop(config):
    # 1. Initialize Tracking & Device
    wandb.init(project="minesweeper-rl-a2c", config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training on {device}...")

    # 2. Setup Environment & Models
    env = MinesweeperEnv(rows=config["rows"], cols=config["cols"], num_mines=config["num_mines"])
    
    # Agent (Actor-Critic)
    agent_net = MinesweeperActorCritic(hidden_channels=config["hidden_channels"]).to(device)
    agent = FCNAgent(model=agent_net, device=device) 
    agent_optimizer = optim.Adam(agent_net.parameters(), lr=config["agent_lr"])
    
    # Generator (Adversarial)
    generator = FullyConvGenerator(hidden_channels=config["hidden_channels"]).to(device)
    generator_optimizer = optim.Adam(generator.parameters(), lr=config["generator_lr"])

    # 3. Main Training Loop
    for episode in range(config["num_episodes"]):
        
        # --- YENİLİK 1: KARIŞIK MÜFREDAT (%50 Jeneratör, %50 Rastgele) ---
        use_generator = random.random() < 0.5
        
        if use_generator:
            custom_board, gen_log_prob = generator.sample_board(
                rows=config["rows"], cols=config["cols"], num_mines=config["num_mines"]
            )
            obs, info = env.reset(custom_board=custom_board)
        else:
            # Jeneratörsüz, %100 adil ve çözülebilir rastgele tahta
            obs, info = env.reset(make_solvable=True)
            gen_log_prob = None
            
        done = False
        
        # Agent's trajectory memory
        log_probs = []
        values = []
        rewards = []
        episode_reward = 0.0
        step_count = 0

        # --- B. Agent Phase ---
        while not done:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
                
            # --- YENİLİK 2: İLK HAMLE KORUMASI (Teacher Forcing) ---
            if step_count == 0:
                # Tahtadaki gizli '0' (güvenli) hücreleri bul
                safe_zeros = np.argwhere(env.internal_board == 0)
                if len(safe_zeros) > 0:
                    # Ajan adına rastgele güvenli bir sıfıra tıkla (Oyunun açılmasını sağla)
                    forced_action = tuple(safe_zeros[np.random.choice(len(safe_zeros))])
                    obs, reward, terminated, truncated, step_info = env.step(forced_action)
                    done = terminated or truncated
                    step_count += 1
                    continue # Bu hamleyi eğitime (log_probs) dahil etme, döngüye devam et!

            # Normal Ajan Hamlesi (İlk hamle sonrası kendi iradesiyle oynar)
            action, log_prob, value = agent.get_action_info(obs, valid_actions)
            
            # Execute action
            obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            
            # Store trajectory (Sadece ajanın KENDİ seçtiği hamleler kaydedilir)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            
            episode_reward += reward
            step_count += 1

        # --- C. Learning Phase ---
        # 1. Update Agent (Learning from the trajectory)
        agent_loss = update_actor_critic(
            optimizer=agent_optimizer, 
            log_probs=log_probs, 
            values=values, 
            rewards=rewards,
            gamma=config["gamma"]
        )

        # 2. Update Generator (Sadece bu bölüm jeneratör kullanıldıysa eğitilir)
        gen_loss = 0.0
        if use_generator and gen_log_prob is not None:
            max_theoretical_reward = ((config["rows"] * config["cols"]) - config["num_mines"]) * 0.1 + 1.0
            gen_loss = FullyConvGenerator.update_generator_weights(
                optimizer=generator_optimizer,
                log_prob=gen_log_prob,
                agent_reward=episode_reward,
                max_possible_reward=max_theoretical_reward 
            )

        # --- D. Logging ---
        wandb.log({
            "episode": episode,
            "agent_reward": episode_reward,
            "agent_steps": step_count,
            "agent_loss": agent_loss,
            "generator_loss": gen_loss,
            "win": int(step_info.get("status") == "win"),
            "curriculum_mode": 1 if use_generator else 0 # Hangi modda oynadığını takip edelim
        })

        if episode % 100 == 0:
            mode_str = "GEN" if use_generator else "RND"
            print(f"Episode: {episode} [{mode_str}] | Reward: {episode_reward:.2f} | Steps: {step_count} | Status: {step_info.get('status')}")

    # Save the trained weights
    torch.save(agent_net.state_dict(), "a2c_minesweeper_agent.pt")
    torch.save(generator.state_dict(), "adversarial_generator.pt")
    wandb.finish()
    print("Training complete. Models saved.")

if __name__ == "__main__":
    hyperparameters = {
        "rows": 9,
        "cols": 9,
        "num_mines": 10,
        "num_episodes": 50000,          # Uzun eğitim maratonu başlıyor
        "agent_lr": 0.0009015308182787209,
        "generator_lr": 0.0003817892790662294,
        "gamma": 0.934204053247915,
        "hidden_channels": 64
    }
    train_adversarial_loop(hyperparameters)