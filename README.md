# AI-Powered Minesweeper: An Explainable RL Approach

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![RL](https://img.shields.io/badge/Reinforcement_Learning-A2C-green)
![XAI](https://img.shields.io/badge/Explainable_AI-Captum-orange)

An advanced, production-ready AI agent designed to solve the classic game of Minesweeper using **Reinforcement Learning (Actor-Critic)**. Unlike traditional solvers that rely on hardcoded rules or constraint satisfaction, this project builds a "memoryless" agent capable of learning the universal rules of the game through pure spatial logic and convolutions.

## ✨ Key Architectural Features

* **Size-Agnostic FCN Architecture:** Built completely with Fully Convolutional Networks (FCN) using 3x3 filters. The agent does not memorize board coordinates; it learns local patterns (tile-counting). An agent trained on a 9x9 board can seamlessly perform inference on a 16x16 or 30x16 board without structural changes.
* **Adversarial Self-Play & Mixed Curriculum:** The agent was not trained on static boards. It survived a rigorous "tug-of-war" against an adversarial **Generator Network** designed to create the most difficult mine placements. A 50/50 mixed curriculum (Generator vs. Random Solvable Boards) was used to prevent catastrophic forgetting and adversarial overfitting.
* **Explainable AI (XAI) Integration:** AI should not be a black box. Integrated with **PyTorch Captum**, the model generates real-time:
  * **Risk Heatmaps:** Visualizing the exact probability of a hidden cell containing a mine (Logits -> Softmax).
  * **Attention Maps:** Using `LayerIntegratedGradients`, proving that the AI is mathematically looking at the correct clues (e.g., focusing on a '3' to deduce surrounding mines).
* **Robust MLOps Pipeline:** Hyperparameter optimization achieved via **Optuna** and full training loop tracking via **Weights & Biases (WandB)**.
* **Interactive GUI:** A fully functional GUI built with PyQt6/PySide6 to watch the agent play in real-time, complete with live risk heatmaps and auto-flagging capabilities.

## ⚙️ Installation & Usage

**1. Clone the repository and install dependencies:**
```bash
git clone [https://github.com/yourusername/minesweeper-rl-xai.git](https://github.com/yourusername/minesweeper-rl-xai.git)
cd minesweeper-rl-xai
pip install -r requirements.txt

**2. Play with the GUI & XAI:**
Launch the interactive PyQt6 interface to watch the agent play in real-time, complete with live risk heatmaps.
```bash
python main_gui.py
```

**3. Run the Evaluation Script:**
Watch the agent play fair games and output its statistical diagnostics (Win Rate, Avg Survival Steps).
```bash
python -m src.mlops.evaluate
```

**4. Train from Scratch:**
Start the adversarial mixed curriculum training loop (Optuna + WandB integrated).
```bash
python -m src.mlops.train
```

## 🧠 Core Technologies

* **Deep Learning:** PyTorch, TorchRL
* **XAI:** Captum (`LayerIntegratedGradients`)
* **MLOps:** Weights & Biases (WandB), Optuna, tqdm
* **Environment:** Custom NumPy-based Gym-like Environment
* **UI:** PyQt6 / PySide6

## Developers 

* @baristp04: [https://github.com/baristp04](https://github.com/baristp04)
* @Pherzan: [https://github.com/Pherzan](https://github.com/Pherzan)
* @gokhanisavailable: [https://github.com/gokhanisavailable](https://github.com/gokhanisavailable)
* @Arda-Aras103: [https://github.com/Arda-Aras103](https://github.com/Arda-Aras103)
* @Beyazt43: [https://github.com/Beyazt43](https://github.com/Beyazt43)
