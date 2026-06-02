import sys
import os
import torch
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox
from PyQt6.QtCore import Qt

# --- 1. YOL AYARI ---
proje_ana_dizini = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(proje_ana_dizini)

# --- 2. GERÇEK MODÜLLERİ İÇERİ AKTARIYORUZ ---
from src.env.minesweeper_env import MinesweeperEnv

# DİKKAT: Eğer aşağıdaki iki satır "ImportError" verirse, dosyaların içindeki "class" isimlerine
# bakıp FCNAgent veya CaptumXAI isimlerini oradakiyle değiştirmelisin.
# Eski hali:
# from src.agent.fcn_agent import FCNAgent

# Yeni hali:
from src.agent.fcn_agent import FCNAgent, MinesweeperActorCritic
from src.xai.captum_xai import CaptumRiskAnalyzer


class MinesweeperGUI(QMainWindow):
    def __init__(self, env_instance, agent_instance, xai_instance):
        super().__init__()
        # Tüm ana bileşenleri arayüze kaydet
        self.env = env_instance
        self.agent = agent_instance
        self.xai = xai_instance
        
        self.rows = self.env.rows
        self.cols = self.env.cols
        self.buttons = {} 
        self.game_over = False
        
        self.init_ui()
        self.reset_game()

    def init_ui(self):
        self.setWindowTitle("Yapay Zeka Destekli Mayın Tarlası")
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QVBoxLayout()
        central_widget.setLayout(self.main_layout)
        
        # Oyun Izgarası
        self.grid_layout = QGridLayout()
        self.main_layout.addLayout(self.grid_layout)
        
        for r in range(self.rows):
            for c in range(self.cols):
                btn = QPushButton()
                btn.setFixedSize(40, 40)
                btn.clicked.connect(lambda checked, row=r, col=c: self.on_cell_clicked(row, col))
                self.grid_layout.addWidget(btn, r, c)
                self.buttons[(r, c)] = btn

        # Alt Butonlar Düzeni
        self.button_layout = QHBoxLayout()
        
        self.reset_btn = QPushButton("🔄 Yeniden Başlat")
        self.reset_btn.setFixedHeight(40)
        self.reset_btn.clicked.connect(self.reset_game)
        self.button_layout.addWidget(self.reset_btn)

        self.xai_btn = QPushButton("🔥 XAI Risk Haritası")
        self.xai_btn.setFixedHeight(40)
        self.xai_btn.setStyleSheet("background-color: #e67e22; color: white; font-weight: bold;")
        self.xai_btn.clicked.connect(self.show_xai_heatmap)
        self.button_layout.addWidget(self.xai_btn)

        self.ai_btn = QPushButton("🤖 Yapay Zeka Hamle Yapsın")
        self.ai_btn.setFixedHeight(40)
        self.ai_btn.setStyleSheet("background-color: #2980b9; color: white; font-weight: bold;")
        self.ai_btn.clicked.connect(self.ai_play_step)
        self.button_layout.addWidget(self.ai_btn)

        self.main_layout.addLayout(self.button_layout)

    def reset_game(self):
        obs, info = self.env.reset()
        self.game_over = False
        self.update_board_ui(obs)

    def on_cell_clicked(self, row: int, col: int):
        if self.game_over or self.env.observation_board[row, col] != -1:
            return

        obs, reward, terminated, truncated, info = self.env.step((row, col))
        self.update_board_ui(obs)

        if terminated:
            self.game_over = True
            print("Oyun Bitti!")

    # --- GERÇEK YAPAY ZEKA ENTEGRASYONU ---
    def ai_play_step(self):
        if self.game_over:
            return

        # Geçerli hamleleri hesapla
        valid_indices = np.argwhere(self.env.observation_board == -1)
        valid_actions = [(int(r), int(c)) for r, c in valid_indices]
        
        if not valid_actions:
            return

        # Eskiden rastgele yaptığımız seçimi artık GERÇEK YAPAY ZEKA yapıyor!
        row, col = self.agent.select_action(self.env.observation_board, valid_actions)
        print(f"Bot karar verdi: Satır {row}, Sütun {col}")
        
        self.on_cell_clicked(row, col)

    # --- GERÇEK XAI ENTEGRASYONU ---
    def show_xai_heatmap(self):
        if self.game_over:
            return
            
        print("XAI Risk Haritası Hesaplanıyor...")
        # Gerçek XAI modelinden risk skorlarını alıyoruz
        risk_matrix = self.xai.generate_risk_heatmap(self.env.observation_board)
        
        for r in range(self.rows):
            for c in range(self.cols):
                # Sadece kapalı olan kutulara ısı haritası uygula
                if self.env.observation_board[r, c] == -1:
                    risk_score = risk_matrix[r, c]
                    red = int(risk_score * 255)
                    green = int((1.0 - risk_score) * 255)
                    
                    btn = self.buttons[(r, c)]
                    btn.setStyleSheet(
                        f"background-color: rgb({red}, {green}, 0); "
                        f"border: 2px solid #2c3e50; "
                        f"border-radius: 4px;"
                    )

    def update_board_ui(self, obs: np.ndarray):
        number_colors = {
            1: "#0000FF", 2: "#008000", 3: "#FF0000", 4: "#000080",
            5: "#800000", 6: "#008080", 7: "#000000", 8: "#808080"
        }

        for r in range(self.rows):
            for c in range(self.cols):
                val = obs[r, c]
                btn = self.buttons[(r, c)]
                
                if val == -1:
                    btn.setText("")
                    btn.setStyleSheet("background-color: #8fa3ad; border: 2px solid #6b7a83; border-radius: 4px;")
                elif val == -2:
                    btn.setText("💣")
                    btn.setStyleSheet("background-color: #e74c3c; border: 1px solid #c0392b; border-radius: 4px; font-size: 18px;")
                elif val >= 0:
                    text = str(val) if val > 0 else ""
                    btn.setText(text)
                    font_color = number_colors.get(val, "#000000")
                    btn.setStyleSheet(
                        f"background-color: #ecf0f1; color: {font_color}; "
                        f"border: 1px solid #bdc3c7; border-radius: 4px; "
                        f"font-weight: 900; font-size: 16px;"
                    )

# --- 3. ANA BAŞLATICI VE MİMARİ BİRLEŞİM ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    print("Modüller yükleniyor...")
    oyun_ortami = MinesweeperEnv(rows=9, cols=9)
    
    # 1. Beynin (Sinir Ağının) boş iskeletini yaratıyoruz
    yapay_zeka_beyni = MinesweeperActorCritic()
    
    # --- YENİ EKLENEN KISIM: ZEKAYI YÜKLEME ---
    # Modelin tam yolunu belirliyoruz (Ana dizindeki .pt dosyası)
    model_yolu = os.path.join(proje_ana_dizini, "a2c_minesweeper_agent.pt")
    
    # Eğitilmiş ağırlıkları beyne yüklüyoruz 
    # (map_location="cpu" kısmı çok önemlidir, model GPU'da eğitilmiş olsa bile senin bilgisayarında hatasız çalışmasını sağlar)
    print(f"Eğitilmiş model yükleniyor: {model_yolu}")
    yapay_zeka_beyni.load_state_dict(torch.load(model_yolu, map_location=torch.device('cpu')))
    # ------------------------------------------

    # 2. Eğitilmiş ve zeki beyni ajanın içine yerleştiriyoruz
    yapay_zeka = FCNAgent(model=yapay_zeka_beyni) 
    
    # 3. Aynı beyni XAI motoruna veriyoruz (Neden o hamleyi yaptığını görebilmek için)
    isi_haritasi_motoru = CaptumRiskAnalyzer(model=yapay_zeka_beyni) 
    
    print("Arayüz başlatılıyor...")
    window = MinesweeperGUI(
        env_instance=oyun_ortami, 
        agent_instance=yapay_zeka, 
        xai_instance=isi_haritasi_motoru
    )
    window.show()
    sys.exit(app.exec())