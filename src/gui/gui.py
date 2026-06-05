import sys
import os
import torch
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QGridLayout, 
                             QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox,
                             QSpinBox, QLabel)
# --- DİKKAT: QTimer EKLENDİ ---
from PyQt6.QtCore import Qt, QTimer 

from PyQt6.QtGui import QIcon
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    return os.path.join(base_path, relative_path)

proje_ana_dizini = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(proje_ana_dizini)

from src.env.minesweeper_env import MinesweeperEnv
from src.agent.fcn_agent import FCNAgent, MinesweeperActorCritic
from src.xai.captum_xai import CaptumRiskAnalyzer 


class MinesweeperGUI(QMainWindow):
    def __init__(self, env_class, agent_instance, xai_instance):
        super().__init__()
        self.EnvClass = env_class
        self.agent = agent_instance
        self.xai = xai_instance
        
        self.rows = 9
        self.cols = 9
        self.env = self.EnvClass(rows=self.rows, cols=self.cols)
        
        self.buttons = {} 
        self.game_over = False
        
        # --- YENİ EKLENEN KISIM: ZAMANLAYICI (TIMER) ---
        self.auto_play_timer = QTimer()
        self.auto_play_timer.timeout.connect(self.ai_play_step)
        
        self.init_ui()
        self.reset_game()

    def init_ui(self):
        self.setWindowTitle("AI - Powered Minesweeper")
        
        self.setWindowIcon(QIcon(resource_path("flag.png")))
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QVBoxLayout()
        central_widget.setLayout(self.main_layout)
        
        # ÜST BAR (BOYUT AYARLARI)
        self.top_layout = QHBoxLayout()
        self.row_spin = QSpinBox()
        self.row_spin.setRange(5, 20) 
        self.row_spin.setValue(self.rows)
        self.col_spin = QSpinBox()
        self.col_spin.setRange(5, 20)
        self.col_spin.setValue(self.cols)
        
        self.apply_size_btn = QPushButton("Boyutu Uygula")
        self.apply_size_btn.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold;")
        self.apply_size_btn.clicked.connect(self.change_board_size)
        
        self.top_layout.addWidget(QLabel("Satır:"))
        self.top_layout.addWidget(self.row_spin)
        self.top_layout.addWidget(QLabel("Sütun:"))
        self.top_layout.addWidget(self.col_spin)
        self.top_layout.addWidget(self.apply_size_btn)
        self.top_layout.addStretch() 
        self.main_layout.addLayout(self.top_layout)
        
        # OYUN IZGARASI
        self.grid_layout = QGridLayout()
        self.grid_layout.setSpacing(0)
        self.main_layout.addLayout(self.grid_layout)
        self.create_grid_ui()

        # ALT BUTONLAR
        self.button_layout = QHBoxLayout()
        
        self.reset_btn = QPushButton("🔄 Yeniden Başlat")
        self.reset_btn.setFixedHeight(40)
        self.reset_btn.clicked.connect(self.reset_game)
        self.button_layout.addWidget(self.reset_btn)

        self.xai_btn = QPushButton("🔥 Risk Haritası")
        self.xai_btn.setFixedHeight(40)
        self.xai_btn.setStyleSheet("background-color: #e67e22; color: white; font-weight: bold;")
        self.xai_btn.clicked.connect(self.show_xai_heatmap)
        self.button_layout.addWidget(self.xai_btn)

        self.ai_btn = QPushButton("🤖 Tek Hamle Yap")
        self.ai_btn.setFixedHeight(40)
        self.ai_btn.setStyleSheet("background-color: #2980b9; color: white; font-weight: bold;")
        self.ai_btn.clicked.connect(self.ai_play_step)
        self.button_layout.addWidget(self.ai_btn)

        # --- YENİ EKLENEN KISIM: OTO OYNA BUTONU ---
        self.auto_btn = QPushButton("🚀 Oto-Oyna")
        self.auto_btn.setFixedHeight(40)
        self.auto_btn.setStyleSheet("background-color: #8e44ad; color: white; font-weight: bold;")
        self.auto_btn.clicked.connect(self.toggle_auto_play)
        self.button_layout.addWidget(self.auto_btn)

        self.main_layout.addLayout(self.button_layout)

    def create_grid_ui(self):
        for r in range(self.rows):
            for c in range(self.cols):
                btn = QPushButton()
                btn.setFixedSize(35, 35)
                btn.clicked.connect(lambda checked, row=r, col=c: self.on_cell_clicked(row, col))
                self.grid_layout.addWidget(btn, r, c)
                self.buttons[(r, c)] = btn

    def clear_grid_ui(self):
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.buttons.clear()

    def change_board_size(self):
        # Boyut değiştirilirken oto-oynamayı durdur
        if self.auto_play_timer.isActive():
            self.toggle_auto_play()
            
        self.rows = self.row_spin.value()
        self.cols = self.col_spin.value()
        self.env = self.EnvClass(rows=self.rows, cols=self.cols)
        self.clear_grid_ui()
        self.create_grid_ui()
        self.reset_game()
        self.adjustSize()

    def reset_game(self):
        # Oyun sıfırlanırken oto-oynamayı durdur
        if self.auto_play_timer.isActive():
            self.toggle_auto_play()
            
        obs, info = self.env.reset()
        self.game_over = False
        self.update_board_ui(obs)

    # --- YENİ EKLENEN KISIM: OTO OYNA KONTROLCÜSÜ ---
    def toggle_auto_play(self):
        if self.auto_play_timer.isActive():
            self.auto_play_timer.stop()
            self.auto_btn.setText("🚀 Oto-Oyna")
            self.auto_btn.setStyleSheet("background-color: #8e44ad; color: white; font-weight: bold;")
        else:
            if self.game_over:
                self.reset_game() # Oyun bittiyse baştan başlatıp oto-oynar
            
            # Her hamle arasında 150 milisaniye (0.15 saniye) bekler
            self.auto_play_timer.start(150) 
            self.auto_btn.setText("⏸️ Durdur")
            self.auto_btn.setStyleSheet("background-color: #c0392b; color: white; font-weight: bold;")

    def on_cell_clicked(self, row: int, col: int):
        if self.game_over or self.env.observation_board[row, col] != -1:
            return

        obs, reward, terminated, truncated, info = self.env.step((row, col))
        self.update_board_ui(obs)

        if terminated:
            self.game_over = True
            
            # Oto-oynatma açıksa durdur
            if self.auto_play_timer.isActive():
                self.toggle_auto_play()
                
            # --- YENİ EKLENEN: KAZANMA VE KAYBETME EKRANLARI ---
            # obs matrisinin içinde -2 (mayın) değeri var mı diye kontrol ediyoruz
            if -2 in obs:
                # Kaybetme durumu (Kırmızı Çarpı ikonlu mesaj kutusu)
                QMessageBox.critical(self, "Game Over", "💣 BOOM! Mayına bastın. GAME OVER!")
            else:
                # Kazanma durumu (Mavi Bilgi ikonlu mesaj kutusu)
                QMessageBox.information(self, "Tebrikler", "🏆 Harika iş! Bütün tarlayı temizledin. YOU WON!")

    def ai_play_step(self):
        if self.game_over:
            if self.auto_play_timer.isActive():
                self.toggle_auto_play()
            return

        valid_indices = np.argwhere(self.env.observation_board == -1)
        valid_actions = [(int(r), int(c)) for r, c in valid_indices]
        
        if not valid_actions:
            if self.auto_play_timer.isActive():
                self.toggle_auto_play()
            return

        row, col = self.agent.select_action(self.env.observation_board, valid_actions)
        self.on_cell_clicked(row, col)

    def show_xai_heatmap(self):
        if self.game_over:
            return
        risk_matrix = self.xai.generate_risk_heatmap(self.env.observation_board)
        for r in range(self.rows):
            for c in range(self.cols):
                if self.env.observation_board[r, c] == -1:
                    risk_score = risk_matrix[r, c]
                    red = int(risk_score * 255)
                    green = int((1.0 - risk_score) * 255)
                    btn = self.buttons[(r, c)]
                    btn.setStyleSheet(
                        f"background-color: rgb({red}, {green}, 0); "
                        f"border: 1px solid #2c3e50; border-radius: 4px;"
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
                    btn.setStyleSheet("background-color: #e74c3c; border: 1px solid #c0392b; border-radius: 4px; font-size: 16px;")
                elif val >= 0:
                    text = str(val) if val > 0 else ""
                    btn.setText(text)
                    font_color = number_colors.get(val, "#000000")
                    btn.setStyleSheet(
                        f"background-color: #ecf0f1; color: {font_color}; "
                        f"border: 1px solid #bdc3c7; border-radius: 4px; font-weight: 900; font-size: 14px;"
                    )

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    ikon_yolu = resource_path("flag.png")
    if os.path.exists(ikon_yolu):
        app.setWindowIcon(QIcon(ikon_yolu))
        
    yapay_zeka_beyni = MinesweeperActorCritic()
    model_yolu = resource_path("a2c_minesweeper_agent.pt")
    
    if os.path.exists(model_yolu):
        yapay_zeka_beyni.load_state_dict(torch.load(model_yolu, map_location=torch.device('cpu')))
        
    yapay_zeka = FCNAgent(model=yapay_zeka_beyni) 
    isi_haritasi_motoru = CaptumRiskAnalyzer(model=yapay_zeka_beyni) 
    
    window = MinesweeperGUI(
        env_class=MinesweeperEnv, 
        agent_instance=yapay_zeka, 
        xai_instance=isi_haritasi_motoru
    )
    window.show()
    sys.exit(app.exec())