import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QComboBox, QSlider, QPushButton, 
                             QLabel, QFrame, QDockWidget, QTabWidget, QStatusBar)
from PyQt6.QtCore import Qt

from signals import event_bus

class EngineeringWorkbench(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("An-32 Engineering Workbench - Early Warning System")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet("background-color: #2b2b2b; color: #ffffff;")

        # Configurar la vista principal con pestañas (Tabs)
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Pestaña 1: Replay & Análisis
        self.tab_replay = QWidget()
        self.tabs.addTab(self.tab_replay, "Replay Station")
        self._init_replay_tab()

        # Pestaña 2: UDP Logger (Futura integración)
        self.tab_logger = QWidget()
        self.tabs.addTab(self.tab_logger, "Live UDP Acquisition")
        # TODO: self._init_logger_tab()

        # Configurar Barra de Estado
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Sistema Inicializado. Esperando datos...")

        # Conectar a señales globales
        event_bus.estado_sistema_cambiado.connect(self.statusBar().showMessage)
        event_bus.error_ocurrido.connect(self._mostrar_error)

    def _init_replay_tab(self):
        layout_principal = QHBoxLayout(self.tab_replay)

        # ==========================================
        # 1. PANEL IZQUIERDO: Controles de Sesión
        # ==========================================
        panel_izquierdo = QFrame()
        panel_izquierdo.setFixedWidth(250)
        layout_izquierdo = QVBoxLayout(panel_izquierdo)
        
        lbl_selector = QLabel("Seleccionar Vuelo:")
        lbl_selector.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        self.combo_vuelos = QComboBox()
        # TODO: Conectar a DataManager para cargar vuelo
        
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.setStyleSheet("background-color: #4CAF50; padding: 10px; font-weight: bold;")
        self.btn_play.clicked.connect(self._toggle_play)
        
        lbl_tiempo = QLabel("Línea de Tiempo:")
        self.slider_tiempo = QSlider(Qt.Orientation.Horizontal)
        self.slider_tiempo.setMinimum(0)
        self.slider_tiempo.valueChanged.connect(self._slider_movido)
        
        layout_izquierdo.addWidget(lbl_selector)
        layout_izquierdo.addWidget(self.combo_vuelos)
        layout_izquierdo.addSpacing(20)
        layout_izquierdo.addWidget(self.btn_play)
        layout_izquierdo.addSpacing(20)
        layout_izquierdo.addWidget(lbl_tiempo)
        layout_izquierdo.addWidget(self.slider_tiempo)
        layout_izquierdo.addStretch()

        # ==========================================
        # 2. PANEL CENTRAL: Visualización y 3D
        # ==========================================
        panel_central = QFrame()
        layout_central = QVBoxLayout(panel_central)
        
        lbl_3d = QLabel("[Placeholder para Vista 3D OpenGL]")
        lbl_3d.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_3d.setStyleSheet("border: 1px dashed #555; background-color: #1e1e1e;")
        
        lbl_grafico = QLabel("[Placeholder para Gráficos Temporales]")
        lbl_grafico.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_grafico.setStyleSheet("border: 1px dashed #555; background-color: #1e1e1e;")
        lbl_grafico.setFixedHeight(250)
        
        layout_central.addWidget(lbl_3d)
        layout_central.addWidget(lbl_grafico)

        # ==========================================
        # 3. PANEL DERECHO: Instrumentos e Inferencia (EICAS)
        # ==========================================
        panel_derecho = QFrame()
        panel_derecho.setFixedWidth(550)
        layout_derecho = QVBoxLayout(panel_derecho)
        
        lbl_sixpack = QLabel("[Placeholder para Panel Six-Pack]")
        lbl_sixpack.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_sixpack.setStyleSheet("border: 1px dashed #555; background-color: #2b2b2b;")
        lbl_sixpack.setFixedHeight(400)
        
        # Marco de Inferencia ML
        marco_inferencia = QFrame()
        marco_inferencia.setStyleSheet("background-color: #383838; border-radius: 5px; padding: 10px;")
        layout_inferencia = QVBoxLayout(marco_inferencia)
        
        lbl_titulo_ml = QLabel("PREDICCIÓN DEL MODELO (ML) - EICAS")
        lbl_titulo_ml.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_titulo_ml.setStyleSheet("font-weight: bold; color: #00BFFF; font-size: 14px;")
        
        self.lbl_salud = QLabel("Índice de Riesgo: --")
        self.lbl_alerta = QLabel("Estado: ESPERANDO DATOS")
        
        layout_inferencia.addWidget(lbl_titulo_ml)
        layout_inferencia.addWidget(self.lbl_salud)
        layout_inferencia.addWidget(self.lbl_alerta)
        
        layout_derecho.addWidget(lbl_sixpack)
        layout_derecho.addWidget(marco_inferencia)
        layout_derecho.addStretch()

        # Ensamblar Layout Principal
        layout_principal.addWidget(panel_izquierdo)
        layout_principal.addWidget(panel_central)
        layout_principal.addWidget(panel_derecho)

        # Conectar actualizaciones del UI a las señales (Placeholders por ahora)
        event_bus.telemetry_updated.connect(self._on_telemetry_updated)
        event_bus.inference_updated.connect(self._on_inference_updated)
        
        self.playing = False

    def _toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            self.btn_play.setText("⏸ Pause")
            self.btn_play.setStyleSheet("background-color: #E74C3C; padding: 10px; font-weight: bold;")
            event_bus.play_requested.emit()
            event_bus.estado_sistema_cambiado.emit("Reproduciendo...")
        else:
            self.btn_play.setText("▶ Play")
            self.btn_play.setStyleSheet("background-color: #4CAF50; padding: 10px; font-weight: bold;")
            event_bus.pause_requested.emit()
            event_bus.estado_sistema_cambiado.emit("Pausado.")

    def _slider_movido(self, valor):
        event_bus.seek_requested.emit(valor)

    def _on_telemetry_updated(self, data: dict):
        # Aquí se actualizará el 3D, gráficos y SixPack delegando a sus respectivos widgets
        pass

    def _on_inference_updated(self, result: dict):
        # Aquí se actualizará el panel de EICAS/ML
        pass
        
    def _mostrar_error(self, msg: str):
        self.statusBar().showMessage(f"ERROR: {msg}")
        self.statusBar().setStyleSheet("color: red; font-weight: bold;")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ventana = EngineeringWorkbench()
    ventana.show()
    sys.exit(app.exec())
