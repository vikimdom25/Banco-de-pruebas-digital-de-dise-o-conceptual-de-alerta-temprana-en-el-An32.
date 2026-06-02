import sys
import numpy as np
import pandas as pd
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QComboBox, QSlider, QPushButton, 
                             QLabel, QFrame, QDockWidget, QTabWidget, QStatusBar,
                             QFileDialog)
from PyQt6.QtCore import Qt, QThread, QTimer
from PyQt6.QtNetwork import QUdpSocket, QHostAddress
from PyQt6.QtGui import QAction

from signals import event_bus
from data_manager import DataManager
from modeldriver import ModelWorker

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'Widgets'))
from panel_sixpack import PanelSixPack

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
        self._init_logger_tab()

        self._init_menu_bar()

        # Configurar Barra de Estado
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Sistema Inicializado. Esperando datos...")

        # Conectar a señales globales
        event_bus.estado_sistema_cambiado.connect(self.statusBar().showMessage)
        event_bus.error_ocurrido.connect(self._mostrar_error)

        # Iniciar Backend
        self._init_backend()

    def _init_menu_bar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("Archivo")

        load_action = QAction("Cargar Dataset (HDF5/CSV)...", self)
        load_action.triggered.connect(self._open_file_dialog)
        file_menu.addAction(load_action)

    def _open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar Base de Datos de Vuelo",
            "",
            "Archivos de Datos (*.h5 *.hdf5 *.csv);;HDF5 (*.h5 *.hdf5);;CSV (*.csv)"
        )
        if file_path:
            self.statusBar().showMessage(f"Cargando {file_path}...")
            # Emitir la señal al DataManager
            event_bus.manual_file_selected.emit(file_path)

    def _init_backend(self):
        # DataManager en Hilo Principal (o puede ir a thread)
        self.data_manager = DataManager()

        # ModelWorker en Hilo Separado (QThread)
        self.ml_thread = QThread()
        self.model_worker = ModelWorker()
        self.model_worker.moveToThread(self.ml_thread)
        self.ml_thread.start()

        # Cargar el vuelo al iniciar
        self.data_manager.cargar_vuelo()

    def _init_logger_tab(self):
        layout = QVBoxLayout(self.tab_logger)

        lbl_titulo = QLabel("Adquisición Live UDP (FlightGear)")
        lbl_titulo.setStyleSheet("font-size: 18px; font-weight: bold; color: #4CAF50;")

        self.lbl_estado_udp = QLabel("Estado: Desconectado")
        self.lbl_paquetes_udp = QLabel("Paquetes Recibidos: 0")

        self.btn_iniciar_udp = QPushButton("Iniciar Captura")
        self.btn_iniciar_udp.clicked.connect(self._toggle_udp)

        layout.addWidget(lbl_titulo)
        layout.addWidget(self.lbl_estado_udp)
        layout.addWidget(self.lbl_paquetes_udp)
        layout.addWidget(self.btn_iniciar_udp)
        layout.addStretch()

        self.udp_socket = QUdpSocket(self)
        self.udp_socket.readyRead.connect(self._read_udp_datagrams)
        self.is_logging_udp = False
        self.udp_paquetes = 0

    def _toggle_udp(self):
        if not self.is_logging_udp:
            # Iniciar (ej. 127.0.0.1 : 5500)
            if self.udp_socket.bind(QHostAddress.SpecialAddress.LocalHost, 5500):
                self.is_logging_udp = True
                self.btn_iniciar_udp.setText("Detener Captura")
                self.lbl_estado_udp.setText("Estado: Escuchando en 127.0.0.1:5500")
                self.lbl_estado_udp.setStyleSheet("color: #4CAF50;")
            else:
                self.lbl_estado_udp.setText("Estado: Error al vincular el puerto 5500")
                self.lbl_estado_udp.setStyleSheet("color: red;")
        else:
            self.udp_socket.close()
            self.is_logging_udp = False
            self.btn_iniciar_udp.setText("Iniciar Captura")
            self.lbl_estado_udp.setText("Estado: Desconectado")
            self.lbl_estado_udp.setStyleSheet("color: white;")

    def _read_udp_datagrams(self):
        while self.udp_socket.hasPendingDatagrams():
            datagram, host, port = self.udp_socket.readDatagram(self.udp_socket.pendingDatagramSize())
            self.udp_paquetes += 1
            if self.udp_paquetes % 50 == 0:
                self.lbl_paquetes_udp.setText(f"Paquetes Recibidos: {self.udp_paquetes}")
            # En el futuro: Parsear datagram.data().decode('utf-8') y guardar a CSV

    def _init_replay_tab(self):
        # En vez de un layout estático, usamos un QMainWindow interior para manejar los Docks
        self.replay_window = QMainWindow()
        self.replay_window.setWindowFlags(Qt.WindowType.Widget)

        # Necesitamos un widget central vacío (o el principal 3D) para que los docks se adhieran
        central_widget = QWidget()
        self.replay_window.setCentralWidget(central_widget)

        # El QTabWidget padre necesita alojar la sub-ventana
        layout_principal = QVBoxLayout(self.tab_replay)
        layout_principal.setContentsMargins(0,0,0,0)
        layout_principal.addWidget(self.replay_window)

        # ==========================================
        # 1. DOCK IZQUIERDO: Controles de Sesión
        # ==========================================
        dock_controls = QDockWidget("Session Controls", self.replay_window)
        dock_controls.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        panel_izquierdo = QWidget()
        layout_izquierdo = QVBoxLayout(panel_izquierdo)
        
        lbl_selector = QLabel("Seleccionar Vuelo:")
        lbl_selector.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        self.combo_vuelos = QComboBox()
        self.combo_vuelos.addItem("Vuelo Actual (HDF5/CSV)")
        self.combo_vuelos.setEnabled(False)
        self.combo_vuelos.currentIndexChanged.connect(self._vuelo_seleccionado)
        
        layout_play_speed = QHBoxLayout()
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.setStyleSheet("background-color: #4CAF50; padding: 10px; font-weight: bold;")
        self.btn_play.clicked.connect(self._toggle_play)

        self.combo_speed = QComboBox()
        self.combo_speed.addItems(["0.5x", "1.0x", "2.0x", "5.0x", "MAX"])
        self.combo_speed.setCurrentIndex(1) # Default 1.0x
        self.combo_speed.currentIndexChanged.connect(self._cambiar_velocidad)

        layout_play_speed.addWidget(self.btn_play)
        layout_play_speed.addWidget(self.combo_speed)
        
        lbl_tiempo = QLabel("Línea de Tiempo:")
        self.slider_tiempo = QSlider(Qt.Orientation.Horizontal)
        self.slider_tiempo.setMinimum(0)
        self.slider_tiempo.valueChanged.connect(self._slider_movido)
        
        layout_izquierdo.addWidget(lbl_selector)
        layout_izquierdo.addWidget(self.combo_vuelos)
        layout_izquierdo.addSpacing(20)
        layout_izquierdo.addLayout(layout_play_speed)
        layout_izquierdo.addSpacing(20)
        layout_izquierdo.addWidget(lbl_tiempo)
        layout_izquierdo.addWidget(self.slider_tiempo)
        layout_izquierdo.addStretch()

        dock_controls.setWidget(panel_izquierdo)
        self.replay_window.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock_controls)

        # ==========================================
        # 2. DOCK CENTRAL: Vista 3D
        # ==========================================
        dock_3d = QDockWidget("3D Aircraft View", self.replay_window)
        dock_3d.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        
        self.vista_3d = gl.GLViewWidget()
        self.vista_3d.opts['distance'] = 15
        self.vista_3d.setBackgroundColor('#1e1e1e')
        grid = gl.GLGridItem()
        grid.setSize(x=20, y=20)
        grid.setSpacing(x=2, y=2)
        self.vista_3d.addItem(grid)
        # Avión 3D Mejorado (Wireframe en lugar de solo ejes)
        vertices = np.array([
            [0, 4, 0],    # 0: Nariz
            [-1, 0, 0],   # 1: Raíz ala izq
            [1, 0, 0],    # 2: Raíz ala der
            [-5, -1, 0],  # 3: Punta ala izq
            [5, -1, 0],   # 4: Punta ala der
            [0, -4, 0],   # 5: Cola base
            [0, -4.5, 2], # 6: Cola timón (Vertical)
            [-2, -4.5, 0],# 7: Elevador izq
            [2, -4.5, 0]  # 8: Elevador der
        ])

        # Conexiones para formar la estructura tipo wireframe
        edges = np.array([
            [0, 1], [0, 2], [1, 5], [2, 5], # Fuselaje
            [1, 3], [2, 4],                 # Alas
            [5, 6],                         # Timón vertical
            [5, 7], [5, 8]                  # Elevadores horizontales
        ])

        colors = np.array([[1.0, 0.5, 0.0, 1.0] for _ in range(len(edges))]) # Naranja vibrante
        self.avion_3d = gl.GLLinePlotItem(pos=vertices, color=colors, width=3, antialias=True, mode='lines')
        self.vista_3d.addItem(self.avion_3d)
        
        # Ejes de referencia sutiles para orientación
        self.avion_ejes = gl.GLAxisItem()
        self.avion_ejes.setSize(x=2, y=2, z=2)
        self.vista_3d.addItem(self.avion_ejes)

        dock_3d.setWidget(self.vista_3d)
        # Seteamos el 3D como el widget principal virtual del área dockable central
        self.replay_window.setCentralWidget(dock_3d)

        # ==========================================
        # 3. DOCK INFERIOR: Gráfico de Altitud
        # ==========================================
        dock_graph = QDockWidget("Altitude Profile", self.replay_window)
        dock_graph.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.TopDockWidgetArea)

        pg.setConfigOption('background', '#1e1e1e')
        pg.setConfigOption('foreground', 'd')
        self.grafico_altitud = pg.PlotWidget()
        self.grafico_altitud.setLabel('left', 'Altitud', units='ft')
        self.grafico_altitud.setLabel('bottom', 'Tiempo', units='s')
        self.grafico_altitud.showGrid(x=True, y=True)
        self.grafico_altitud.setFixedHeight(250)
        self.curva_altitud = self.grafico_altitud.plot(pen=pg.mkPen('#00BFFF', width=2))
        
        self.hist_tiempo = []
        self.hist_altitud = []

        dock_graph.setWidget(self.grafico_altitud)
        self.replay_window.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock_graph)

        # ==========================================
        # 4. DOCK DERECHO: Instrumentos e Inferencia (EICAS)
        # ==========================================
        dock_instruments = QDockWidget("EICAS & Instruments", self.replay_window)
        dock_instruments.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)

        panel_derecho = QWidget()
        panel_derecho.setMinimumWidth(350)
        layout_derecho = QVBoxLayout(panel_derecho)
        
        # Instantiate Panel SixPack with dummy data
        df_dummy = pd.DataFrame([{
            'airspeed-kt': 0.0,
            'altitude-ft': 0.0,
            'pitch-deg': 0.0,
            'roll-deg': 0.0,
            'heading-deg': 0.0,
            'vertical-speed-fps': 0.0
        }])
        self.panel_sixpack = PanelSixPack(df=df_dummy)

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
        
        layout_derecho.addWidget(self.panel_sixpack)
        layout_derecho.addWidget(marco_inferencia)
        layout_derecho.addStretch()

        dock_instruments.setWidget(panel_derecho)
        self.replay_window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock_instruments)

        # Conectar actualizaciones del UI a las señales
        event_bus.vuelos_disponibles.connect(self._on_vuelos_disponibles)
        event_bus.vuelo_cargado.connect(self._on_vuelo_cargado)
        event_bus.telemetry_updated.connect(self._on_telemetry_updated)
        event_bus.inference_updated.connect(self._on_inference_updated)
        
        self.playing = False
        self.ignorar_combo = False

        # Throttler para Frame-Skipping: ~25 FPS maximos para la UI (independiente de los 50Hz de datos)
        import time
        self.last_ui_update_time = time.time()
        self.UI_UPDATE_INTERVAL = 1.0 / 25.0

    def _cambiar_velocidad(self):
        text = self.combo_speed.currentText()
        if text == "MAX":
            mult = 999.0
        else:
            mult = float(text.replace("x", ""))
        self.data_manager.set_speed_multiplier(mult)

    def _on_vuelos_disponibles(self, vuelos: list):
        self.ignorar_combo = True
        self.combo_vuelos.clear()
        self.combo_vuelos.addItems(vuelos)
        self.combo_vuelos.setEnabled(True)
        self.ignorar_combo = False

    def _vuelo_seleccionado(self):
        if self.ignorar_combo: return
        vuelo = self.combo_vuelos.currentText()
        if vuelo:
            self.statusBar().showMessage(f"Cargando vuelo: {vuelo}...")
            self.data_manager.seleccionar_vuelo(vuelo)

    def _on_vuelo_cargado(self, total_pasos: int):
        self.slider_tiempo.setMaximum(total_pasos - 1)
        self.slider_tiempo.setValue(0)
        self.hist_tiempo.clear()
        self.hist_altitud.clear()

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
        import time
        es_salto = data.get('es_salto', False)

        # El slider siempre avanza sin importar el throttler
        if not es_salto:
            self.slider_tiempo.blockSignals(True)
            self.slider_tiempo.setValue(self.slider_tiempo.value() + 1)
            self.slider_tiempo.blockSignals(False)

            # Acumulamos en la lista histórica (la data sí fluye a los arrays aunque no dibujemos)
            t_actual = len(self.hist_tiempo) * 0.02
            self.hist_tiempo.append(t_actual)
            self.hist_altitud.append(data.get('altitude-ft', 0))

        now = time.time()
        # Renderizado pesado condicionado por el Throttler (Frame-Skipping)
        # O forzado si es un salto manual para actualizar todo de golpe
        if es_salto or (now - self.last_ui_update_time) >= self.UI_UPDATE_INTERVAL:
            self.last_ui_update_time = now

            # Actualizar Vista 3D
            pitch = data.get('pitch-deg', 0)
            roll = data.get('roll-deg', 0)
            yaw = data.get('heading-deg', 0)

            self.avion_3d.resetTransform()
            self.avion_3d.rotate(yaw, 0, 0, 1)
            self.avion_3d.rotate(-pitch, 1, 0, 0)
            self.avion_3d.rotate(roll, 0, 1, 0)

            self.avion_ejes.resetTransform()
            self.avion_ejes.rotate(yaw, 0, 0, 1)
            self.avion_ejes.rotate(-pitch, 1, 0, 0)
            self.avion_ejes.rotate(roll, 0, 1, 0)

            # Actualizar Gráfica
            if es_salto:
                historia = data.get('historia_completa', [])
                self.hist_tiempo = [i * 0.02 for i in range(len(historia))]
                self.hist_altitud = [f.get('altitude-ft', 0) for f in historia]
                self.curva_altitud.setData(self.hist_tiempo, self.hist_altitud)
            else:
                self.curva_altitud.setData(self.hist_tiempo, self.hist_altitud)

            # Actualizar Panel SixPack
            df_un_instante = pd.DataFrame([data])
            try:
                self.panel_sixpack.df = df_un_instante
                self.panel_sixpack._actualizar_instrumentos(0)
            except Exception as e:
                pass

    def _on_inference_updated(self, result: dict):
        riesgo = result.get('riesgo_salud', 0.0)
        alerta_roja = result.get('alerta_roja_eicas', False)
        alerta_amarilla = result.get('alerta_amarilla_eicas', False)

        self.lbl_salud.setText(f"Índice de Riesgo: {riesgo:.2f}")
        if alerta_roja:
            self.lbl_alerta.setText("Estado: ¡ALERTA STALL/DIVE!")
            self.lbl_alerta.setStyleSheet("color: red; font-weight: bold;")
        elif alerta_amarilla:
            self.lbl_alerta.setText("Estado: PRE-ALERTA (Stall inminente)")
            self.lbl_alerta.setStyleSheet("color: orange; font-weight: bold;")
        else:
            self.lbl_alerta.setText("Estado: NORMAL")
            self.lbl_alerta.setStyleSheet("color: green; font-weight: bold;")

    def _mostrar_error(self, msg: str):
        self.statusBar().showMessage(f"ERROR: {msg}")
        self.statusBar().setStyleSheet("color: red; font-weight: bold;")

    def closeEvent(self, event):
        # Limpieza de hilos al cerrar
        self.ml_thread.quit()
        self.ml_thread.wait()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ventana = EngineeringWorkbench()
    ventana.show()
    sys.exit(app.exec())
