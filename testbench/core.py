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

import csv
import time
import os
import datetime
import xml.etree.ElementTree as ET

from config import RUTA_XML_PROTOCOL, DIR_LOGS
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

        view_menu = menubar.addMenu("Ver")

        # Permitimos reabrir paneles cerrados si están en la pestaña actual
        # createPopupMenu genera las acciones para mostrar/ocultar los docks activos
        self.view_menu_action = view_menu.aboutToShow.connect(self._actualizar_menu_vista)
        self.view_menu = view_menu

    def _actualizar_menu_vista(self):
        self.view_menu.clear()

        # Acción personalizada para restaurar la vista por defecto
        reset_action = QAction("Restaurar Vista por Defecto", self)
        reset_action.triggered.connect(self._restaurar_docks)
        self.view_menu.addAction(reset_action)
        self.view_menu.addSeparator()

        if hasattr(self, 'replay_window'):
            # Añade las acciones de los QDockWidgets al menú "Ver"
            self.view_menu.addActions(self.replay_window.createPopupMenu().actions())

    def _restaurar_docks(self):
        """Restaura todos los QDockWidgets a su posición y estado visible original"""
        for dock in self.replay_window.findChildren(QDockWidget):
            dock.setFloating(False)
            dock.setVisible(True)

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
        self.csv_file_handle = None
        self.csv_writer = None
        self.udp_start_time = 0

    def _extraer_encabezados_xml(self):
        try:
            tree = ET.parse(RUTA_XML_PROTOCOL)
            root = tree.getroot()
            headers = []
            for var in root.findall(".//chunk"):
                name = var.find("name")
                if name is not None and name.text:
                    headers.append(name.text)
            return ["timestamp_ms"] + headers
        except Exception as e:
            self._mostrar_error(f"Error leyendo XML: {e}")
            return []

    def _toggle_udp(self):
        if not self.is_logging_udp:
            headers = self._extraer_encabezados_xml()
            if not headers:
                return

            timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            os.makedirs(DIR_LOGS, exist_ok=True)
            self.current_csv_path = os.path.join(DIR_LOGS, f"log_datos_{timestamp_str}.csv")

            try:
                self.csv_file_handle = open(self.current_csv_path, "w", newline="")
                self.csv_writer = csv.writer(self.csv_file_handle)
                self.csv_writer.writerow(headers)
            except Exception as e:
                self._mostrar_error(f"No se pudo crear CSV: {e}")
                return

            # Iniciar escucha UDP en LocalHost:5500
            if self.udp_socket.bind(QHostAddress.SpecialAddress.LocalHost, 5500):
                self.is_logging_udp = True
                self.udp_paquetes = 0
                self.udp_start_time = time.time()
                self.btn_iniciar_udp.setText("Detener Captura")
                self.lbl_estado_udp.setText(f"Estado: Grabando en {self.current_csv_path}")
                self.lbl_estado_udp.setStyleSheet("color: #4CAF50;")
            else:
                self.lbl_estado_udp.setText("Estado: Error al vincular el puerto 5500")
                self.lbl_estado_udp.setStyleSheet("color: red;")
                self.csv_file_handle.close()
        else:
            self.udp_socket.close()
            if self.csv_file_handle:
                self.csv_file_handle.close()

            self.is_logging_udp = False
            self.btn_iniciar_udp.setText("Iniciar Captura")
            self.lbl_estado_udp.setText("Estado: Guardado y Desconectado.")
            self.lbl_estado_udp.setStyleSheet("color: white;")

            # Auto-cargar el archivo capturado en el Replay Station
            event_bus.manual_file_selected.emit(self.current_csv_path)

    def _read_udp_datagrams(self):
        while self.udp_socket.hasPendingDatagrams():
            datagram, host, port = self.udp_socket.readDatagram(self.udp_socket.pendingDatagramSize())
            self.udp_paquetes += 1
            if self.udp_paquetes % 50 == 0:
                self.lbl_paquetes_udp.setText(f"Paquetes Recibidos: {self.udp_paquetes}")

            try:
                valores_str = datagram.data().decode("utf-8").strip().split(",")
                valores = list(map(float, valores_str))
                elapsed_time_ms = int((time.time() - self.udp_start_time) * 1000)
                fila = [elapsed_time_ms] + valores
                self.csv_writer.writerow(fila)
            except Exception as e:
                pass # Ignorar paquetes malformados sin crashear la UI

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

        # Para formar un wireframe conexo en OpenGL usando GLLinePlotItem (mode='lines'),
        # debemos asegurarnos de que la estructura repita los vértices de inicio/fin en segmentos desconectados.
        # Repetiremos vértices para crear trazos explícitos P1->P2, P2->P3, etc.

        # Secuencia contigua para la silueta:
        pos_lines = np.array([
            # Contorno del fuselaje
            vertices[0], vertices[1],
            vertices[1], vertices[5],
            vertices[5], vertices[2],
            vertices[2], vertices[0],
            # Ala izquierda
            vertices[1], vertices[3],
            # Ala derecha
            vertices[2], vertices[4],
            # Elevadores
            vertices[7], vertices[5],
            vertices[5], vertices[8],
            # Timón
            vertices[5], vertices[6]
        ])

        colors = np.array([[1.0, 0.5, 0.0, 1.0] for _ in range(len(pos_lines))]) # Naranja vibrante
        self.avion_3d = gl.GLLinePlotItem(pos=pos_lines, color=colors, width=3, antialias=True, mode='lines')
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

        # ==========================================
        # 5. DOCK IZQUIERDO: FSM Ground Truth vs IA
        # ==========================================
        dock_fsm = QDockWidget("FSM Ground Truth", self.replay_window)
        dock_fsm.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)

        panel_fsm = QWidget()
        panel_fsm.setMinimumWidth(300)
        layout_fsm = QVBoxLayout(panel_fsm)

        lbl_titulo_fsm = QLabel("LABORATORIO DE VALIDACIÓN FSM")
        lbl_titulo_fsm.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_titulo_fsm.setStyleSheet("font-weight: bold; color: #FFFFFF; font-size: 14px;")

        # Ground Truth Física
        marco_fisica = QFrame()
        marco_fisica.setStyleSheet("background-color: #2b2b2b; border-radius: 5px; padding: 10px;")
        layout_fisica = QVBoxLayout(marco_fisica)
        lbl_titulo_fisica = QLabel("FSM Ground Truth (Física)")
        lbl_titulo_fisica.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_estado_fsm = QLabel("Estado: --")
        self.lbl_estado_fsm.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_estado_fsm.setStyleSheet("font-size: 16px; font-weight: bold; color: gray;")
        layout_fisica.addWidget(lbl_titulo_fisica)
        layout_fisica.addWidget(self.lbl_estado_fsm)

        # Predicción IA
        marco_ia = QFrame()
        marco_ia.setStyleSheet("background-color: #2b2b2b; border-radius: 5px; padding: 10px;")
        layout_ia = QVBoxLayout(marco_ia)
        lbl_titulo_ia = QLabel("Predicción Modelo (IA)")
        lbl_titulo_ia.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_estado_ia = QLabel("Estado: --")
        self.lbl_estado_ia.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_estado_ia.setStyleSheet("font-size: 16px; font-weight: bold; color: gray;")
        layout_ia.addWidget(lbl_titulo_ia)
        layout_ia.addWidget(self.lbl_estado_ia)

        layout_fsm.addWidget(lbl_titulo_fsm)
        layout_fsm.addWidget(marco_fisica)
        layout_fsm.addWidget(marco_ia)
        layout_fsm.addStretch()

        dock_fsm.setWidget(panel_fsm)
        self.replay_window.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock_fsm)

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
            historia = data.get('historia_completa', [])
            if len(historia) > 0:
                self.hist_tiempo = [i * 0.02 for i in range(len(historia))]
                self.hist_altitud = [f.get('altitude-ft', 0) for f in historia]

            self.curva_altitud.setData(self.hist_tiempo, self.hist_altitud)

            # Actualizar Panel SixPack
            df_un_instante = pd.DataFrame([data])
            try:
                self.panel_sixpack.df = df_un_instante
                self.panel_sixpack._actualizar_instrumentos(0)
            except Exception as e:
                pass

            # Actualizar FSM Ground Truth
            fsm_state = data.get('FSM_Realtime_State', data.get('FSM_State', 0))
            if fsm_state == 0:
                self.lbl_estado_fsm.setText("Estado: NORMAL")
                self.lbl_estado_fsm.setStyleSheet("font-size: 16px; font-weight: bold; color: green;")
            elif fsm_state == 1:
                self.lbl_estado_fsm.setText("Estado: ADVERTENCIA")
                self.lbl_estado_fsm.setStyleSheet("font-size: 16px; font-weight: bold; color: yellow;")
            elif fsm_state == 2:
                self.lbl_estado_fsm.setText("Estado: PÉRDIDA (Stall)")
                self.lbl_estado_fsm.setStyleSheet("font-size: 16px; font-weight: bold; color: red;")
            elif fsm_state == 3:
                self.lbl_estado_fsm.setText("Estado: PICADO (Dive)")
                self.lbl_estado_fsm.setStyleSheet("font-size: 16px; font-weight: bold; color: purple;")
            elif fsm_state == 4:
                self.lbl_estado_fsm.setText("Estado: IMPACTO")
                self.lbl_estado_fsm.setStyleSheet("font-size: 16px; font-weight: bold; color: darkred;")
            elif fsm_state == 5:
                self.lbl_estado_fsm.setText("Estado: RECUPERACIÓN")
                self.lbl_estado_fsm.setStyleSheet("font-size: 16px; font-weight: bold; color: cyan;")

    def _on_inference_updated(self, result: dict):
        riesgo = result.get('riesgo_salud', 0.0)
        alerta_roja = result.get('alerta_roja_eicas', False)
        alerta_amarilla = result.get('alerta_amarilla_eicas', False)

        self.lbl_salud.setText(f"Índice de Riesgo: {riesgo:.2f}")
        if alerta_roja:
            self.lbl_alerta.setText("Estado: ¡ALERTA STALL/DIVE!")
            self.lbl_alerta.setStyleSheet("color: red; font-weight: bold;")
            self.lbl_estado_ia.setText("Estado: ¡ALERTA STALL/DIVE!")
            self.lbl_estado_ia.setStyleSheet("font-size: 16px; font-weight: bold; color: red;")
        elif alerta_amarilla:
            self.lbl_alerta.setText("Estado: PRE-ALERTA (Stall inminente)")
            self.lbl_alerta.setStyleSheet("color: orange; font-weight: bold;")
            self.lbl_estado_ia.setText("Estado: PRE-ALERTA")
            self.lbl_estado_ia.setStyleSheet("font-size: 16px; font-weight: bold; color: orange;")
        else:
            self.lbl_alerta.setText("Estado: NORMAL")
            self.lbl_alerta.setStyleSheet("color: green; font-weight: bold;")
            self.lbl_estado_ia.setText("Estado: NORMAL")
            self.lbl_estado_ia.setStyleSheet("font-size: 16px; font-weight: bold; color: green;")

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
