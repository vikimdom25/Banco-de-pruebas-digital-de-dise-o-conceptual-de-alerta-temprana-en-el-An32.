import torch 
from modeldriver import MotorInferenciaStall

import sys
import h5py
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import pandas as pd
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QComboBox, QSlider, QPushButton, 
                             QLabel, QFrame, QGridLayout, QProgressBar)
from PyQt6.QtCore import Qt, QTimer

# Importar tus módulos
from panel_sixpack import PanelSixPack
# from panel_3d import Panel3D
# from panel_variables_vs_tiempo import PanelVariables


class VisualizadorStall(QMainWindow):
    def __init__(self, ruta_hdf5, raw):
        super().__init__()
        self.raw = raw
        self.ruta_hdf5 = ruta_hdf5
        self.vuelo_actual_data = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.tick_simulacion)
        self.playing = False
        self.current_step = 0
        self.last_step_processed = -1
        
        # Inicializar UI
        self.init_ui()
        # Cargar lista de vuelos al iniciar
        self.cargar_lista_vuelos()
        
        # ¡ACTIVAMOS EL MOTOR DE INFERENCIA!
        # Reemplaza ruta_scalers con la carpeta donde guardaste tus .pkl
        ruta_modelo_real = r"C:\Users\santi\OneDrive\Documentos\visualizer\EDAs\mejor_modelo_v3_8heads.pth"
        ruta_scalers = r"C:\Users\santi\OneDrive\Documentos\visualizer\EDAs"
        
        # Y asegúrate de estar importando de modeldriver
        from modeldriver import MotorInferenciaStall
        self.motor = MotorInferenciaStall(ruta_modelo_real, ruta_scalers)

    def init_ui(self):
        self.setWindowTitle("Simulador de Análisis de Pérdida Acelerada (Stall)")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet("background-color: #2b2b2b; color: #ffffff;")

        # Widget central
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout_principal = QHBoxLayout(main_widget)

        # ==========================================
        # 1. PANEL IZQUIERDO: Controles
        # ==========================================
        panel_izquierdo = QFrame()
        panel_izquierdo.setFixedWidth(250)
        layout_izquierdo = QVBoxLayout(panel_izquierdo)
        
        lbl_selector = QLabel("Seleccionar Vuelo:")
        lbl_selector.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        self.combo_vuelos = QComboBox()
        self.combo_vuelos.currentIndexChanged.connect(self.cargar_datos_vuelo)
        
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.setStyleSheet("background-color: #4CAF50; padding: 10px; font-weight: bold;")
        self.btn_play.clicked.connect(self.toggle_play)
        
        lbl_tiempo = QLabel("Línea de Tiempo:")
        self.slider_tiempo = QSlider(Qt.Orientation.Horizontal)
        self.slider_tiempo.setMinimum(0)
        self.slider_tiempo.valueChanged.connect(self.slider_movido)
        
        layout_izquierdo.addWidget(lbl_selector)
        layout_izquierdo.addWidget(self.combo_vuelos)
        layout_izquierdo.addSpacing(20)
        layout_izquierdo.addWidget(self.btn_play)
        layout_izquierdo.addSpacing(20)
        layout_izquierdo.addWidget(lbl_tiempo)
        layout_izquierdo.addWidget(self.slider_tiempo)
        layout_izquierdo.addStretch()

        # ==========================================
        # 2. PANEL CENTRAL: 3D y Gráficos
        # ==========================================
        panel_central = QFrame()
        layout_central = QVBoxLayout(panel_central)
        
        # --- 2.1 El Modelo 3D ("Los 3 Palitos") ---
        self.vista_3d = gl.GLViewWidget()
        self.vista_3d.opts['distance'] = 15 # Distancia de la cámara
        self.vista_3d.setBackgroundColor('#1e1e1e')
        
        # Agregamos una cuadrícula de referencia para el suelo
        grid = gl.GLGridItem()
        grid.setSize(x=20, y=20)
        grid.setSpacing(x=2, y=2)
        self.vista_3d.addItem(grid)
        
        # ¡NUESTRO AVIÓN! (Eje X=Rojo, Eje Y=Verde, Eje Z=Azul)
        self.avion_3d = gl.GLAxisItem()
        self.avion_3d.setSize(x=4, y=4, z=4) # Tamaño de los palitos
        self.vista_3d.addItem(self.avion_3d)
        
        # --- 2.2 Gráfico de Altitud Dinámico ---
        pg.setConfigOption('background', '#1e1e1e')
        pg.setConfigOption('foreground', 'd')
        self.grafico_altitud = pg.PlotWidget(title="Perfil de Altitud")
        self.grafico_altitud.setLabel('left', 'Altitud', units='ft')
        self.grafico_altitud.setLabel('bottom', 'Tiempo', units='s')
        self.grafico_altitud.showGrid(x=True, y=True)
        self.grafico_altitud.setFixedHeight(250)
        
        # Creamos la "pluma" (línea) que se irá dibujando
        self.curva_altitud = self.grafico_altitud.plot(pen=pg.mkPen('#00BFFF', width=2))
        
        layout_central.addWidget(self.vista_3d)
        layout_central.addWidget(self.grafico_altitud)

        # ==========================================
        # 3. PANEL DERECHO: Instrumentos e Inferencia
        # ==========================================
        panel_derecho = QFrame()
        panel_derecho.setFixedWidth(550)
        layout_derecho = QVBoxLayout(panel_derecho)
        
        df_dummy = pd.DataFrame([{
            'airspeed-kt': 0.0, 
            'altitude-ft': 0.0, 
            'pitch-deg': 0.0, 
            'roll-deg': 0.0, 
            'side-slip-deg': 0.0, 
            'vertical-speed-fps': 0.0
        }])
        
        self.panel_sixpack = PanelSixPack(df=df_dummy)
        self.panel_sixpack.setFixedHeight(600)
        layout_derecho.addWidget(self.panel_sixpack)

        # --- CEREBRO DEL MODELO (Inferencia) ---
        marco_inferencia = QFrame()
        marco_inferencia.setStyleSheet("background-color: #383838; border-radius: 5px; padding: 10px;")
        layout_inferencia = QVBoxLayout(marco_inferencia)
        
        lbl_titulo_ml = QLabel("PREDICCIÓN DEL MODELO (ML)")
        lbl_titulo_ml.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_titulo_ml.setStyleSheet("font-weight: bold; color: #00BFFF; font-size: 14px;")
        
        # Salida 1: Estado Actual
        self.lbl_estado_actual = QLabel("Estado: NORMAL")
        self.lbl_estado_actual.setStyleSheet("color: #2ECC71; font-weight: bold; font-size: 16px;")
        
        # --- ¡ESTAS SON LAS DOS LÍNEAS QUE FALTABAN! ---
        self.lbl_estado_futuro = QLabel("Predicción Futura: --")
        self.lbl_estado_futuro.setStyleSheet("color: #BDC3C7; font-weight: bold; font-size: 14px;")
        
        # Salida 2: Alerta Predictiva
        self.lbl_prob_futuro = QLabel("Probabilidad futura de Pérdida:")
        self.barra_prob = QProgressBar()
        self.barra_prob.setRange(0, 100)
        self.barra_prob.setValue(0)
        self.barra_prob.setStyleSheet("QProgressBar::chunk { background-color: #E74C3C; }")
        
        self.lbl_t2s = QLabel("Time-to-Stall (T2S): -- s")
        self.lbl_t2s.setStyleSheet("color: #F1C40F; font-size: 16px; font-weight: bold;")
        
        layout_inferencia.addWidget(lbl_titulo_ml)
        layout_inferencia.addWidget(self.lbl_estado_actual)
        layout_inferencia.addWidget(self.lbl_estado_futuro)
        layout_inferencia.addWidget(self.lbl_prob_futuro)
        layout_inferencia.addWidget(self.barra_prob)
        layout_inferencia.addWidget(self.lbl_t2s)
        
        layout_derecho.addWidget(marco_inferencia)
        layout_derecho.addStretch()

        layout_principal.addWidget(panel_izquierdo)
        layout_principal.addWidget(panel_central)
        layout_principal.addWidget(panel_derecho)

    def cargar_lista_vuelos(self):
        import h5py
        try:
            with h5py.File(self.raw, 'r') as hf:
                simulaciones = list(hf['dynamic_states'].keys())
                self.combo_vuelos.clear()
                self.combo_vuelos.addItems(simulaciones)
                print(f"¡Cargados {len(simulaciones)} vuelos exitosamente desde GOLD!")
        except Exception as e:
            print(f"Error explorando el HDF5 GOLD: {e}")

    def cargar_datos_vuelo(self):
        seleccion = self.combo_vuelos.currentText()
        if not seleccion: return
        
        import pandas as pd
        import h5py
        import numpy as np
        
        try:
            ruta_interna = f"dynamic_states/{seleccion}"
            with h5py.File(self.raw, 'r') as hf:
                dataset = hf[ruta_interna]
                matriz_cruda = np.array(dataset)
                
                nombres_columnas = None
                for attr in ['columns', 'feature_names', 'labels']:
                    if attr in dataset.attrs:
                        nombres_columnas = [c.decode('utf-8') if isinstance(c, bytes) else c for c in dataset.attrs[attr]]
                        break

            if nombres_columnas is not None:
                df_vuelo = pd.DataFrame(matriz_cruda, columns=nombres_columnas)
            else:
                columnas_gold = [
                    'timestamp_ms', 'ID_Vuelo', 'throttle', 'flap-pos-norm', 
                    'elevator-pos-norm', 'left-aileron-pos-norm', 'rudder-pos-norm', 
                    'altitude-ft', 'pitch-deg', 'roll-deg', 'alpha-deg', 
                    'side-slip-deg', 'airspeed-kt', 'vertical-speed-fps', 
                    'q_rad_sec', 'p_rad_sec', 'r_rad_sec', 'pilot-z-accel-fps_sec'
                ]
                if matriz_cruda.shape[1] == len(columnas_gold):
                    df_vuelo = pd.DataFrame(matriz_cruda, columns=columnas_gold)
                else:
                    print(f"🚨 ERROR: La matriz tiene {matriz_cruda.shape[1]} columnas, tu lista tiene {len(columnas_gold)}")
                    return

            if 'pilot-z-accel-fps_sec' in df_vuelo.columns and 'nlf' not in df_vuelo.columns:
                df_vuelo.rename(columns={'pilot-z-accel-fps_sec': 'nlf'}, inplace=True)
                
            if 'timestamp_ms' in df_vuelo.columns:
                df_vuelo = df_vuelo.sort_values('timestamp_ms')
            
            dt = 0.02 
            df_vuelo['alpha-deg_dot'] = df_vuelo['alpha-deg'].diff() / dt
            df_vuelo['nlf_dot'] = df_vuelo['nlf'].diff() / dt
            df_vuelo['airspeed-kt_dot'] = df_vuelo['airspeed-kt'].diff() / dt
            df_vuelo.fillna({'alpha-deg_dot': 0.0, 'nlf_dot': 0.0, 'airspeed-kt_dot': 0.0}, inplace=True)
            
            self.vuelo_actual_data = df_vuelo.to_dict('records')
            total_pasos = len(self.vuelo_actual_data)
            
            self.slider_tiempo.setMaximum(total_pasos - 1)
            self.slider_tiempo.setValue(0)
            self.current_step = 0

            # --- NUEVO: Extraer vectores rápidos para la gráfica ---
            self.array_altitud = df_vuelo['altitude-ft'].to_numpy()
            self.array_tiempo = np.arange(total_pasos) * 0.02 # 50Hz
            
            # ¡NUEVO! Limpiar el buffer de memoria del modelo al cambiar de vuelo
            if hasattr(self, 'motor'):
                self.motor.buffer.clear()
            
            print(f"¡Vuelo {seleccion} cargado y listo para simular! Pasos: {total_pasos}")
            
        except Exception as e:
            print(f"Error crítico procesando el vuelo {seleccion}: {e}")

    def toggle_play(self):
        if self.vuelo_actual_data is None: return
        self.playing = not self.playing
        if self.playing:
            self.btn_play.setText("⏸ Pause")
            self.btn_play.setStyleSheet("background-color: #E74C3C; padding: 10px; font-weight: bold;")
            self.timer.start(20)
        else:
            self.btn_play.setText("▶ Play")
            self.btn_play.setStyleSheet("background-color: #4CAF50; padding: 10px; font-weight: bold;")
            self.timer.stop()

    def slider_movido(self, valor):
        self.current_step = valor
        self.actualizar_instrumentos()

    def tick_simulacion(self):
        if self.current_step < self.slider_tiempo.maximum():
            self.current_step += 1
            self.slider_tiempo.setValue(self.current_step)
        else:
            self.toggle_play()

    def actualizar_instrumentos(self):
        if self.panel_sixpack.width() == 0 or self.panel_sixpack.height() == 0: 
            return
            
        if self.vuelo_actual_data is None: return
        
        fila_dict = self.vuelo_actual_data[self.current_step]
        # ==========================================
        # --- ACTUALIZACIÓN VISUAL (Gráfica y 3D) ---
        # ==========================================
        # 1. Dibujar la altitud hasta el segundo actual
        self.curva_altitud.setData(
            self.array_tiempo[:self.current_step + 1], 
            self.array_altitud[:self.current_step + 1]
        )
        
        # 2. Rotar los "3 palitos" (Avión 3D)
        pitch = fila_dict['pitch-deg']
        roll = fila_dict['roll-deg']
        yaw = fila_dict.get('heading-deg', 0) # Si no tienes heading, asume 0
        
        self.avion_3d.resetTransform()
        # En OpenGL clásico: Eje Z es arriba (Azul), Eje X es derecha (Rojo), Eje Y es al frente (Verde)
        self.avion_3d.rotate(yaw, 0, 0, 1)    # Guiñada (sobre eje Z)
        self.avion_3d.rotate(-pitch, 1, 0, 0) # Cabeceo (sobre eje X de las alas)
        self.avion_3d.rotate(roll, 0, 1, 0)   # Alabeo (sobre eje Y del morro)
        # ==========================================

        import pandas as pd
        df_un_instante = pd.DataFrame([fila_dict])
        
        try:
            self.panel_sixpack.df = df_un_instante
            self.panel_sixpack._actualizar_instrumentos(0)
        except Exception as e:
            pass

        # =========================================================
        # --- INFERENCIA EN TIEMPO REAL CON SINCRONIZACIÓN 50HZ ---
        # =========================================================
        if hasattr(self, 'motor'):
            # --- LÓGICA DE SINCRONIZACIÓN DEL BUFFER ---
            if self.current_step == self.last_step_processed + 1:
                # Playback normal: Solo enviamos el paso actual
                ventana_datos = [fila_dict]
                salida = self.motor.procesar_secuencia(ventana_datos, es_salto=False)
            else:
                # ¡SALTO DETECTADO! El usuario movió el slider manualmente.
                # Reconstruimos los últimos 500 pasos instantáneamente
                inicio = max(0, self.current_step - 499)
                ventana_datos = self.vuelo_actual_data[inicio : self.current_step + 1]
                salida = self.motor.procesar_secuencia(ventana_datos, es_salto=True)
                
            self.last_step_processed = self.current_step
            # -------------------------------------------

            clase_act, probs_act, clase_fut, probs_fut, t2s, sigma = salida
            
            if clase_act is not None:
                # 1. Mapeo de Clases 
                nombres_clases = {
                    0: ("VUELO NORMAL", "#2ECC71"),                # Verde
                    1: ("ALERTA: ENTRADA / BUFFETING", "#F1C40F"), # Amarillo
                    2: ("STALL", "#E74C3C"),           # Naranja
                    3: ("DIVE", "#E67E22")     # Rojo
                }
                
                # Actualizar Estado Actual
                nom_act, col_act = nombres_clases.get(clase_act, ("DESCONOCIDO", "#FFFFFF"))
                self.lbl_estado_actual.setText(f"Estado Actual: {nom_act}")
                self.lbl_estado_actual.setStyleSheet(f"color: {col_act}; font-weight: bold; font-size: 16px;")

                # Actualizar Estado Futuro
                nom_fut, col_fut = nombres_clases.get(clase_fut, ("DESCONOCIDO", "#FFFFFF"))
                self.lbl_estado_futuro.setText(f"Predicción (5s): {nom_fut}")
                self.lbl_estado_futuro.setStyleSheet(f"color: {col_fut}; font-weight: bold; font-size: 14px;")

                # 2. Alerta Predictiva (Suma de clases 2 y 3)
                prob_peligro_futuro = (probs_fut[2] + probs_fut[3]) * 100
                self.barra_prob.setValue(int(prob_peligro_futuro))
                
                if prob_peligro_futuro > 75:
                    self.barra_prob.setStyleSheet("QProgressBar::chunk { background-color: #E74C3C; }")
                elif prob_peligro_futuro > 35:
                    self.barra_prob.setStyleSheet("QProgressBar::chunk { background-color: #F39C12; }")
                else:
                    self.barra_prob.setStyleSheet("QProgressBar::chunk { background-color: #2ECC71; }")

                # 3. Actualizar Time-To-Stall (Tobit) con Varianza Sigma
                if clase_act == 0 and prob_peligro_futuro < 20:
                    self.lbl_t2s.setText("Time-to-Stall (T2S): Estable")
                    self.lbl_t2s.setStyleSheet("color: #2ECC71; font-size: 16px; font-weight: bold;")
                else:
                    self.lbl_t2s.setText(f"T2S: {max(0, t2s):.1f} s ± {sigma:.1f} s")
                    self.lbl_t2s.setStyleSheet("color: #F1C40F; font-size: 16px; font-weight: bold;")
                    
            else:
                # Cargando buffer de 10 segundos (500 pasos a 50Hz)
                self.lbl_estado_actual.setText(f"Cargando Buffer... ({len(self.motor.buffer)}/500)")
                self.lbl_estado_actual.setStyleSheet("color: #95A5A6; font-size: 14px;")
                self.lbl_estado_futuro.setText("Predicción (5s): Esperando datos...")
                self.barra_prob.setValue(0)
                self.lbl_t2s.setText("Time-to-Stall (T2S): -- s")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    ruta_dataset_raw = r"C:\Users\santi\OneDrive\Documentos\visualizer\Base_de_datos_GOLD_Ultimate_PCHIP.h5"
    ruta_dataset = r"C:\Users\santi\OneDrive\Documentos\visualizer\EDAs\dataset_Transformer_Final_Escalado.h5"
    
    ventana = VisualizadorStall(ruta_dataset, ruta_dataset_raw)
    ventana.show()
    sys.exit(app.exec())
