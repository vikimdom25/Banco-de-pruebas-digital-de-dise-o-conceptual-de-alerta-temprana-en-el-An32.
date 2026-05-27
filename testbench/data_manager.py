import os
import h5py
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator
from PyQt6.QtCore import QObject, QTimer

from config import RUTA_HDF5, RUTA_CSV_FALLBACK, DT, FRECUENCIA_HZ
from signals import event_bus

class DataManager(QObject):
    def __init__(self):
        super().__init__()
        self.vuelo_actual_data = []
        self.current_step = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick_simulacion)

        # Conectar a señales globales de control
        event_bus.play_requested.connect(self._play)
        event_bus.pause_requested.connect(self._pause)
        event_bus.seek_requested.connect(self._seek)

    def cargar_vuelo(self):
        """Carga datos desde HDF5 o, en su defecto, CSV con PCHIP."""
        if os.path.exists(RUTA_HDF5):
            print(f"[DataManager] Cargando HDF5 desde: {RUTA_HDF5}")
            self._cargar_desde_hdf5()
        elif os.path.exists(RUTA_CSV_FALLBACK):
            print(f"[DataManager] HDF5 no encontrado. Fallback a CSV: {RUTA_CSV_FALLBACK}")
            self._cargar_desde_csv()
        else:
            event_bus.error_ocurrido.emit("No se encontraron fuentes de datos (HDF5 ni CSV).")
            return

        if self.vuelo_actual_data:
            total_pasos = len(self.vuelo_actual_data)
            self.current_step = 0
            event_bus.vuelo_cargado.emit(total_pasos)
            event_bus.estado_sistema_cambiado.emit(f"Vuelo cargado exitosamente. {total_pasos} pasos.")

    def _cargar_desde_hdf5(self):
        try:
            with h5py.File(RUTA_HDF5, 'r') as hf:
                if 'test' in hf:
                    grupo_base = hf['test']
                elif 'val' in hf:
                    grupo_base = hf['val']
                else:
                    grupo_base = hf['train']

                sim_keys = list(grupo_base.keys())
                if not sim_keys: return

                seleccion = sim_keys[0] # Por ahora auto-seleccionamos el primero
                dataset = grupo_base[seleccion]['features']
                matriz_cruda = np.array(dataset)

                nombres_columnas = [c.decode('utf-8') if isinstance(c, bytes) else c for c in dataset.attrs['feature_names']]

                df_vuelo = pd.DataFrame(matriz_cruda, columns=nombres_columnas)
                self.vuelo_actual_data = df_vuelo.to_dict('records')
        except Exception as e:
            event_bus.error_ocurrido.emit(f"Error HDF5: {e}")

    def _cargar_desde_csv(self):
        try:
            df = pd.read_csv(RUTA_CSV_FALLBACK)

            if 'timestamp_ms' in df.columns:
                # Ordenar y eliminar duplicados temporales que romperían scipy PchipInterpolator
                df = df.sort_values('timestamp_ms').drop_duplicates(subset=['timestamp_ms'])

            print("[DataManager] Aplicando PCHIP al CSV crudo...")
            time_raw = df['timestamp_ms'].values

            # Crear nuevo eje temporal uniforme a 50Hz (20ms)
            time_uniform = np.arange(time_raw[0], time_raw[-1], int(DT * 1000))

            df_interp = pd.DataFrame({'timestamp_ms': time_uniform})

            for col in df.columns:
                if col != 'timestamp_ms':
                    if df[col].dtype == object: continue
                    interpolator = PchipInterpolator(time_raw, df[col].values)
                    df_interp[col] = interpolator(time_uniform)

            df_vuelo = df_interp

            if 'pilot-z-accel-fps_sec' in df_vuelo.columns and 'nlf' not in df_vuelo.columns:
                df_vuelo.rename(columns={'pilot-z-accel-fps_sec': 'nlf'}, inplace=True)

            # Calcular derivadas
            df_vuelo['alpha-deg_dot'] = df_vuelo['alpha-deg'].diff() / DT
            df_vuelo['nlf_dot'] = df_vuelo['nlf'].diff() / DT
            df_vuelo['airspeed-kt_dot'] = df_vuelo['airspeed-kt'].diff() / DT
            df_vuelo.fillna({'alpha-deg_dot': 0.0, 'nlf_dot': 0.0, 'airspeed-kt_dot': 0.0}, inplace=True)

            self.vuelo_actual_data = df_vuelo.to_dict('records')

        except Exception as e:
            event_bus.error_ocurrido.emit(f"Error CSV: {e}")

    def _play(self):
        if not self.vuelo_actual_data: return
        self.timer.start(int(DT * 1000)) # 20 ms

    def _pause(self):
        self.timer.stop()

    def _seek(self, idx: int):
        if 0 <= idx < len(self.vuelo_actual_data):
            self.current_step = idx
            fila = self.vuelo_actual_data[self.current_step].copy()

            # Extraemos los datos anteriores para reconstruir buffer en el ML Worker
            inicio = max(0, self.current_step - 499)
            fila['ventana_salto'] = self.vuelo_actual_data[inicio : self.current_step + 1]
            fila['es_salto'] = True

            event_bus.telemetry_updated.emit(fila)

    def _tick_simulacion(self):
        if self.current_step < len(self.vuelo_actual_data) - 1:
            self.current_step += 1
            self._emit_telemetry()
        else:
            self._pause()
            event_bus.estado_sistema_cambiado.emit("Fin de la simulación.")

    def _emit_telemetry(self):
        if self.vuelo_actual_data:
            fila = self.vuelo_actual_data[self.current_step].copy()
            fila['es_salto'] = False
            event_bus.telemetry_updated.emit(fila)
