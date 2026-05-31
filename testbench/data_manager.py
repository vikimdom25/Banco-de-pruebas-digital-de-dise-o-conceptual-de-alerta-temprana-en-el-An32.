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
        self.last_ruta_hdf5 = None
        self.speed_multiplier = 1.0

        # Conectar a señales globales de control
        event_bus.play_requested.connect(self._play)
        event_bus.pause_requested.connect(self._pause)
        event_bus.seek_requested.connect(self._seek)
        event_bus.manual_file_selected.connect(self.cargar_vuelo_manual)

    def cargar_vuelo_manual(self, ruta_archivo: str):
        """Carga datos desde un archivo específico seleccionado por el usuario."""
        if ruta_archivo.endswith('.h5') or ruta_archivo.endswith('.hdf5'):
            print(f"[DataManager] Carga manual HDF5: {ruta_archivo}")
            self._cargar_desde_hdf5(ruta_archivo)
        elif ruta_archivo.endswith('.csv'):
            print(f"[DataManager] Carga manual CSV: {ruta_archivo}")
            self._cargar_desde_csv(ruta_archivo)
        else:
            event_bus.error_ocurrido.emit("Formato de archivo no soportado.")
            return

        self._finalizar_carga()

    def cargar_vuelo(self):
        """Carga automática: datos desde HDF5 por defecto o CSV fallback."""
        if os.path.exists(RUTA_HDF5):
            print(f"[DataManager] Cargando HDF5 por defecto: {RUTA_HDF5}")
            self._cargar_desde_hdf5(RUTA_HDF5)
        elif os.path.exists(RUTA_CSV_FALLBACK):
            print(f"[DataManager] HDF5 no encontrado. Fallback a CSV: {RUTA_CSV_FALLBACK}")
            self._cargar_desde_csv(RUTA_CSV_FALLBACK)
        else:
            event_bus.error_ocurrido.emit("No se encontraron fuentes de datos (HDF5 ni CSV). Use Archivo -> Cargar Dataset para cargar manualmente.")
            return

        self._finalizar_carga()

    def _finalizar_carga(self):
        if self.vuelo_actual_data:
            total_pasos = len(self.vuelo_actual_data)
            self.current_step = 0
            event_bus.vuelo_cargado.emit(total_pasos)
            event_bus.estado_sistema_cambiado.emit(f"Vuelo cargado exitosamente. {total_pasos} pasos.")

    def seleccionar_vuelo(self, flight_key: str):
        """Carga un vuelo especifico si el HDF5 previamente cargado tiene varios."""
        if not self.last_ruta_hdf5: return
        try:
            with h5py.File(self.last_ruta_hdf5, 'r') as hf:
                grupo_base = self._get_hdf5_group(hf)
                if grupo_base is None or flight_key not in grupo_base: return
                self._extraer_matriz_hdf5(grupo_base, flight_key)
            self._finalizar_carga()
        except Exception as e:
            event_bus.error_ocurrido.emit(f"Error cargando el vuelo {flight_key}: {e}")

    def _get_hdf5_group(self, hf):
        if 'test' in hf: return hf['test']
        if 'val' in hf: return hf['val']
        if 'train' in hf: return hf['train']
        if 'dynamic_states' in hf: return hf['dynamic_states']
        return hf

    def _cargar_desde_hdf5(self, ruta: str):
        try:
            self.last_ruta_hdf5 = ruta
            with h5py.File(ruta, 'r') as hf:
                grupo_base = self._get_hdf5_group(hf)

                sim_keys = list(grupo_base.keys())
                if not sim_keys:
                    event_bus.error_ocurrido.emit("El archivo HDF5 no contiene grupos válidos.")
                    return

                # Emitimos la lista de vuelos a la interfaz
                event_bus.vuelos_disponibles.emit(sim_keys)

                # Por defecto cargamos el primero
                seleccion = sim_keys[0]
                self._extraer_matriz_hdf5(grupo_base, seleccion)
        except Exception as e:
            import traceback
            traceback.print_exc()
            event_bus.error_ocurrido.emit(f"Error HDF5: {e}")

    def _extraer_matriz_hdf5(self, grupo_base, seleccion):
        # Dependiendo de la estructura, extraer matriz
        node = grupo_base[seleccion]
        if isinstance(node, h5py.Group) and 'features' in node:
            # Estructura procesada
            dataset = node['features']
            matriz_cruda = np.array(dataset)
            nombres_columnas = [c.decode('utf-8') if isinstance(c, bytes) else c for c in dataset.attrs.get('feature_names', [])]
        else:
            # Estructura cruda o plana
            dataset = node
            matriz_cruda = np.array(dataset)
            # Intentar leer los nombres de los features desde varios attrs comunes
            nombres_columnas = None
            for attr in ['columns', 'feature_names', 'labels']:
                if attr in dataset.attrs:
                    nombres_columnas = [c.decode('utf-8') if isinstance(c, bytes) else c for c in dataset.attrs[attr]]
                    break
            if nombres_columnas is None:
                # Fallback a los nombres gold si no tiene metadata
                nombres_columnas = [
                    'timestamp_ms', 'ID_Vuelo', 'throttle', 'flap-pos-norm',
                    'elevator-pos-norm', 'left-aileron-pos-norm', 'rudder-pos-norm',
                    'altitude-ft', 'pitch-deg', 'roll-deg', 'alpha-deg',
                    'side-slip-deg', 'airspeed-kt', 'vertical-speed-fps',
                    'q_rad_sec', 'p_rad_sec', 'r_rad_sec', 'pilot-z-accel-fps_sec'
                ]
                # Trim o extender para hacer coincidir si es necesario
                if matriz_cruda.shape[1] < len(nombres_columnas):
                    nombres_columnas = nombres_columnas[:matriz_cruda.shape[1]]

        df_vuelo = pd.DataFrame(matriz_cruda, columns=nombres_columnas)
        self.vuelo_actual_data = df_vuelo.to_dict('records')

    def _cargar_desde_csv(self, ruta: str):
        try:
            try:
                df = pd.read_csv(ruta)
            except pd.errors.EmptyDataError:
                event_bus.error_ocurrido.emit("El archivo CSV está vacío o corrupto.")
                return

            if df.empty or len(df) < 2:
                event_bus.error_ocurrido.emit("El CSV no tiene datos suficientes para interpolar.")
                return

            if 'timestamp_ms' in df.columns:
                # Ordenar y eliminar duplicados temporales que romperían scipy PchipInterpolator
                df = df.sort_values('timestamp_ms').drop_duplicates(subset=['timestamp_ms'])

            print("[DataManager] Aplicando PCHIP al CSV crudo...")
            time_raw = df['timestamp_ms'].values

            # Crear nuevo eje temporal uniforme a 50Hz (20ms)
            time_uniform = np.arange(time_raw[0], time_raw[-1] + int(DT * 1000), int(DT * 1000))

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

            # Convirtiendo a float nativo para evitar problemas de pyqtSignal con numpy.float64
            df_vuelo = df_vuelo.astype(float)

            self.vuelo_actual_data = df_vuelo.to_dict('records')

        except Exception as e:
            event_bus.error_ocurrido.emit(f"Error CSV: {e}")

    def set_speed_multiplier(self, mult: float):
        self.speed_multiplier = mult
        if self.timer.isActive():
            if self.speed_multiplier > 10.0: # MAX mode (as fast as possible)
                self.timer.start(0)
            else:
                base_ms = int(DT * 1000)
                new_interval = max(1, int(base_ms / self.speed_multiplier))
                self.timer.start(new_interval)

    def _play(self):
        if not self.vuelo_actual_data: return
        if self.speed_multiplier > 10.0:
            self.timer.start(0)
        else:
            base_ms = int(DT * 1000)
            new_interval = max(1, int(base_ms / self.speed_multiplier))
            self.timer.start(new_interval)
        print(f"[DataManager] Play activado. Multiplicador: {self.speed_multiplier}x")

    def _pause(self):
        self.timer.stop()

    def _seek(self, idx: int):
        if 0 <= idx < len(self.vuelo_actual_data):
            self.current_step = idx
            fila = self.vuelo_actual_data[self.current_step].copy()

            # Extraemos TODOS los datos anteriores para reconstruir la gráfica
            # y el buffer de ML Worker
            inicio_ml = max(0, self.current_step - 499)
            fila['ventana_salto'] = self.vuelo_actual_data[inicio_ml : self.current_step + 1]
            fila['historia_completa'] = self.vuelo_actual_data[0 : self.current_step + 1]
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
