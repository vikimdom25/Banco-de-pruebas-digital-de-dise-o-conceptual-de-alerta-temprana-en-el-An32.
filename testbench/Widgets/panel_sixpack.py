# panel_sixpack.py
import sys
import pandas as pd
import numpy as np
from PyQt6 import QtWidgets, QtCore, QtGui

# Importar TODOS los widgets de instrumentos
from Instruments import (
    AttitudeIndicatorWidget, HeadingIndicatorWidget, 
    AirspeedIndicatorWidget, AltimeterWidget, 
    VSIndicatorWidget, TurnIndicatorWidget
)

from Instruments import (
    AttitudeIndicatorWidget, HeadingIndicatorWidget, 
    AirspeedIndicatorWidget, AltimeterWidget, 
    VSIndicatorWidget, TurnIndicatorWidget
)

class PanelSixPack(QtWidgets.QWidget):
    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        # --- Preprocesamiento (debe estar definido o copiado aquí si no es global) ---
        self.df = self._preprocesar_dataframe(df.copy()) if df is not None else pd.DataFrame()
        # --------------------------------------------------------------------------
        self.current_idx = 0

        # self.setWindowTitle("Panel de Instrumentos Six-Pack") # El título lo pone la pestaña
        main_layout = QtWidgets.QVBoxLayout(self)
        self.setLayout(main_layout) # Asignar layout al widget PanelSixPack

        instrument_grid_layout = QtWidgets.QGridLayout()
        instrument_grid_layout.setSpacing(6) # Espacio entre instrumentos
        instrument_grid_layout.setContentsMargins(5,5,5,5)


        instrument_names = [
            "VELOCÍMETRO", "ACTITUD", "ALTÍMETRO",
            "IND. DE VIRAJE", "RUMBO", "VARIÓMETRO"
        ]
        
        self.airspeed_indicator = AirspeedIndicatorWidget()
        self.attitude_indicator = AttitudeIndicatorWidget()
        self.altimeter = AltimeterWidget()                
        self.turn_indicator = TurnIndicatorWidget()      
        self.heading_indicator = HeadingIndicatorWidget()
        self.vsi_indicator = VSIndicatorWidget()           

        instruments = [
            self.airspeed_indicator, self.attitude_indicator, self.altimeter,
            self.turn_indicator, self.heading_indicator, self.vsi_indicator
        ]

        positions = [(i, j) for i in range(2) for j in range(3)] 

        for position, name, instrument_widget in zip(positions, instrument_names, instruments):
            container = QtWidgets.QWidget()
            v_layout = QtWidgets.QVBoxLayout(container)
            v_layout.setContentsMargins(1,1,1,1) # Márgenes pequeños para el contenedor
            v_layout.setSpacing(1) # Espacio mínimo entre etiqueta e instrumento

            label = QtWidgets.QLabel(name)
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            font = label.font(); font.setPointSize(7); font.setBold(True); label.setFont(font)
            label.setFixedHeight(15) # Altura fija para la etiqueta
            
            v_layout.addWidget(label)
            v_layout.addWidget(instrument_widget)
            
            # Para que los instrumentos se expandan y ocupen el espacio disponible en la celda
            instrument_widget.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
            instrument_grid_layout.addWidget(container, position[0], position[1])
        
        main_layout.addLayout(instrument_grid_layout)

        # --- Controles de Tiempo ---
        time_controls_group = QtWidgets.QGroupBox("Control de Simulación")
        time_controls_group.setFixedHeight(70) # Altura fija para el grupo de controles
        controls_layout_h = QtWidgets.QHBoxLayout(time_controls_group) 
        
        controls_layout_h.addWidget(QtWidgets.QLabel("Índice Dato:")) 
        self.time_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        if not self.df.empty:
            self.time_slider.setRange(0, len(self.df) - 1); self.time_slider.setValue(0)
        else:
            self.time_slider.setRange(0, 0); self.time_slider.setEnabled(False)
        self.time_slider.valueChanged.connect(self._actualizar_por_slider)
        controls_layout_h.addWidget(self.time_slider)
        
        self.current_time_label = QtWidgets.QLabel("Tiempo: 0.00 s")
        self.current_time_label.setFixedWidth(100) 
        self.current_time_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        controls_layout_h.addWidget(self.current_time_label)
        
        # time_controls_group.setLayout(controls_layout_h) # No es necesario si se pasa como padre al crear QHBoxLayout
        main_layout.addWidget(time_controls_group)

        if not self.df.empty:
            QtCore.QTimer.singleShot(0, lambda: self._actualizar_instrumentos(self.current_idx))


    def _preprocesar_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty: return pd.DataFrame()
        # print("[PanelSixPack] Preprocesando DataFrame...")
        df_copy = df.copy()
        expected_cols_defaults = {
            'pitch-deg': 0.0, 'roll-deg': 0.0, 'heading-deg': 0.0,
            'altitude-ft': 0.0, 'airspeed-kt': 0.0, 'vertical-speed-fps': 0.0,
            'timestamp': None
        }
        for col, default_val in expected_cols_defaults.items():
            if col not in df_copy.columns:
                if col == 'timestamp': df_copy[col] = pd.Series(np.arange(len(df_copy))) # Timestamp de emergencia
                else: df_copy[col] = default_val
            if df_copy[col].dtype == 'object' or pd.api.types.is_string_dtype(df_copy[col]):
                try:
                    if hasattr(df_copy[col], 'str') and df_copy[col].str.contains(',',na=False).any():
                        df_copy[col] = df_copy[col].str.replace(',', '.', regex=False)
                except: pass
            if col != 'timestamp': df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(default_val)
        if 'timestamp' in df_copy.columns:
            col_ts = df_copy['timestamp']
            if pd.api.types.is_datetime64_any_dtype(col_ts.dtype): df_copy['timestamp_dt'] = col_ts
            elif pd.api.types.is_numeric_dtype(col_ts.dtype): df_copy['timestamp_dt'] = pd.to_datetime(col_ts, unit='s', errors='coerce')
            else: df_copy['timestamp_dt'] = pd.to_datetime(col_ts, errors='coerce')
            if 'timestamp_dt' in df_copy and not df_copy['timestamp_dt'].isnull().all():
                first_valid = df_copy['timestamp_dt'].dropna().iloc[0] if not df_copy['timestamp_dt'].dropna().empty else pd.NaT
                if pd.notna(first_valid): df_copy['time_seconds'] = (df_copy['timestamp_dt'] - first_valid).dt.total_seconds()
                else: df_copy['time_seconds'] = np.arange(len(df_copy))
            else: df_copy['time_seconds'] = np.arange(len(df_copy))
            df_copy.drop(columns=['timestamp_dt'], errors='ignore', inplace=True)
        else: df_copy['time_seconds'] = np.arange(len(df_copy))
        # print("[PanelSixPack] Preprocesamiento completo.")
        return df_copy.copy()

    def _actualizar_por_slider(self, value: int):
        self._actualizar_instrumentos(value)

    def _actualizar_instrumentos(self, idx: int):
        if self.df.empty or idx < 0 or idx >= len(self.df): return
        self.current_idx = idx
        current_data_row = self.df.iloc[self.current_idx]
        if self.time_slider.value() != self.current_idx: self.time_slider.setValue(self.current_idx)
        time_val = current_data_row.get('time_seconds', float(self.current_idx))
        self.current_time_label.setText(f"Tiempo: {time_val:.2f}s")
        self.attitude_indicator.set_attitude(current_data_row.get('pitch-deg',0.0), current_data_row.get('roll-deg',0.0))
        self.heading_indicator.set_heading(current_data_row.get('heading-deg',0.0))
        self.airspeed_indicator.set_airspeed(current_data_row.get('airspeed-kt',0.0))
        self.altimeter.set_altitude(current_data_row.get('altitude-ft',0.0))
        self.vsi_indicator.set_vertical_speed(current_data_row.get('vertical-speed-fps',0.0))
        self.turn_indicator.set_bank_angle(current_data_row.get('roll-deg',0.0))


# --- Bloque if __name__ == "__main__" (Sin cambios, asumiendo que ya tiene los datos) ---
if __name__ == '__main__':
    # ... (El código del bloque if __name__ == "__main__" que ya tienes) ...
    app = QtWidgets.QApplication(sys.argv)
    time_steps = 500
    t_sec = np.linspace(0, 120, time_steps) 
    data_dict = {
        'timestamp': pd.to_datetime(t_sec, unit='s', origin=pd.Timestamp('2025-05-15 09:00:00')),
        'pitch-deg': 15 * np.sin(t_sec * 0.1) + np.random.randn(time_steps) * 0.5,
        'roll-deg': 30 * np.sin(t_sec * 0.07 + np.pi/2) + np.random.randn(time_steps) * 1,
        'heading-deg': (t_sec * 15 + np.random.randn(time_steps) * 5) % 360,
        'altitude-ft': 5000 + 1000 * np.sin(t_sec * 0.05) + np.random.randn(time_steps) * 10,
        'airspeed-kt': 80 + 20 * np.sin(t_sec * 0.08) + np.random.randn(time_steps) * 2,
        'vertical-speed-fps': 10 * np.cos(t_sec * 0.1) * 1.5 + np.random.randn(time_steps) * 0.5,
        'glideslope': 0.5 * np.sin(t_sec * 0.02) 
    }
    sample_df_list = []
    for i in range(time_steps):
        row = {}
        for key, series in data_dict.items():
            val = series[i]
            if key in ['pitch-deg', 'airspeed-kt'] and i % 30 < 2 : row[key] = f"{val:.2f}".replace('.', ',')
            else: row[key] = val
        if i % 60 == 5 and key not in ['timestamp', 'time_seconds']: row[key] = np.nan
        sample_df_list.append(row)
    main_df = pd.DataFrame(sample_df_list)
    main_panel_widget = PanelSixPack(main_df)
    main_panel_widget.show()
    sys.exit(app.exec())