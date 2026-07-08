from PyQt6 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg
import pandas as pd
import numpy as np
import random # Sigue siendo útil si la paleta de colores se agota y quieres un fallback

from signals import event_bus

class PanelVariableVsTiempo(QtWidgets.QWidget):
    def __init__(self, df=None, parent=None):
        super().__init__(parent)
        if df is None:
            self.df = pd.DataFrame()
        else:
            self.df = self.preprocesar_df(df.copy()) # Trabajar con una copia para no modificar el original
        self.plotteable_columns = []
        self.color_palette = [
            QtGui.QColor(255, 87, 34),   # Naranja Intenso
            QtGui.QColor(33, 150, 243),  # Azul
            QtGui.QColor(76, 175, 80),   # Verde
            QtGui.QColor(255, 193, 7),   # Ámbar
            QtGui.QColor(103, 58, 183),  # Morado Intenso
            QtGui.QColor(0, 188, 212),   # Cian
            QtGui.QColor(233, 30, 99),   # Rosa
            QtGui.QColor(158, 158, 158), # Gris
            QtGui.QColor(139, 195, 74),  # Verde Lima
            QtGui.QColor(255, 235, 59),  # Amarillo
            QtGui.QColor(0, 150, 136),   # Teal
            QtGui.QColor(96, 125, 139)   # Azul Grisáceo
        ]
        self.init_ui()

    def preprocesar_df(self, df):
        # Reemplazar comas decimales por puntos solo en columnas de tipo string/object
        for col in df.columns:
            if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
                try:
                    # Intentar reemplazar comas solo si la columna parece contener números como strings
                    if df[col].str.contains(',', na=False).any():
                         df[col] = df[col].str.replace(',', '.', regex=False)
                except AttributeError: # Si no es un string accessor, puede que ya esté bien
                    pass
            # Convertir todo a numérico, los errores se convierten en NaT/NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')


        # Asumiendo que 'timestamp' es la primera columna si existe y es convertible.
        # O podría ser una columna específica que sabes que es tu referencia de tiempo.
        # Aquí mantenemos la lógica original de normalizar el timestamp si existe.
        if 'timestamp' in df.columns and not df['timestamp'].isnull().all():
            df['timestamp'] = df['timestamp'] - df['timestamp'].iloc[0]
        elif 'timestamp_ms' in df.columns and not df['timestamp_ms'].isnull().all():
            df['timestamp_ms'] = df['timestamp_ms'] - df['timestamp_ms'].iloc[0]
        else:
            # Si no hay 'timestamp', crear uno a partir del índice para poder graficar
            df['timestamp'] = np.arange(len(df))

        if not df.empty:
            df.dropna(how='all', axis=0, inplace=True) # Elimina filas donde *todo* es NaN
        return df

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Lista para selección múltiple
        self.selector = QtWidgets.QListWidget()
        # Usar QAbstractItemView.ExtendedSelection para el modo de selección
        self.selector.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.selector.setMaximumHeight(80) # Hacer el selector más pequeño
        self.selector.itemSelectionChanged.connect(self.actualizar_grafica)

        self.cb_sombras_fsm = QtWidgets.QCheckBox("Mostrar Sombras de Estado FSM (Ground Truth)")
        self.cb_sombras_fsm.setChecked(True)
        self.cb_sombras_fsm.stateChanged.connect(self.actualizar_grafica)

        event_bus.dataframe_ready.connect(self.cargar_nuevo_df)

        # Widget de Gráfico
        self.plot_widget = pg.PlotWidget(background='k') # Fondo negro
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.addLegend(offset=(-10,10)) # Añadir leyenda (offset para que no se superponga mucho)

        # Título
        self.plot_widget.setTitle("Variables vs Tiempo", color='w', size="14pt")
        self.plot_widget.setLabel('bottom', "Tiempo (unidades relativas)", color='w')
        self.plot_widget.setLabel('left', "Valor Variable", color='w')


        # Armar layout
        label_selector = QtWidgets.QLabel("Selecciona variables para graficar (Ctrl/Shift para múltiple):")
        layout.addWidget(label_selector)
        layout.addWidget(self.selector)
        layout.addWidget(self.cb_sombras_fsm)
        layout.addWidget(self.plot_widget)

        # Graficar inicialmente las primeras variables (opcional)
        if len(self.plotteable_columns) > 0:
            # Seleccionar la primera variable por defecto, si hay alguna
            self.selector.setCurrentRow(0)

    def cargar_nuevo_df(self, df_nuevo):
        self.df = self.preprocesar_df(df_nuevo.copy())

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.plotteable_columns = [col for col in numeric_cols if col not in ['timestamp', 'timestamp_ms']]

        self.selector.blockSignals(True)
        self.selector.clear()
        self.selector.addItems(self.plotteable_columns)
        self.selector.blockSignals(False)

        # Seleccionar algunas por defecto si existen
        default_cols = ['FSM_State', 'altitude-ft', 'alpha-deg', 'airspeed-kt']
        for i in range(self.selector.count()):
            item = self.selector.item(i)
            if item.text() in default_cols:
                item.setSelected(True)

        self.actualizar_grafica()

    def _dibujar_fondos_fsm(self):
        if not self.cb_sombras_fsm.isChecked():
            return

        fsm_col = None
        if 'FSM_Realtime_State' in self.df.columns:
            fsm_col = 'FSM_Realtime_State'
        elif 'FSM_State' in self.df.columns:
            fsm_col = 'FSM_State'

        if fsm_col is None:
            return

        # Determinar el eje de tiempo
        if 'timestamp_ms' in self.df.columns:
            tiempo = self.df['timestamp_ms'] / 1000.0
        elif 'timestamp' in self.df.columns:
            tiempo = self.df['timestamp']
        else:
            tiempo = np.arange(len(self.df))

        tiempos = tiempo.to_numpy()
        estados = self.df[fsm_col].to_numpy()

        # Colores por estado (mismo mapping que en el Dock)
        colores = {
            1: QtGui.QColor(255, 255, 0, 50),     # Amarillo translúcido (Advertencia)
            2: QtGui.QColor(255, 0, 0, 70),       # Rojo translúcido (Pérdida/Stall)
            3: QtGui.QColor(128, 0, 128, 70),     # Morado translúcido (Picado/Dive)
            4: QtGui.QColor(139, 0, 0, 90),       # Rojo oscuro (Impacto)
            5: QtGui.QColor(0, 255, 255, 60)      # Cyan translúcido (Recuperación)
        }

        estado_actual = 0
        inicio_idx = 0

        for i in range(len(estados)):
            if estados[i] != estado_actual:
                # Cerrar región anterior si no era Normal (0)
                if estado_actual in colores and i > 0:
                    t_inicio = tiempos[inicio_idx]
                    t_fin = tiempos[i-1]
                    if t_fin > t_inicio:
                        region = pg.LinearRegionItem(values=[t_inicio, t_fin], movable=False, brush=colores[estado_actual], pen=None)
                        self.plot_widget.addItem(region)

                estado_actual = estados[i]
                inicio_idx = i

        # Cerrar la última región si quedó abierta al final
        if estado_actual in colores and len(estados) > 0:
            t_inicio = tiempos[inicio_idx]
            t_fin = tiempos[-1]
            if t_fin > t_inicio:
                region = pg.LinearRegionItem(values=[t_inicio, t_fin], movable=False, brush=colores[estado_actual], pen=None)
                self.plot_widget.addItem(region)

    def actualizar_grafica(self):
        self.plot_widget.clear() # Limpiar plots anteriores

        # Dibujar fondos FSM si está activado
        self._dibujar_fondos_fsm()

        # La leyenda se limpia con clear(), pero se puede volver a añadir si se desea
        # o configurarla para que persista si los items se eliminan individualmente.
        # Por simplicidad, se recrea con cada update si es necesario.
        # Si ya se añadió en init_ui y no se borra explícitamente, podría no necesitarse aquí.
        # No es necesario volver a añadirla si ya existe y solo actualizamos los datos de las curvas.
        # self.plot_widget.addLegend(offset=(-10,10)) # Se añade en init_ui

        # Usar timestamp_ms si existe, sino timestamp
        if 'timestamp_ms' in self.df.columns:
            tiempo = self.df['timestamp_ms'] / 1000.0 # Segundos
        else:
            tiempo = self.df.get('timestamp', np.arange(len(self.df)))

        selected_items = self.selector.selectedItems()
        if not selected_items: # Si no hay nada seleccionado, no hacer nada más
             self.plot_widget.setTitle("Variables vs Tiempo (Ninguna variable seleccionada)", color='w', size="14pt")
             return
        else:
            self.plot_widget.setTitle("Variables vs Tiempo", color='w', size="14pt")


        for i, item_widget in enumerate(selected_items):
            variable_name = item_widget.text()

            if variable_name not in self.df.columns:
                continue # Seguridad, aunque no debería pasar con QListWidget poblado así

            valores = self.df[variable_name] # Ya es numérico

            # Máscara para filtrar datos válidos (no NaN/NaT en tiempo o valores)
            # NaT (Not a Time) también se considera no finito por isfinite en algunos contextos numéricos
            valid_mask = np.isfinite(tiempo) & np.isfinite(valores)

            t_plot = tiempo[valid_mask].to_numpy()
            y_plot = valores[valid_mask].to_numpy()

            if t_plot.size == 0 or y_plot.size == 0: # No hay datos válidos para graficar
                continue

            # Obtener el índice original de la columna para un color consistente
            try:
                original_col_index = self.plotteable_columns.index(variable_name)
                color = self.color_palette[original_col_index % len(self.color_palette)]
            except ValueError: # Si por alguna razón no está en plotteable_columns (no debería)
                color = QtGui.QColor(random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))

            pen = pg.mkPen(color=color, width=2)

            # Añadir la curva al plot. El 'name' es usado por la leyenda.
            self.plot_widget.plot(t_plot, y_plot, pen=pen, name=variable_name)


# --- Ejemplo de Uso ---
if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)

    # Crear un DataFrame de ejemplo
    num_puntos = 200
    datos = {
        'timestamp': pd.to_datetime(np.arange(num_puntos), unit='s'), # Ejemplo con datetime real
        'Temperatura': np.random.rand(num_puntos) * 30 + 10 + np.sin(np.linspace(0, 10, num_puntos)) * 5,
        'Presion': np.random.rand(num_puntos) * 500 + 98000 + np.cos(np.linspace(0, 15, num_puntos)) * 200,
        'Humedad': np.random.rand(num_puntos) * 60 + 20,
        'SensorX_str_comma': [f"{val:.2f}".replace('.', ',') for val in np.random.rand(num_puntos) * 10], # Datos con comas
        'Vibracion': np.sin(np.linspace(0, 50, num_puntos)) * 2 + np.random.randn(num_puntos)*0.2,
        'Luz': None # Columna con Nones para probar
    }
    # Añadir más columnas para probar la paleta de colores
    for i in range(10):
        datos[f'ExtraSensor_{i+1}'] = np.random.rand(num_puntos) * (i+1)


    sample_df = pd.DataFrame(datos)
    sample_df.loc[5:10, 'Temperatura'] = np.nan # Introducir algunos NaNs
    sample_df.loc[15:20, 'Presion'] = None   # Introducir algunos None


    panel = PanelVariableVsTiempo(sample_df)
    panel.setWindowTitle("Panel de Gráficas Variables vs Tiempo")
    panel.resize(800, 600)
    panel.show()

    sys.exit(app.exec())