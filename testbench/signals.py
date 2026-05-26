from PyQt6.QtCore import QObject, pyqtSignal

class GlobalEventBus(QObject):
    # Señales de Datos (Emitidas por DataManager/Replay Engine)
    # Envía un diccionario con el estado crudo del avión en el instante actual
    telemetry_updated = pyqtSignal(dict)

    # Señal cuando se carga un nuevo vuelo (reinicia buffers e interfaces)
    vuelo_cargado = pyqtSignal(int) # total_pasos

    # Señales de Control de Simulación (Emitidas por UI -> DataManager)
    play_requested = pyqtSignal()
    pause_requested = pyqtSignal()
    seek_requested = pyqtSignal(int) # Indice del nuevo paso de tiempo

    # Señales de Inferencia (Emitidas por ML Worker -> UI)
    # dict con: 'estado_actual', 'estado_futuro', 'prob_peligro', 't2s', 'sigma', 'riesgo_salud'
    inference_updated = pyqtSignal(dict)

    # Alerta amarilla de EICAS
    eicas_yellow_alert = pyqtSignal(bool)

    # Señales de Sistema
    error_ocurrido = pyqtSignal(str)
    estado_sistema_cambiado = pyqtSignal(str)

# Instancia global (Singleton) para ser importada en todo el Testbench
event_bus = GlobalEventBus()
