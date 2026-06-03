import sys
import os
import time
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

# Add testbench to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'testbench'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'testbench', 'Widgets'))

from testbench.signals import event_bus
from testbench.core import EngineeringWorkbench

app = QApplication(sys.argv)
wb = EngineeringWorkbench()

def check_dock():
    try:
        # En PyQt6, un QDockWidget que se cierra se oculta, no se elimina, se puede restaurar con setVisible(True)
        # o mediante los menús nativos del QMainWindow
        pass
    except Exception as e:
        print(e)
    app.quit()

QTimer.singleShot(100, check_dock)

sys.exit(app.exec())
