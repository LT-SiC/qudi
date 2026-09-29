#!/usr/bin/env python3
"""
Quantum Composer — entry point.

Works from VS Code, Jupyter, or plain terminal:
    python run_composer.py

From a Jupyter cell:
    %run run_composer.py
"""
from __future__ import print_function
import sys
import os

# Add THIS file's directory to sys.path so all sibling .py files are importable
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from main_window import MainWindow

#######

from qtpy import QtCore, QtWidgets, QtGui

from PyQt5 import QtCore as QtCore_PyQt
from PyQt5 import QtGui as QtGui_PyQt
from PyQt5 import QtWidgets as QtWidgets_PyQt
from PyQt5.QtCore import pyqtSignal, QTimer
import PyQt5.uic
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QMainWindow,
    QFileDialog,
    QMessageBox,
)


import sip

# ---- QUDI imports ----
import logic.misc
import logic.misc as misc
from logic.generic_logic import GenericLogic
from core.connector import Connector
from core.util.mutex import Mutex


from gui.guibase import GUIBase
from logic.qudip_enhanced import *   # legacy QUDI API requires star import
from core.statusvariable import StatusVar

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Quantum Composer")

    font = QFont("Segoe UI", 10)
    app.setFont(font)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    window = MainWindow()
    window.show()

    # In Jupyter don't sys.exit — just run the event loop
    if 'ipykernel' in sys.modules:
        app.exec_()
    else:
        sys.exit(app.exec_())


class circuit_gui(GUIBase):
    # Connect to the Queue.
    queue_logic = Connector(interface="queue_logic")  # class name

    def on_activate(self):
        self._win = MainWindow()
        self._win.setWindowTitle("Circuit designer")
        self._mw = self._win
        self._queue_logic = self.queue_logic()
        # Wire the queue logic into the window so ▶ Run can call add_task()
        self._win.queuelogic = self._queue_logic
        self._win.show()

    def on_deactivate(self):
        self._mw.close()
