#!/usr/bin/env python3
"""
QueueLogic GUI (Hybrid Version) — cleaned and fixed
- QueueManager uses a QObject "signals" proxy so it does not need to inherit QObject
  (avoids metaclass/MRO conflicts)
- ScriptThread runs embedded classes safely and reports via signals
- MainWindow is a wrapper around a real QMainWindow (no direct inheritance)
- Safe table rebuild (reorder) and stable signal/slot wiring
"""
from __future__ import print_function

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from datetime import datetime
from typing import Optional, List

from qtpy import QtCore, QtWidgets, QtGui

#### imports from qudi
import sys, os
import logic.misc
import imp
from gui.queue.Queue import queue_gui

from queue import Queue

from PyQt5.QtWidgets import QAbstractItemView, QMainWindow, QFileDialog
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSignal
import PyQt5.uic
import PyQt5.QtWidgets
from logic.qudip_enhanced import *
# FIXME
from PyQt5.QtCore import QTimer
from PyQt5 import QtTest
# import multi_channel_awg_seq as MCAS; reload(MCAS)
import logic.misc as misc
import datetime
import logging.handlers
import os
import pickle
import sys
import threading
import time
import traceback
from logic.generic_logic import GenericLogic
from core.connector import Connector
import multiprocess
import numpy as np
import logging

import collections
import importlib
import logic.ODMR_nops as odmr;

importlib.reload(odmr)
from core.util.mutex import Mutex

import datetime
import logging
import logging.handlers
import os
import pickle
import shutil
import sys
import threading
import time
from core.connector import Connector
from gui.guibase import GUIBase
import traceback
#import TimeTaggerHandler
import numpy as np
import psutil
import collections
from PyQt5 import QtCore, QtWidgets

import sys, os
import logic.misc

from PyQt5.QtWidgets import QAbstractItemView, QMainWindow, QFileDialog
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSignal
import PyQt5.uic
import PyQt5.QtWidgets
from logic.qudip_enhanced import *
import importlib
setup = False
#import multi_channel_awg_seq as MCAS; importlib.reload(MCAS)
import sip

#### end imports from qudi



try:
    import qdarktheme
except Exception:
    qdarktheme = None

try:
    import psutil
except Exception:
    psutil = None

QUEUE_DIR = Path.cwd() / "queue"
QUEUE_DIR.mkdir(exist_ok=True)
CURRENT_RUNNING_NAME = "queuelogic.cun"


class Status(Enum):
    QUEUED = "Queued"
    RUNNING = "Running"
    PAUSED = "Paused"
    COMPLETED = "Completed"
    ERROR = "Error"
    ABORTED = "Aborted"


@dataclass
class ScriptTask:
    id: str
    original_path: Path
    queued_path: Path
    status: Status = Status.QUEUED
    process: Optional[QtCore.QProcess] = None
    thread: Optional[QtCore.QThread] = None
    worker: Optional[object] = None
    pid: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    accumulated_time: float = 0.0
    pause_start: Optional[datetime] = None
    is_embedded: bool = False


# ---------------------------
# MainWindow wrapper (no direct inheritance)
# ---------------------------
class MainWindow(object,GUIBase):
    """
    Wrapper around QMainWindow so the class itself does not directly inherit from QMainWindow.
    Exposes show(), resize(), setWindowTitle(), raise_(), and makes widget attributes
    available as attributes on this wrapper.
    """
    # Connect to the Queue.
    queue_logic = Connector(interface="queue_logic")  # class name

    def __init__(self):
        # real internal QMainWindow
        self._win = QtWidgets.QMainWindow()
        #self._init_ui()
        #self.queue_manager = QueueManager()

        # set a fixed window title / size like before
        self._win.setWindowTitle("QueueLogic")
        self._win.resize(1100, 650)

    def on_activate(self):
        self._mw = self.win
        self._queue_logic = self.queue_logic()
        self.init_gui()
        self._connect_signals()
        self.apply_theme()

    def on_deactivate(self):
        self._mw.close()

    def manuel_pause(self,value):
        print("Manuel pause:",value)
        if hasattr(self._queue_logic,"cun"):
            self._queue_logic.cun.manual_pause=value

    # --- convenience API that mirrors the real QMainWindow ---
    def show(self):
        return self._win.show()

    def resize(self, w, h):
        return self._win.resize(w, h)

    def setWindowTitle(self, title):
        return self._win.setWindowTitle(title)

    def raise_(self):
        try:
            return self._win.raise_()
        except Exception:
            return None

    def _init_ui(self):
        # Build the UI elements and attach central widget to the internal QMainWindow
        central = QtWidgets.QWidget()
        self._win.setCentralWidget(central)
        v = QtWidgets.QVBoxLayout(central)

        top_h = QtWidgets.QHBoxLayout()
        self.folder_line = QtWidgets.QLineEdit()
        self.folder_btn = QtWidgets.QPushButton("Select Folder")
        self.refresh_btn = QtWidgets.QPushButton("Refresh .py list")
        self.file_combo = QtWidgets.QComboBox()
        self.file_combo.setMinimumWidth(350)
        self.file_combo.setMinimumHeight(35)
        self.add_btn = QtWidgets.QPushButton("Add to Queue")
        top_h.addWidget(self.folder_line)
        top_h.addWidget(self.folder_btn)
        top_h.addWidget(self.refresh_btn)
        top_h.addWidget(self.file_combo)
        top_h.addWidget(self.add_btn)
        v.addLayout(top_h)

        self.queue_table = QtWidgets.QTableWidget(0, 7)
        self.queue_table.setHorizontalHeaderLabels(
            ["Name", "Status", "Actions", "Start Time", "End Time", "Exec Time (s)", "ID"]
        )
        self.queue_table.setColumnHidden(6, True)
        self.queue_table.verticalHeader().setVisible(False)
        self.queue_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.queue_table.setColumnWidth(2, 320)
        self.queue_table.verticalHeader().setDefaultSectionSize(45)
        v.addWidget(self.queue_table)

        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        v.addWidget(self.log, stretch=1)

    def apply_theme(self):
        if qdarktheme:
            qdarktheme.setup_theme("dark")
        else:
            self._win.setStyleSheet("QWidget { background-color: #2b2b2b; color: #e6e6e6; }")

    def _connect_signals(self):
        # UI event wiring
        self.folder_btn.clicked.connect(self.select_folder)
        self.refresh_btn.clicked.connect(self.refresh_py_list)
        self.add_btn.clicked.connect(self.add_selected_to_queue)

        # QueueManager signals (proxy)
        s = self.queue_manager.signals
        s.task_added.connect(self.on_task_added)
        s.task_updated.connect(self.on_task_updated)
        s.log_output.connect(self.append_log)

    # ---- UI callbacks ----
    def select_folder(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self._win, "Select Folder", str(Path.cwd()))
        if d:
            self.folder_line.setText(d)
            self.refresh_py_list()

    def refresh_py_list(self):
        folder = Path(self.folder_line.text())
        self.file_combo.clear()
        if not folder.exists() or not folder.is_dir():
            return
        for p in sorted(folder.glob("*.py")):
            self.file_combo.addItem(str(p.name), str(p))

    def add_selected_to_queue(self):
        idx = self.file_combo.currentIndex()
        if idx < 0:
            return
        path = self.file_combo.currentData()
        if not path:
            return
        self.queue_manager.add_task(Path(path))

    # ---- table management ----
    def on_task_added(self, task: ScriptTask):
        row = self.queue_table.rowCount()
        self.queue_table.insertRow(row)
        self.queue_table.setItem(row, 0, QtWidgets.QTableWidgetItem(task.queued_path.name))
        self.queue_table.setItem(row, 1, QtWidgets.QTableWidgetItem(task.status.value))
        for col in range(3, 6):
            self.queue_table.setItem(row, col, QtWidgets.QTableWidgetItem(""))
        id_item = QtWidgets.QTableWidgetItem(task.id)
        self.queue_table.setItem(row, 6, id_item)

        w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(w)
        h.setContentsMargins(2, 2, 2, 2)
        btn_pause = QtWidgets.QPushButton("Pause")
        btn_run = QtWidgets.QPushButton("Run")
        btn_abort = QtWidgets.QPushButton("Abort")
        btn_pause.clicked.connect(lambda _, tid=task.id: self.queue_manager.pause_task(tid))
        btn_run.clicked.connect(lambda _, t=task: self.queue_manager.start_task(t))
        btn_abort.clicked.connect(lambda _, tid=task.id: self.queue_manager.abort_task(tid))
        h.addWidget(btn_pause)
        h.addWidget(btn_run)
        h.addWidget(btn_abort)
        self.queue_table.setCellWidget(row, 2, w)

    def on_task_updated(self, task: ScriptTask):
        # Update row cells
        for r in range(self.queue_table.rowCount()):
            id_item = self.queue_table.item(r, 6)
            if id_item and id_item.text() == task.id:
                self.queue_table.item(r, 1).setText(task.status.value)
                start_str = task.start_time.strftime("%H:%M:%S") if task.start_time else ""
                end_str = task.end_time.strftime("%H:%M:%S") if task.end_time else ""
                exec_time = "{:.1f}".format(task.accumulated_time) if task.accumulated_time else ""
                self.queue_table.item(r, 3).setText(start_str)
                self.queue_table.item(r, 4).setText(end_str)
                self.queue_table.item(r, 5).setText(exec_time)
                break

        # schedule a safe reorder of the whole table in the next event loop pass
        QtCore.QTimer.singleShot(0, self.reorder_queue_table)

    def reorder_queue_table(self):
        """Rebuild table from authoritative manager.tasks; keep active tasks on top."""
        try:
            order = {
                Status.RUNNING: 0,
                Status.PAUSED: 1,
                Status.QUEUED: 2,
                Status.ERROR: 3,
                Status.COMPLETED: 4,
                Status.ABORTED: 5,
            }

            self.queue_table.setSortingEnabled(False)
            self.queue_table.blockSignals(True)
            self.queue_table.setUpdatesEnabled(False)
            self.queue_table.viewport().setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
            QtWidgets.QApplication.instance().processEvents()

            # detach & schedule deletion of any current action widgets
            old_widgets = []
            for r in range(self.queue_table.rowCount()):
                w = self.queue_table.cellWidget(r, 2)
                if w is not None:
                    # disconnect buttons inside to be safe
                    for btn in w.findChildren(QtWidgets.QPushButton):
                        try:
                            btn.clicked.disconnect()
                        except Exception:
                            pass
                    w.setParent(None)
                    old_widgets.append(w)
                    w.deleteLater()

            # Sort tasks from manager
            tasks_sorted = sorted(
                list(self.queue_manager.tasks),
                key=lambda t: (order.get(t.status, 99), getattr(t, "start_time") or datetime.min),
            )

            self.queue_table.setRowCount(0)
            for task in tasks_sorted:
                r = self.queue_table.rowCount()
                self.queue_table.insertRow(r)
                self.queue_table.setItem(r, 0, QtWidgets.QTableWidgetItem(task.queued_path.name if task.queued_path else ""))
                self.queue_table.setItem(r, 1, QtWidgets.QTableWidgetItem(task.status.value))

                # New action widgets (fresh)
                w = QtWidgets.QWidget()
                h = QtWidgets.QHBoxLayout(w)
                h.setContentsMargins(2, 2, 2, 2)
                btn_pause = QtWidgets.QPushButton("Pause")
                btn_run = QtWidgets.QPushButton("Run")
                btn_abort = QtWidgets.QPushButton("Abort")
                btn_pause.clicked.connect(lambda _, tid=task.id: self.queue_manager.pause_task(tid))
                btn_run.clicked.connect(lambda _, t=task: self.queue_manager.start_task(t))
                btn_abort.clicked.connect(lambda _, tid=task.id: self.queue_manager.abort_task(tid))
                h.addWidget(btn_pause)
                h.addWidget(btn_run)
                h.addWidget(btn_abort)
                self.queue_table.setCellWidget(r, 2, w)

                start_str = task.start_time.strftime("%H:%M:%S") if task.start_time else ""
                end_str = task.end_time.strftime("%H:%M:%S") if task.end_time else ""
                exec_time = "{:.1f}".format(task.accumulated_time) if task.accumulated_time else ""
                self.queue_table.setItem(r, 3, QtWidgets.QTableWidgetItem(start_str))
                self.queue_table.setItem(r, 4, QtWidgets.QTableWidgetItem(end_str))
                self.queue_table.setItem(r, 5, QtWidgets.QTableWidgetItem(exec_time))
                id_item = QtWidgets.QTableWidgetItem(task.id)
                self.queue_table.setItem(r, 6, id_item)

            self.queue_table.viewport().setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, False)
        except Exception as e:
            try:
                self.append_log("[ERROR] reorder_queue_table failed: {}\n".format(e))
            except Exception:
                pass
        finally:
            self.queue_table.setUpdatesEnabled(True)
            self.queue_table.blockSignals(False)
            self.queue_table.setSortingEnabled(True)
            QtWidgets.QApplication.instance().processEvents()

    # ---- logging ----
    def append_log(self, text: str):
        self.log.moveCursor(QtGui.QTextCursor.End)
        self.log.insertPlainText(text)
        self.log.moveCursor(QtGui.QTextCursor.End)