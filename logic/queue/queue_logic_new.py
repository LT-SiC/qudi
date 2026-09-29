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

import sys
import os
import shutil
import uuid
import threading
import importlib.util
import ast
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

from typing import Optional

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
# ScriptThread
# ---------------------------
class ScriptThread(QtCore.QThread):
    """Run embedded class-based tasks inside a QThread and report via signals."""

    log_output = QtCore.Signal(str)
    finished_signal = QtCore.Signal(object)

    def __init__(self, task: ScriptTask, task_updated_signal: Optional[QtCore.Signal] = None):
        super(ScriptThread, self).__init__()
        self.task = task
        self._stop_event = threading.Event()
        self.task_updated_signal = task_updated_signal

    def run(self):
        path = str(self.task.queued_path)
        name = "task_" + self.task.id
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception as e:
            self.log_output.emit("[ERROR] Failed to load module: {}\n".format(e))
            self.task.status = Status.ERROR
            self.finished_signal.emit(self.task)
            return

        # Find any class with a 'run_fun' method
        candidate_cls = None
        for attr_name in dir(mod):
            try:
                attr = getattr(mod, attr_name)
            except Exception:
                continue
            if isinstance(attr, type) and hasattr(attr, "run_fun"):
                candidate_cls = attr
                break

        if not candidate_cls:
            self.log_output.emit("[ERROR] No class with 'run_fun' found.\n")
            self.task.status = Status.ERROR
            self.finished_signal.emit(self.task)
            return

        # Instantiate and run
        instance = candidate_cls()
        self.task.worker = instance
        self.task.status = Status.RUNNING
        self.task.start_time = datetime.now()
        if self.task_updated_signal is not None:
            # Emit the bound signal (it's a real Qt bound signal)
            try:
                self.task_updated_signal.emit(self.task)
            except Exception:
                pass
        self.log_output.emit("[INFO] Started task '{}'\n".format(candidate_cls.__name__))

        try:
            # Run blocking run_fun() inside this QThread
            instance.run_fun()
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.log_output.emit("[ERROR] Exception in run_fun: {}\n{}\n".format(e, tb))
        finally:
            # Stop/cleanup: we use stop() only from stop(), not here (avoid double stop)
            self.task.end_time = datetime.now()
            if self.task.start_time:
                self.task.accumulated_time += (self.task.end_time - self.task.start_time).total_seconds()
            self.task.status = Status.ABORTED if self._stop_event.is_set() else Status.COMPLETED
            self.finished_signal.emit(self.task)

    def stop(self):
        """Request stop from other thread (e.g., abort). Calls worker.stop() once."""
        self._stop_event.set()
        if self.task.worker and hasattr(self.task.worker, "stop"):
            try:
                self.task.worker.stop()
            except Exception:
                pass


# ---------------------------
# QueueManager with signal proxy
# ---------------------------
class QueueManagerSignals(QtCore.QObject):
    task_added = QtCore.Signal(object)
    task_updated = QtCore.Signal(object)
    log_output = QtCore.Signal(str)


class QueueManager(object,GenericLogic):
    """Queue manager (not a QObject) — uses .signals (a QObject) to emit signals."""

    # declare connections
    mcas_holder = Connector(interface='McasDictHolderInterface')
    transition_tracker = Connector(interface='TransitionTracker')  # Should be a name of the class
    confocal = Connector('ConfocalLogic')
    gated_counter = Connector('GatedCounter')  # Should be name of the class.
    optimizer = Connector('OptimizerLogic')
    fastcounter = Connector(interface='TimeTaggerInterface')
    PLE_logic = Connector("LaserScannerLogic")
    odmr_logic = Connector("ODMRLogic_holder")
    currentmeasurementlogic = Connector("CurrentMeasurementLogic")
    # biaslogic= Connector("BiasLogic")
    poimanagerlogic = Connector('PoiManagerLogic')
    powerstabilization_logic = Connector("PowerStabilizationLogic")
    counterlogic1 = Connector("CounterLogic")

    # update_selected_user_script_combo_box_signal = pyqtSignal(collections.OrderedDict)
    # update_queue_list = pyqtSignal(collections.OrderedDict)
    # user_script_list = misc.ret_property_array_like_typ('user_script_list', str)
    guis = []  # stores names of all open guis (later on used to dump them periodically)
    _StopTimeout = 60.

    __TIME_FORMAT_STR__ = '%Y%m%d-h%Hm%Ms%S'



    def __init__(self):
        super(QueueManager, self).__init__()
        self.signals = QueueManagerSignals()
        self.tasks: List[ScriptTask] = []

    ####### start ported from old queuelogic #####
    def on_activate(self):

        self._awg = self.mcas_holder()
        self._transition_tracker = self.transition_tracker()
        self._gated_counter = self.gated_counter()
        self._optimizer = self.optimizer()
        self._PLE_logic = self.PLE_logic()
        self._ODMR_logic = self.odmr_logic()
        self._currentmeasurementlogic: CurrentMeasurementLogic = self.currentmeasurementlogic()
        # self._biaslogic:BiasLogic = self.biaslogic()
        self._powerstabilization_logic = self.powerstabilization_logic()
        self._poimanagerlogic = self.poimanagerlogic()
        self._counter = self.counterlogic1()
        self._fast_counter_device = self.fastcounter()  # FIXME
        self.create_odmr()  # only logic (no gui)
        # self.init_run()
        self.write_standard_awg_sequences()
        self._confocal = self.confocal()
        self.tt = self._transition_tracker

    def on_deactivate(self):
        pass
        # FIXME destroy me gently

    @property
    def script_queue(self):
        return self._script_queue

    @property
    def nowstr(self):
        return datetime.datetime.now().strftime('%Y%m%d-h%Hm%Ms%S')

    @property
    def nowstr_colon(self):
        return datetime.datetime.now().strftime('%Y/%m/%d-%H:%M:%S')

    @property
    def nowstr_pd(self):
        return datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

    @property
    def awgs(self):
        return self._mcas_dict.awgs

    def create_odmr(self):
        self.odmr = odmr.ODMR(self)

    def set_stop_request(self):
        self.thread.stop_request.set()

    @property
    def cun(self) -> Optional[ScriptTask]:
        """Return the currently running task, if any."""
        for t in self.tasks:
            if t.status == Status.RUNNING:
                return t
        return None

    def save_pi3diamond(self, destination_dir):
        src = os.getcwd()
        f = '{}/qudi.zip'.format(destination_dir)
        if not os.path.isfile(f):
            zf = zipfile.ZipFile(f, 'a')
            for root, dirs, files in os.walk(src):
                if (not any([i in root for i in ['__pycache__',
                                                 'awg_settings',
                                                 'currently_unused',
                                                 ".idea",
                                                 ".hg",
                                                 'UserScripts',
                                                 'log',
                                                 'notebooks'  # FIXME - include logic, explode heavy stuff.
                                                 ]
                             ]
                            )
                ) or root.endswith('transition_tracker_log') or root.endswith('helpers'):
                    for file in files:
                        if any([file.endswith(i) for i in ['.py', '.dat', '.ui', '.hdf']]):
                            zf.write(os.path.join(root, file),
                                     os.path.join(root.replace(os.path.commonprefix([root, src]), ""), file))
            zf.close()

    ####### End ported from old queuelogic #####

    # convenience properties to match some QObject-like usage if needed
    @property
    def task_added(self):
        return self.signals.task_added

    @property
    def task_updated(self):
        return self.signals.task_updated

    @property
    def log_output(self):
        return self.signals.log_output

    def add_task(self, source_path: Path):
        unique_id = uuid.uuid4().hex[:8]
        dest_name = "{}_{}{}".format(source_path.stem, unique_id, source_path.suffix)
        dest = QUEUE_DIR / dest_name
        shutil.copy2(source_path, dest)
        task = ScriptTask(id=unique_id, original_path=source_path, queued_path=dest)
        self.tasks.append(task)

        # Emit via proxy
        self.signals.task_added.emit(task)
        self.signals.log_output.emit("Added {} as {}\n".format(source_path.name, dest.name))

        if not self.any_active_running():
            QtCore.QTimer.singleShot(100, self.start_next)
        return task

    def any_active_running(self):
        return any(t.status == Status.RUNNING for t in self.tasks)

    def _detect_embedded_task(self, path: Path):
        """Detect if the script defines *any* class with a `run_fun()` method using AST."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source, filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for body_item in node.body:
                        if isinstance(body_item, ast.FunctionDef) and body_item.name == "run_fun":
                            return True
            return False
        except Exception:
            return False

    def start_next(self):
        next_task = next((t for t in self.tasks if t.status == Status.QUEUED), None)
        if next_task:
            self.start_task(next_task)

    def start_task(self, task: ScriptTask):
        # If already paused embedded, resume
        if task.is_embedded and task.thread and task.status == Status.PAUSED:
            self.resume_task(task.id)
            return

        if self._detect_embedded_task(task.queued_path):
            self._start_embedded(task)
        else:
            self._start_process(task)

    def _start_embedded(self, task: ScriptTask):
        # Pass the bound signal self.signals.task_updated so the thread can emit immediate updates
        thread = ScriptThread(task, task_updated_signal=self.signals.task_updated)
        task.is_embedded = True
        task.thread = thread

        # Connect thread signals to proxy: forward text to log_output
        thread.log_output.connect(lambda msg: self.signals.log_output.emit(msg))
        # When thread finishes, call local handler
        thread.finished_signal.connect(lambda t: self._on_thread_finished(t))
        thread.start()

        # mark running and notify GUI
        task.status = Status.RUNNING
        task.start_time = datetime.now()
        self.signals.task_updated.emit(task)
        self.signals.log_output.emit("Started embedded {}\n".format(task.id))

    def _on_thread_finished(self, task: ScriptTask):
        # Ensure thread objects are cleaned up
        thr = task.thread
        if thr is not None:
            try:
                thr.quit()
            except Exception:
                pass
            try:
                thr.wait(200)
            except Exception:
                pass
            # We don't call deleteLater here (QThread without QObject parent is ok)
            task.thread = None

        task.status = Status.COMPLETED if task.status not in (Status.ABORTED, Status.ERROR) else task.status
        task.end_time = datetime.now()
        if task.start_time:
            task.accumulated_time += (task.end_time - task.start_time).total_seconds()

        self.signals.task_updated.emit(task)
        self.signals.log_output.emit("Thread task {} finished.\n".format(task.id))
        QtCore.QTimer.singleShot(100, self.start_next)

    def _start_process(self, task: ScriptTask):
        # Move file to current running sentinel name to avoid conflicts
        running_path = task.queued_path.parent / CURRENT_RUNNING_NAME
        if running_path.exists():
            try:
                running_path.unlink()
            except Exception:
                pass
        try:
            task.queued_path.rename(running_path)
            task.queued_path = running_path
        except Exception as e:
            self.signals.log_output.emit("Error renaming file: {}\n".format(e))
            task.status = Status.ERROR
            self.signals.task_updated.emit(task)
            return

        proc = QtCore.QProcess()
        proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)

        proc.readyReadStandardOutput.connect(lambda: self._on_stdout(proc, task))
        proc.finished.connect(lambda exitCode, exitStatus: self._on_finished(proc, task, exitCode, exitStatus))
        proc.start(sys.executable, [str(task.queued_path)])

        task.process = proc
        task.status = Status.RUNNING
        task.start_time = datetime.now()
        self.signals.task_updated.emit(task)
        self.signals.log_output.emit("Started {}\n".format(task.queued_path.name))

    def _on_stdout(self, proc, task):
        try:
            data = proc.readAllStandardOutput().data().decode(errors="ignore")
        except Exception:
            data = ""
        if data:
            task.stdout += data
            self.signals.log_output.emit(data)

    def _on_finished(self, proc, task, exitCode, exitStatus):
        if task.status != Status.ABORTED:
            task.status = Status.COMPLETED
        task.end_time = datetime.now()
        if task.start_time:
            task.accumulated_time += (task.end_time - task.start_time).total_seconds()
        self.signals.log_output.emit("Finished {} ({})\n".format(task.id, task.status.value))
        self.signals.task_updated.emit(task)
        try:
            if task.queued_path.exists() and task.queued_path.name == CURRENT_RUNNING_NAME:
                task.queued_path.unlink()
        except Exception:
            pass
        QtCore.QTimer.singleShot(100, self.start_next)

    def pause_task(self, task_id: str):
        task = self._find_task(task_id)
        if not task:
            return
        if task.is_embedded and task.worker and hasattr(task.worker, "pause"):
            try:
                task.worker.pause()
            except Exception:
                pass
            task.status = Status.PAUSED
            task.pause_start = datetime.now()
            self.signals.task_updated.emit(task)
            self.signals.log_output.emit("Paused embedded {}\n".format(task.id))
            return

        if not task.process or task.status != Status.RUNNING:
            return
        pid = task.process.processId()
        if not pid:
            return
        try:
            if os.name == "posix":
                import signal
                os.kill(pid, signal.SIGSTOP)
            elif psutil:
                psutil.Process(pid).suspend()
            task.status = Status.PAUSED
            task.pause_start = datetime.now()
            self.signals.log_output.emit("Paused PID {}\n".format(pid))
            self.signals.task_updated.emit(task)
        except Exception as e:
            self.signals.log_output.emit("Failed to pause: {}\n".format(e))

    def resume_task(self, task_id: str):
        task = self._find_task(task_id)
        if not task:
            return
        if task.is_embedded and task.worker and hasattr(task.worker, "resume"):
            try:
                task.worker.resume()
            except Exception:
                pass
            task.status = Status.RUNNING
            self.signals.task_updated.emit(task)
            self.signals.log_output.emit("Resumed embedded {}\n".format(task.id))
            return

        if not task.process or task.status != Status.PAUSED:
            return
        pid = task.process.processId()
        if not pid:
            return
        try:
            if os.name == "posix":
                import signal
                os.kill(pid, signal.SIGCONT)
            elif psutil:
                psutil.Process(pid).resume()
            task.status = Status.RUNNING
            self.signals.log_output.emit("Resumed PID {}\n".format(pid))
            self.signals.task_updated.emit(task)
        except Exception as e:
            self.signals.log_output.emit("Failed to resume: {}\n".format(e))

    def abort_task(self, task_id: str):
        task = self._find_task(task_id)
        if not task:
            return
        # mark first to prevent requeue
        task.status = Status.ABORTED
        self.signals.task_updated.emit(task)

        try:
            if task.is_embedded and task.thread:
                task.thread.stop()
            elif task.process:
                task.process.kill()
            task.end_time = datetime.now()
            self.signals.log_output.emit("Aborted {}\n".format(task.id))
        except Exception as e:
            self.signals.log_output.emit("Failed to abort: {}\n".format(e))

    def _find_task(self, task_id: str) -> Optional[ScriptTask]:
        for t in self.tasks:
            if t.id == task_id:
                return t
        return None
