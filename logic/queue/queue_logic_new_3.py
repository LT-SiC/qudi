#!/usr/bin/env python3
"""
QueueLogic GUI (Hybrid Version) — cleaned and fixed, with captured-GUI mode.

Two execution modes:
 - capture_gui_mode = False -> legacy behavior (embedded worker runs imports & UI creation directly)
 - capture_gui_mode = True  -> attempt to intercept PyQt5.uic.loadUi and execute it in main GUI thread
"""
# ------------------------
# Standard library imports
# ------------------------
import sys
import os
import shutil
import uuid
import threading
import importlib
import importlib.util
import ast
import pickle
import time
import traceback
import zipfile
import logging
import logging.handlers
import collections
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Callable
from queue import Queue
from datetime import datetime
import datetime as dt

# ------------------------
# Third-party imports
# ------------------------
import numpy as np
import multiprocess

from qtpy import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QMainWindow,
    QFileDialog
)
from PyQt5.QtCore import pyqtSignal, QTimer, QProcessEnvironment
import PyQt5.uic
from PyQt5 import QtTest

try:
    import qdarktheme
except Exception:
    qdarktheme = None

try:
    import psutil
except Exception:
    psutil = None

# ------------------------
# QUDI imports
# ------------------------
import logic.misc as misc
from logic.generic_logic import GenericLogic
from core.connector import Connector
from core.util.mutex import Mutex

import logic.ODMR_nops as odmr
importlib.reload(odmr)

from logic.qudip_enhanced import *

# ------------------------
# Configuration
# ------------------------
QUEUE_DIR = Path.cwd() / "queue"
QUEUE_DIR.mkdir(exist_ok=True)
# Root folder that should contain top-level packages like 'notebooks'
# Adjust as needed for your environment:
SCRIPT_ROOT_FOLDER = r"C:\src\qudi"

# ------------------------
# Types
# ------------------------
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
    thread: Optional[threading.Thread] = None
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
# Stream redirector for embedded runs (captures prints & errors)
# ---------------------------
import io


class StreamRedirect(io.StringIO):
    """Redirect stdout/stderr to the real console only, no GUI signals."""
    def write(self, msg):
        if msg:
            try:
                sys.__stdout__.write(msg)
                sys.__stdout__.flush()
            except Exception:
                pass
        return len(msg)
    
    def flush(self):
        try:
            sys.__stdout__.flush()
        except Exception:
            pass


# ---------------------------
# ScriptWorker (embedded) - Pure Python thread with optional GUI-capture trampoline
# ---------------------------
class ScriptWorker(threading.Thread):
    """
    Run embedded tasks inside a regular Python thread (not QThread).
    If owner.capture_gui_mode is True, intercept PyQt5.uic.loadUi calls and execute them in the main thread.
    """

    def __init__(self,
                 task: ScriptTask,
                 signals: 'QueueManagerSignals',
                 on_finished: Callable[[ScriptTask], None],
                 owner_queue_logic,   # the queue_logic instance (passed as 'queue' to run_fun)
                 root_folder: Optional[str] = None):
        super().__init__(daemon=True)
        self.task = task
        self.signals = signals
        self.on_finished = on_finished
        self.owner = owner_queue_logic
        self._abort_event = threading.Event()
        self.root_folder = root_folder or SCRIPT_ROOT_FOLDER
        # store an original reference to uic.loadUi if needed
        self._orig_loadUi = getattr(PyQt5.uic, "loadUi", None)

    # convenience wrappers to schedule emissions on the Qt main thread
    def _emit_log(self, msg: str):
        try:
            QtCore.QTimer.singleShot(0, lambda m=msg: self.signals.log_output.emit(m))
        except Exception:
            try:
                self.signals.log_output.emit(msg)
            except Exception:
                pass

    def _emit_task_updated(self, task: ScriptTask):
        try:
            QtCore.QTimer.singleShot(0, lambda t=task: self.signals.task_updated.emit(t))
        except Exception:
            try:
                self.signals.task_updated.emit(task)
            except Exception:
                pass

    def _call_on_finished(self, task: ScriptTask):
        # schedule the manager's finish handler on the main thread
        try:
            QtCore.QTimer.singleShot(0, lambda t=task: self.on_finished(t))
        except Exception:
            try:
                self.on_finished(task)
            except Exception:
                pass

    def _install_uic_trampoline(self):
        """
        Replace PyQt5.uic.loadUi with a wrapper that executes the real loadUi on the main Qt thread.
        The wrapper is blocking from the worker side (it waits until the main-thread load completes).
        """
        orig = self._orig_loadUi
        if orig is None:
            return  # nothing to patch

        owner = self.owner

        def loadUi_trampoline(uifile, baseinstance=None, resource_suffix=None, *args, **kwargs):
            """
            Schedules the real loadUi(uifile, baseinstance, resource_suffix) to run in the main Qt thread.
            Blocks this worker until completion and re-raises exceptions if they occur in main thread.
            """
            # If no QApplication instance, can't trampoline — fallback to direct call (will likely fail)
            app = QtWidgets.QApplication.instance()
            if app is None:
                # no app yet; call directly (original behavior)
                return orig(uifile, baseinstance, resource_suffix, *args, **kwargs)

            result = {}
            done = threading.Event()

            def target():
                try:
                    # execute original loadUi (this runs in main thread)
                    res = orig(uifile, baseinstance, resource_suffix, *args, **kwargs)
                    result['res'] = res
                except Exception:
                    result['exc'] = traceback.format_exc()
                finally:
                    done.set()

            # schedule in main thread
            QtCore.QTimer.singleShot(0, target)
            # wait for main thread to finish loading
            done.wait()
            if 'exc' in result:
                # re-raise in worker context so traceback is visible
                raise RuntimeError(result['exc'])
            return result.get('res', None)

        # Monkey-patch
        PyQt5.uic.loadUi = loadUi_trampoline
        self._emit_log("[QueueLogic] Installed uic.loadUi trampoline (main-thread UI load)\n")

    def _restore_uic(self):
        try:
            if self._orig_loadUi is not None:
                PyQt5.uic.loadUi = self._orig_loadUi
                self._emit_log("[QueueLogic] Restored original uic.loadUi\n")
        except Exception:
            pass

    def run(self):
        """Load the queued module and run embedded function or class in a pure Python thread."""
        import sys

        # Redirect stdout/stderr for the duration of the script run
        stdout_backup = sys.stdout
        stderr_backup = sys.stderr
        sys.stdout = StreamRedirect()
        sys.stderr = StreamRedirect()

        self._emit_log(f"[QueueLogic] sys.meta_path={sys.meta_path}\n")
        self._emit_log(f"[QueueLogic] loading script: {self.task.queued_path}\n")
        print(f"[QueueLogic] loading script: {self.task.queued_path}")

        path = str(self.task.queued_path)
        name = "task_" + self.task.id

        # prepare environ and cwd
        try:
            root_folder = Path(self.root_folder).resolve()
            if str(root_folder) not in sys.path:
                sys.path.insert(0, str(root_folder))
            old_cwd = Path.cwd()
            os.chdir(Path(path).parent)
        except Exception:
            old_cwd = None

        # If owner requests GUI capture mode, install trampoline BEFORE importing module
        installed_trampoline = False
        try:
            if getattr(self.owner, "capture_gui_mode", False):
                # install trampoline for PyQt5.uic.loadUi
                self._install_uic_trampoline()
                installed_trampoline = True
        except Exception:
            # log and continue without trampoline
            self._emit_log("[QueueLogic] Failed to install GUI trampoline; proceeding without it.\n")
            installed_trampoline = False

        # Load module dynamically
        mod = None
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception as e:
            tb = traceback.format_exc()
            self._emit_log(f"[ERROR] Failed to load module: {e}\n{tb}\n")
            print(f"[ERROR] Failed to load module: {e}\n{tb}\n", file=sys.__stderr__)
            # cleanup
            try:
                sys.stdout = stdout_backup
                sys.stderr = stderr_backup
            except Exception:
                pass
            if old_cwd:
                try:
                    os.chdir(old_cwd)
                except Exception:
                    pass
            # restore uic if patched
            if installed_trampoline:
                self._restore_uic()
            self.task.status = Status.ERROR
            self._emit_task_updated(self.task)
            self._call_on_finished(self.task)
            return

        # Determine run target: top-level run_fun or class-based run_fun
        run_target = None
        if hasattr(mod, 'run_fun') and callable(getattr(mod, 'run_fun')):
            run_target = getattr(mod, 'run_fun')
            self._emit_log("[QueueLogic] Found top-level run_fun().\n")
            print("[QueueLogic] Found top-level run_fun().")
        else:
            # Fallback to original class-based logic
            for attr_name in dir(mod):
                try:
                    attr = getattr(mod, attr_name)
                except Exception:
                    continue
                if isinstance(attr, type) and hasattr(attr, "run_fun"):
                    instance = attr()
                    run_target = instance.run_fun
                    # keep class reference for pause/resume/stop hooks
                    self.task.worker = instance
                    self._emit_log(f"[QueueLogic] Found class-based run_fun() in {attr_name}.\n")
                    print(f"[QueueLogic] Found class-based run_fun() in {attr_name}.")
                    break

        if not run_target:
            self._emit_log("[QueueLogic] No run_fun found; treating as external.\n")
            print("[QueueLogic] No run_fun found; treating as external.")
            # cleanup
            try:
                sys.stdout = stdout_backup
                sys.stderr = stderr_backup
            except Exception:
                pass
            if old_cwd:
                try:
                    os.chdir(old_cwd)
                except Exception:
                    pass
            if installed_trampoline:
                self._restore_uic()
            self.task.status = Status.ERROR
            self._emit_task_updated(self.task)
            self._call_on_finished(self.task)
            return

        # --- Found run_fun; execute it inside this Python thread ---
        self.task.status = Status.RUNNING
        self.task.start_time = datetime.now()
        self._emit_task_updated(self.task)

        self._emit_log("[QueueLogic] run_fun starting...\n")
        print("[QueueLogic] run_fun starting...")

        try:
            # IMPORTANT: per requirement, pass abort and the queue_logic instance as 'queue'
            # run_fun expects signature like run_fun(abort, **kwargs)
            # We pass abort=self._abort_event and queue=self.owner
            try:
                run_target(abort=self._abort_event, queue=self.owner)
            except TypeError:
                # fallback: maybe script expects positional args
                run_target(self._abort_event, queue=self.owner)
        except Exception as e:
            tb = traceback.format_exc()
            self._emit_log(f"[ERROR] Exception in run_fun: {e}\n{tb}\n")
            print(f"[ERROR] Exception in run_fun: {e}\n{tb}\n", file=sys.__stderr__)
            self.task.status = Status.ERROR
        finally:
            # restore stdout/stderr and cwd
            try:
                sys.stdout = stdout_backup
                sys.stderr = stderr_backup
            except Exception:
                pass
            if old_cwd:
                try:
                    os.chdir(old_cwd)
                except Exception:
                    pass

            # restore uic if patched
            if installed_trampoline:
                self._restore_uic()

            # finalize task state
            self.task.end_time = datetime.now()
            if self.task.start_time:
                self.task.accumulated_time += (self.task.end_time - self.task.start_time).total_seconds()
            if self.task.status not in (Status.ABORTED, Status.ERROR):
                self.task.status = Status.COMPLETED

            # emit update and call finish handler on main thread
            self._emit_task_updated(self.task)
            self._emit_log(f"[QueueLogic] run_fun finished for {self.task.id}\n")
            self._call_on_finished(self.task)

    # Public interface to request stop from main thread
    def stop(self):
        """Request stop from main thread (sets abort flag) and call worker.stop() if present."""
        self._abort_event.set()
        if self.task.worker and hasattr(self.task.worker, "stop"):
            try:
                self.task.worker.stop()
            except Exception:
                pass

    # Expose is_set helper
    def abort_is_set(self) -> bool:
        return self._abort_event.is_set()


# ---------------------------
# QueueManager with signal proxy
# ---------------------------
class QueueManagerSignals(QtCore.QObject):
    task_added = QtCore.Signal(object)
    task_updated = QtCore.Signal(object)
    log_output = QtCore.Signal(str)


class queue_logic(GenericLogic):
    """Queue manager (not a QObject) — uses .signals (a QObject) to emit signals."""

    # connectors (unchanged)
    mcas_holder = Connector(interface='McasDictHolderInterface')
    transition_tracker = Connector(interface='TransitionTracker')
    confocal = Connector('ConfocalLogic')
    gated_counter = Connector('GatedCounter')
    optimizer = Connector('OptimizerLogic')
    fastcounter = Connector(interface='TimeTaggerInterface')
    PLE_logic = Connector("LaserScannerLogic")
    odmr_logic = Connector("ODMRLogic_holder")
    currentmeasurementlogic = Connector("CurrentMeasurementLogic")
    poimanagerlogic = Connector('PoiManagerLogic')
    powerstabilization_logic = Connector("PowerStabilizationLogic")
    counterlogic1 = Connector("CounterLogic")

    guis = []
    _StopTimeout = 60.
    __TIME_FORMAT_STR__ = '%Y%m%d-h%Hm%Ms%S'
    # default: try to capture GUI loads and run them in main thread
    capture_gui_mode = False

    def on_activate(self):
        # initialize signals/tasks and connectors
        self.signals = QueueManagerSignals()
        self.tasks: List[ScriptTask] = []
        self._awg = self.mcas_holder()
        self._transition_tracker = self.transition_tracker()
        self._gated_counter = self.gated_counter()
        self._optimizer = self.optimizer()
        self._PLE_logic = self.PLE_logic()
        self._ODMR_logic = self.odmr_logic()
        self._currentmeasurementlogic = self.currentmeasurementlogic()
        self._powerstabilization_logic = self.powerstabilization_logic()
        self._poimanagerlogic = self.poimanagerlogic()
        self._counter = self.counterlogic1()
        self._fast_counter_device = self.fastcounter()
        self.create_odmr()
        self._confocal = self.confocal()
        self.tt = self._transition_tracker

    def on_deactivate(self):
        pass

    @property
    def script_queue(self):
        return getattr(self, "_script_queue", None)

    @property
    def nowstr(self):
        return datetime.now().strftime('%Y%m%d-h%Hm%Ms%S')

    @property
    def nowstr_colon(self):
        return datetime.now().strftime('%Y/%m/%d-%H:%M:%S')

    @property
    def nowstr_pd(self):
        return datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

    @property
    def awgs(self):
        return self._mcas_dict.awgs

    def create_odmr(self):
        self.odmr = odmr.ODMR(self)

    def set_stop_request(self):
        for t in self.tasks:
            if t.is_embedded and getattr(t, "thread", None):
                try:
                    t.thread.stop()
                except Exception:
                    pass

    @property
    def cun(self) -> Optional[ScriptTask]:
        for t in self.tasks:
            if t.status == Status.RUNNING:
                return getattr(t, "nuclear", None)
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
                                                 'notebooks'
                                                 ]
                             ]
                            )
                ) or root.endswith('transition_tracker_log') or root.endswith('helpers'):
                    for file in files:
                        if any([file.endswith(i) for i in ['.py', '.dat', '.ui', '.hdf']]):
                            zf.write(os.path.join(root, file),
                                     os.path.join(root.replace(os.path.commonprefix([root, src]), ""), file))
            zf.close()

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
        """
        Add a script to the queue. The script is copied immediately into a
        per-task directory that preserves the original filename.
        """

        unique_id = uuid.uuid4().hex[:8]
        task_dir = QUEUE_DIR / unique_id
        task_dir.mkdir(parents=True, exist_ok=True)

        dest = task_dir / source_path.name
        shutil.copy2(source_path, dest)

        # --- Prepend original path as comment ---
        with open(dest, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Handle shebang if present
        if lines and lines[0].startswith('#!'):
            lines = [lines[0]] + [f"# Original file path: {source_path.resolve()}\n"] + lines[1:]
        else:
            lines = [f"# Original file path: {source_path.resolve()}\n"] + lines

        with open(dest, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        # --- End prepend ---

        task = ScriptTask(
            id=unique_id,
            original_path=source_path,
            queued_path=dest
        )
        self.tasks.append(task)

        # Logging
        self.signals.task_added.emit(task)
        self.signals.log_output.emit(f"Added {source_path.name} → {unique_id}/{source_path.name}\n")

        if not self.any_active_running():
            QtCore.QTimer.singleShot(100, self.start_next)

        return task

    def any_active_running(self):
        return any(t.status == Status.RUNNING for t in self.tasks)

    def _detect_embedded_task(self, path: Path):
        """Detect if the script defines a function or class with a `run_fun()` method using AST."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source, filename=str(path))
            for node in ast.walk(tree):
                # Check for top-level function run_fun()
                if isinstance(node, ast.FunctionDef) and node.name == "run_fun":
                    return True
                # Also check for class method run_fun()
                if isinstance(node, ast.ClassDef):
                    for child in node.body:
                        if isinstance(child, ast.FunctionDef) and child.name == "run_fun":
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
        """
        Start an embedded task using a pure Python Thread (ScriptWorker).
        The worker receives the queue_logic instance via owner_queue_logic and the abort flag is controllable via task.thread.stop()
        """
        worker = ScriptWorker(task=task,
                              signals=self.signals,
                              on_finished=self._on_thread_finished,
                              owner_queue_logic=self,
                              root_folder=SCRIPT_ROOT_FOLDER)
        task.is_embedded = True
        task.thread = worker
        task.worker = getattr(task, "worker", None)  # keep existing worker placeholder if set later

        # Start the worker thread
        worker.start()

        task.status = Status.RUNNING
        task.start_time = datetime.now()
        self.signals.task_updated.emit(task)
        self.signals.log_output.emit(f"Started embedded {task.id}\n")

    def _on_thread_finished(self, task: ScriptTask):
        thr = task.thread
        # attempt to join if a thread exists (but don't block GUI)
        if thr is not None:
            try:
                thr.join(timeout=0.2)
            except Exception:
                pass
            task.thread = None

        task.status = Status.COMPLETED if task.status not in (Status.ABORTED, Status.ERROR) else task.status
        task.end_time = datetime.now()
        if task.start_time:
            task.accumulated_time += (task.end_time - task.start_time).total_seconds()

        self.signals.task_updated.emit(task)
        self.signals.log_output.emit(f"Thread task {task.id} finished.\n")
        QtCore.QTimer.singleShot(100, self.start_next)

    def _start_process(self, task: ScriptTask):
        # Start external python process; set PYTHONPATH and working directory so imports & relative paths work
        proc = QtCore.QProcess()
        proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)

        # Prepare environment (prepend SCRIPT_ROOT_FOLDER to PYTHONPATH)
        env = QProcessEnvironment.systemEnvironment()
        existing = env.value("PYTHONPATH", "")
        sep = ";" if os.name == "nt" else ":"
        new_pythonpath = str(Path(SCRIPT_ROOT_FOLDER)) + (sep + existing if existing else "")
        env.insert("PYTHONPATH", new_pythonpath)
        proc.setProcessEnvironment(env)

        # Ensure process runs with working directory set to the task folder (so __file__ and relative paths work)
        proc.setWorkingDirectory(str(task.queued_path.parent))

        proc.readyReadStandardOutput.connect(lambda: self._on_proc_stdout(proc, task))
        proc.finished.connect(lambda exitCode, exitStatus: self._on_finished(proc, task, exitCode, exitStatus))
        proc.start(sys.executable, [str(task.queued_path)])

        task.process = proc
        task.status = Status.RUNNING
        task.start_time = datetime.now()
        self.signals.task_updated.emit(task)
        self.signals.log_output.emit(f"Started {task.queued_path.name}\n")
        print(f"Started {task.queued_path.name}")

    def _on_proc_stdout(self, proc, task):
        try:
            data = proc.readAllStandardOutput().data().decode(errors="ignore")
        except Exception:
            data = ""
        if data:
            task.stdout += data
            # emit to GUI and also print to console
            self.signals.log_output.emit(data)
            print(data, end='')

    def _on_finished(self, proc, task, exitCode, exitStatus):
        if task.status != Status.ABORTED:
            task.status = Status.COMPLETED
        task.end_time = datetime.now()
        if task.start_time:
            task.accumulated_time += (task.end_time - task.start_time).total_seconds()
        self.signals.log_output.emit(f"Finished {task.id} ({task.status.value})\n")
        self.signals.task_updated.emit(task)
        # Optionally keep the copied script or move it to an archive. For now, remove the script copy:
        try:
            if task.queued_path.exists():
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
            self.signals.log_output.emit(f"Paused embedded {task.id}\n")
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
            self.signals.log_output.emit(f"Paused PID {pid}\n")
            self.signals.task_updated.emit(task)
        except Exception as e:
            self.signals.log_output.emit(f"Failed to pause: {e}\n")

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
            self.signals.log_output.emit(f"Resumed embedded {task.id}\n")
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
            self.signals.log_output.emit(f"Resumed PID {pid}\n")
            self.signals.task_updated.emit(task)
        except Exception as e:
            self.signals.log_output.emit(f"Failed to resume: {e}\n")

    def abort_task(self, task_id: str):
        task = self._find_task(task_id)
        if not task:
            return
        task.status = Status.ABORTED
        self.signals.task_updated.emit(task)

        try:
            if task.is_embedded and task.thread:
                # thread.stop() triggers abort flag and worker.stop() if available
                try:
                    task.thread.stop()
                except Exception:
                    pass
            elif task.process:
                task.process.kill()
            task.end_time = datetime.now()
            self.signals.log_output.emit(f"Aborted {task.id}\n")
        except Exception as e:
            self.signals.log_output.emit(f"Failed to abort: {e}\n")

    def _find_task(self, task_id: str) -> Optional[ScriptTask]:
        for t in self.tasks:
            if t.id == task_id:
                return t
        return None
