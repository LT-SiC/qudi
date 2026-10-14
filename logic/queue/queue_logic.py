# coding=utf-8
from __future__ import print_function, absolute_import, division
__metaclass__ = type

# ----------------------------
# Original imports (kept)
# ----------------------------
import sys, os
import logic.misc
import imp
# GUI import kept but gui runs differently; no GUI code changed here
from gui.queue.Queue import queue_gui

from queue import Queue

import zipfile

from PyQt5.QtWidgets import QAbstractItemView, QMainWindow, QFileDialog
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSignal, QObject
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
import shutil

import collections
import importlib
import logic.ODMR_nops as odmr; importlib.reload(odmr)
from core.util.mutex import Mutex

from logic.currentmeasurement.current_measurement import CurrentMeasurementLogic
# from logic.biaslogic import BiasLogic

# ----------------------------
# Additions for GUI adapter
# ----------------------------
import uuid

# Lightweight Status enum for GUI
try:
    # Python 3 Enum exists
    from enum import Enum
except Exception:
    Enum = object

class Status(Enum):
    QUEUED = "Queued"
    RUNNING = "Running"
    PAUSED = "Paused"
    COMPLETED = "Completed"
    ERROR = "Error"
    ABORTED = "Aborted"

# @dataclass
# class ScriptTask:
#     id: str
#     original_path: Path
#     queued_path: Path
#     status: Status = Status.QUEUED
#     process: Optional[QtCore.QProcess] = None
#     thread: Optional[threading.Thread] = None
#     worker: Optional[object] = None
#     pid: Optional[int] = None
#     stdout: str = ""
#     stderr: str = ""
#     start_time: Optional[datetime] = None
#     end_time: Optional[datetime] = None
#     accumulated_time: float = 0.0
#     pause_start: Optional[datetime] = None
#     is_embedded: bool = False

# Task representation used for GUI signals
class ScriptTask:
    def __init__(self, id, module_name, pd, queued_time=None,original_path="",queued_path=""):
        self.id = id
        self.module_name = module_name
        self.pd = pd or {}
        self.status = Status.QUEUED
        self.start_time = None
        self.end_time = None
        self.accumulated_time = 0.0
        self.queued_time = queued_time or datetime.datetime.now()
        self.is_embedded = True
        self.original_path=original_path
        self.queued_path=queued_path

    def __repr__(self):
        return "<ScriptTask id={} module={} status={}>".format(self.id, self.module_name, getattr(self.status, "value", self.status))

# QObject wrapper for signals to avoid metaclass issues
class QueueSignals(QObject):
    task_added = pyqtSignal(object)    # emits ScriptTask
    task_updated = pyqtSignal(object)  # emits ScriptTask
    log_output = pyqtSignal(str)       # emits log text

# ----------------------------
# Original ScriptQueueStep & ScriptQueueList (kept as-is)
# ----------------------------
class ScriptQueueStep:
    def __init__(self, name, pd):
        self.name = name
        self.pd = pd


class ScriptQueueList(collections.MutableSequence):
    def __init__(self, oktypes, list_owner, *args):
        self.oktypes = oktypes
        self.list_owner = list_owner
        self._list = list()
        self.extend(list(args))

    @property
    def list(self):
        return self._list

    @list.setter
    def list(self, val):
        """
        inefficient, calls script_queue_changed multiple times. In practice not relevant.
        """
        self._list = []
        if len(val) == 0:
            self.list_owner.script_queue_changed()
        else:
            try:
                for i in val:
                    self.append(i)
            except Exception:
                self._list = []
                self.list_owner.script_queue_changed()
                exc_type, exc_value, exc_tb = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_tb)

    def check(self, val):
        if not isinstance(val, self.oktypes):
            raise TypeError("list item {} is not allowed, as it can not be found in {}".format(val, self.oktypes))

    # def check_duplicate(self, v):
    #     duplicates = [item.name for item in self.list if item.name == v.name]
    #     if len(duplicates) > 0:
    #         raise Exception('Error: {}, {}, {}'.format(duplicates, self.list, v))

    def set_parent(self, v):
        v.parent = self.list_owner

    def __len__(self):
        return len(self.list)

    def __getitem__(self, i):
        return self.list[i]

    def __delitem__(self, i):
        del self.list[i]
        self.list_owner.script_queue_changed()

    def __setitem__(self, i, v):
        raise NotImplementedError
        # self.check(v)
        # self.check_duplicate(v)
        # self.list[i] = v
        # self.list_owner.script_queue_changed(i, v)

    def insert(self, i, v):
        if i != len(self.list):
            raise Exception('Only appending and popping items allowed')
        self.check(v)
        # self.check_duplicate(v)
        self.list.insert(i, v)
        self.list_owner.script_queue_changed()

    def __str__(self):
        return str(self.list)

    def __repr__(self):
        return str(self.list)


# ----------------------------
# Main queue_logic (original preserved, with adapter additions)
# ----------------------------
class queue_logic(GenericLogic):

    # declare connections
    mcas_holder = Connector(interface='McasDictHolderInterface')
    transition_tracker = Connector(interface = 'TransitionTracker') # Should be a name of the class
    confocal = Connector('ConfocalLogic')
    gated_counter = Connector('GatedCounter') # Should be name of the class.
    optimizer= Connector('OptimizerLogic')
    fastcounter = Connector(interface='TimeTaggerInterface')
    PLE_logic= Connector("LaserScannerLogic")
    odmr_logic= Connector("ODMRLogic_holder")
    currentmeasurementlogic= Connector("CurrentMeasurementLogic")
    # biaslogic= Connector("BiasLogic")
    poimanagerlogic = Connector('PoiManagerLogic')
    powerstabilization_logic = Connector("PowerStabilizationLogic")
    counterlogic1=Connector("CounterLogic")


    update_selected_user_script_combo_box_signal = pyqtSignal(collections.OrderedDict)
    update_queue_list = pyqtSignal(collections.OrderedDict)
    user_script_list = misc.ret_property_array_like_typ('user_script_list', str)
    guis = []  #stores names of all open guis (later on used to dump them periodically)
    _StopTimeout = 60.

    __TIME_FORMAT_STR__ = '%Y%m%d-h%Hm%Ms%S'

    # smiq_visa_device = 'GPIB0::28::INSTR'
    # app_dir = r'D:/Python/pi3diamond'
    # log_dir = '{}/log/'.format(app_dir)
    # log_archive_dir = '{}/log/archive/'.format(app_dir)
    # log_single_val_dir = '{}/log/single_values/'.format(app_dir)
    # log_tmp = '{}/log/temp/'.format(app_dir)

    def __init__(self, config , **kwargs):
        super(queue_logic, self).__init__(config=config, **kwargs)
        self._threadlock=Mutex()
        self.script_history = []

        # ----------------------------
        # Adapter state for new GUI
        # ----------------------------
        self.signals = QueueSignals()   # exposes task_added, task_updated, log_output
        self.tasks = []                 # list[ScriptTask] visible to GUI
        self._task_by_id = {}           # id -> ScriptTask
        self._task_by_module = {}       # module_name -> ScriptTask (module names are unique in this design)
        self._monitor_thread = None
        self._monitor_stop_event = threading.Event()
        self._last_script_history_len = 0

    def on_activate(self):

        self._awg = self.mcas_holder()
        self._transition_tracker = self.transition_tracker()
        self._gated_counter = self.gated_counter()
        self._optimizer = self.optimizer()
        self._PLE_logic = self.PLE_logic()
        self._ODMR_logic = self.odmr_logic()
        self._currentmeasurementlogic:CurrentMeasurementLogic = self.currentmeasurementlogic()
        # self._biaslogic:BiasLogic = self.biaslogic()
        self._powerstabilization_logic = self.powerstabilization_logic()
        self._poimanagerlogic = self.poimanagerlogic()
        self._counter=self.counterlogic1()
        self._fast_counter_device = self.fastcounter()  # FIXME
        self.create_odmr()  #only logic (no gui)
        self.init_run()
        self.write_standard_awg_sequences()
        self._confocal = self.confocal()
        self.tt = self._transition_tracker

    def on_deactivate(self):
        # stop task monitor thread gracefully if active
        try:
            self._monitor_stop_event.set()
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=1.0)
        except Exception:
            pass
        # pass
        #FIXME destroy me gently

    @property
    def md(self):
        return self._mcas_dict#

    @property
    def gui(self):
        return self._gui

    # def restart_timetagger(self):
    #     import TimeTaggerHandler
    #     reload(TimeTaggerHandler)
    #     self._timetagger = TimeTaggerHandler.init_timetagger()

    def init_run(self):
        self.user_script_folder = r"C:/src/qudi/notebooks/UserScripts/electron_t2"
        self._script_queue = ScriptQueueList(oktypes=(ScriptQueueStep), list_owner=self)
        self.q = Queue() # use connector
        self.run_thread()

        # start the monitor thread that observes current_script/script_history
        # This allows emitting task_updated events without changing run()
        try:
            self._monitor_stop_event.clear()
            self._monitor_thread = threading.Thread(target=self._monitor_tasks_loop, name="QueueLogicMonitor", daemon=True)
            self._monitor_thread.start()
        except Exception:
            # non-fatal; continue
            pass

        # self.track_memory_usage_thread()

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

    ####################################################################################################################
    # dump and restore
    ####################################################################################################################

    def persistent_file_name(self, model):
        if hasattr(model, 'pi3d_dump_filename'):
            return self.log_dir + model.pi3d_dump_filename
        return self.log_dir + str(model.__class__).replace('>', '').replace('<',

                                                                            '') + '.pyd'  # windows does not allow '>' and '<' in filenames
    #
    def restore(self, model, fp=None):
        filename = self.persistent_file_name(model) if fp is None else fp

        if os.access(filename, os.F_OK):

            self.log.info('Restoring state of ' + model.__str__() + '\nfrom ' + filename + '..')

            try:
                a = pickle.load(open(filename, 'rb'))
                a = a if type(a) is dict else a.__getstate__()
                model.set_items(a)
                self.log.info('[DONE1]')

            except Exception as inst:
                self.log.exception(str(inst))
                self.log.warning('[FAILED]')
                raise inst

    def dump(self, model):
        filename = self.persistent_file_name(model)
        self.log.info('attempting to save state of ' + model.__str__() + '\nto ' + filename + '..', )
        try:
            fil = open(filename, 'wb')
            pickle.dump(model.__getstate__(), fil)
            fil.close()
            self.log.info('[DONE]')
        except Exception:
            self.log.exception(str(Exception))
            self.log.warning('[FAILED]')

    # @property
    # def current_nuclear(self):
    #     l = [i for i in self.__dict__ if 'Nuclear' in i]
    #     df = pd.DataFrame({'attr_name': l, 'date': [datetime.datetime.strptime(i[-17:], '%Y%m%dh%Hm%Ms%S') for i in l]})
    #     return getattr(self, df[df['date'] == df['date'].max()].iloc[0, 0])

    @property
    def script_module_names(self):
        return [i for i in sys.modules if '__script__' in i]

    @property
    def last_running_script_name(self):
        sm = self.script_module_names
        if hasattr(self, 'current_script'):
            if not self.current_script['module_name'] in sm:
                raise Exception('Error: {}, {}'.format(sm, self.current_script))
            else:
                return self.current_script['module_name']
        else:
            return sm[-1]
    @property
    def cun_modules(self):
        lrs = self.last_running_script_name
        if hasattr(sys.modules[lrs], 'nuclear'):
            return sys.modules[lrs]
        else:
            for smi in self.script_module_names[::-1]:
                if hasattr(sys.modules[smi], 'nuclear'):
                    return sys.modules[smi]

    @property
    def cun(self):
        return self.cun_modules.nuclear

    def track_memory_usage(self):
        while True:
            self.save_value_to_file(self.current_memory_usage(), 'memory_mb')
            # CAREFUL WITH THREADING AND WRITING TO SAME HDF FILE self.save_values_hdf(classifier='memory_mb', vd=dict(none=self.current_memory_usage()))
            # possible solution: https://stackoverflow.com/questions/22522551/pandas-hdf5-as-a-database
            QtTest.QTest.qSleep(5000)#time.sleep(5)

    # def current_memory_usage(self):
    #     print(os.getpid())
    #     p = psutil.Process(os.getpid())
    #     return p.memory_info()[0] / 1024.
    #
    # def track_memory_usage_thread(self):
    #     self.tmu_thread = threading.Thread(target=self.track_memory_usage)
    #     self.tmu_thread.stop_request = multiprocess.Event()
    #     self.tmu_thread.start()

    def script_queue_changed(self):
        self.update_script_queue_table_data()

    @property
    def script_queue_table_data(self):
        return self._script_queue_table_data

    def update_script_queue_table_data(self):
        out = collections.OrderedDict([('name', []), ('pd', [])])
        for ridx, i in enumerate(self.script_queue):
            for cidx, attr_name in enumerate(['name', 'pd']):
                out[attr_name].append(getattr(i, attr_name))
        self._script_queue_table_data = out
        if hasattr(self, '_gui'): # this is the problem.
            self.gui.update_script_queue_table_data(self.script_queue_table_data)
        self.update_queue_list.emit(self.script_queue_table_data)

    ####################################################################################################################
    # script queue
    ####################################################################################################################
    def run(self):


        ## Why this is needed??????

        #from tools_2 import emod
        #emod.JobManager().start() ## maybe this is something which makes multiple sequences actuakly working.

        # start the CronDaemon
        #from tools_2 import cron
        #cron.CronDaemon().start()
        #self.dummy_test_variable = 123

        while True:
            if self.thread.stop_request.is_set():
                self.q.queue.clear()
                self.script_queue.list = []
                self.thread.stop_request.clear()
            try: ### runs the measurement!
                self.current_script = self.q.get()
                
                self.thread.stop_request.clear() # this is necessary although it shouldn't be.
                self.log.info("Starting Userscript {}...{}".format(
                    self.current_script['module_name'][10:],
                   self.thread.stop_request.is_set()))
                sys.modules[self.current_script['module_name']].run_fun(
                    self.thread.stop_request, queue = self, **self.current_script['pd']) ## Creates a nuclear and runs it.!!!
                print('entering waiting loop in queue...')
                if hasattr(self, 'cun'):
                    task = self._task_by_module[self.current_script['module_name']]
                    task.saved_dir=self.cun.save_dir

                    while self.cun.state == 'run':
                        QtTest.QTest.qSleep(1000)  #This is Qt version for time.sleep to prevent freezinng.
                    else:
                        print("new measurement can be started")
                else:
                    pass
                
                print('entering waiting loop in queue... 2')
                    
                self.script_history.append(self.current_script)
                self.script_queue.pop(0)
                self.log.info("Userscript {} has finished...".format(self.current_script['module_name'][10:]))
                del self.current_script
                self.q.task_done()
            except Exception: #Not running the measurement.
                self.q.queue.clear()
                self.script_queue.list = []
                exc_type, exc_value, exc_tb = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_tb)

    def run_thread(self):
        self.thread = threading.Thread(target=self.run)
        self.thread.stop_request = multiprocess.Event()
        self.thread.start()

    def add_to_queue(self, name=None, pd=None, folder=None):
        #self.confocal.counter_state = 'idle'
        #self.md.stop_awgs()
        folder = self.user_script_folder if folder is None else folder
        name = self.selected_user_script if name is None else name
        pd = self.user_script_params if pd is None else pd
        if name == '':
            return
        try:
            module_name = self.init_task(name, folder)
            self.q.put({'module_name': module_name, 'pd': pd})
            self.script_queue.append(ScriptQueueStep(module_name[10:], self.user_script_params))

            # ----------------------------
            # GUI adapter: create ScriptTask & emit task_added
            # ----------------------------
            try:
                tid = uuid.uuid4().hex
                task = ScriptTask(id=tid, module_name=module_name, pd=pd, queued_time=datetime.datetime.now(),original_path=folder,queued_path=name)
                # try:
                #     task.saved_dir=sys.modules[module_name].nuclear.save_dir
                # except Exception as e:
                #     print("error1: ",e)

                task.start_time=datetime.datetime.now()
                self.tasks.append(task)
                self._task_by_id[tid] = task
                self._task_by_module[module_name] = task
                # emit signal for GUI
                try:
                    self.signals.task_added.emit(task)
                except Exception:
                    # if emitting fails, fallback to log_output
                    try:
                        self.signals.log_output.emit("[QueueLogic] task_added emit failed for {}\n".format(module_name))
                    except Exception:
                        pass
            except Exception:
                pass

        except Exception:
            print("Queuelogic: Could not add script to queue.")
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    def set_user_script_list(self):
        file_list = []
        unwanted_files = ['__init__.py', 'refocus_confocal_odmr.py']
        for files in os.listdir(self.user_script_folder):
            if files.endswith(".py") and not files in unwanted_files:
                file_list.append(str(files.split('.')[0]))
        self._user_script_list = file_list
        if len(self.user_script_list) > 0:
            self._selected_user_script = self.user_script_list[0]
            #if hasattr(self, '_gui'):
            val = collections.OrderedDict([('user_script_list', self.user_script_list), ('selected_user_script', self.selected_user_script)])
            #self.update_selected_user_script_combo_box(val)
            #Instead emit a signal which will updates it.
            self.update_selected_user_script_combo_box_signal.emit(val)

    # Convenience wrapper for GUI: accept a Path and add it to queue
    def add_task(self, path):
        """
        GUI-friendly add: path is a pathlib.Path or string path to a .py file
        It extracts folder and module name and calls add_to_queue.
        """
        try:
            p = path if isinstance(path, str) else str(path)
            folder, base = os.path.split(p)
            name = os.path.splitext(base)[0]
            # Temporarily set user_script_folder so set_user_script_list doesn't override
            old_folder = getattr(self, '_user_script_folder', None)
            self.user_script_folder = folder
            # set selected user script if possible
            try:
                if name in self.user_script_list:
                    self.selected_user_script = name
            except Exception:
                # ignore
                pass
            # add to queue using existing routine
            self.add_to_queue(name=name, pd=self.user_script_params, folder=folder)
            # restore folder (keep it changed because GUI may expect the new folder)
            # self.user_script_folder = old_folder
        except Exception:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    @property
    def selected_user_script(self):
        return self._selected_user_script

    @selected_user_script.setter
    def selected_user_script(self, val):
        if val != '':
            if val not in self.user_script_list:
                raise Exception('Script {} not in {}'.format(val, self.user_script_list))
            self._selected_user_script = val
            #if hasattr(self, '_gui'):
            val = collections.OrderedDict([('selected_user_script', self.selected_user_script)])
            self.update_selected_user_script_combo_box_signal.emit(val)
            #self.gui.update_selected_user_script_combo_box(val)

    @property
    def user_script_params(self):
        return getattr(self, '_user_script_params', {})

    @user_script_params.setter
    def user_script_params(self, val):
        self.user_script_params = misc.check_type(val, 'user_script_params', dict)

    @property
    def user_script_folder(self):
        return self._user_script_folder

    @user_script_folder.setter
    def user_script_folder(self, val):
        if os.path.isdir(val):
            self._user_script_folder = val
            self.set_user_script_list()
            #This now is done in gui automatically.
            #if hasattr(self, '_gui'):
                #self.gui.update_user_script_folder_text_field(val)

    

    # Convenience wrapper for GUI: accept a Path and add it to queue
    def add_task(self, path):
        """
        GUI-friendly add: path is a pathlib.Path or string path to a .py file
        It extracts folder and module name and calls add_to_queue.
        """
        try:
            p = path if isinstance(path, str) else str(path)
            folder, base = os.path.split(p)
            name = os.path.splitext(base)[0]
            # Temporarily set user_script_folder so set_user_script_list doesn't override
            old_folder = getattr(self, '_user_script_folder', None)
            self.user_script_folder = folder
            # set selected user script if possible
            try:
                if name in self.user_script_list:
                    self.selected_user_script = name
            except Exception:
                # ignore
                pass
            # add to queue using existing routine
            self.add_to_queue(name=name, pd=self.user_script_params, folder=folder)
            # restore folder (keep it changed because GUI may expect the new folder)
            # self.user_script_folder = old_folder
        except Exception:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    def add_rco(self):
        folder = r"C:\src\qudi\notebooks\UserScripts"
        name = 'refocus_confocal_odmr'
        pd = self.user_script_params
        self.add_to_queue(name, pd, folder)

    def in_script_queue(self, name):
        for step in self.script_queue:
            if step.name == name:
                return True
        else:
            return False

    def evaluate(self):
        raise NotImplementedError
        # name = self.selected_user_script
        # pd = self.user_script_params
        # if self.in_script_queue(name):
        #     raise Exception("Running the 'evaluate' function of a script is only possible when its 'run_fun' is not running.")
        # else:
        #     self.get_user_script(name).evaluate(**pd)

    def remove_last_script(self):
        if self.q.qsize() > 0:
            del self.script_queue[-1]
            self.q.get()

    def create_odmr(self):

        self.odmr = odmr.ODMR(self)


    def init_task(self, name, folder=None):
        folder = self.user_script_folder if folder is None else folder
        funa = "{}/{}.py".format(folder, name)
        task_name = "__script__{}_{}".format(self.nowstr, name)
        _ = imp.load_source(task_name, funa)
        return task_name

    def set_stop_request(self):
        self.thread.stop_request.set()

    def write_standard_awg_sequences(self):
        self.add_to_queue('standard_awg_sequences', folder=r'C:\src\qudi\notebooks\UserScripts\helpers')

    def dl(self, key, *args, **kwargs):
        return self._mcas_dict[key].dl(*args, **kwargs)

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
                                                 'notebooks' #FIXME - include logic, explode heavy stuff.
                                                 ]
                             ]
                            )
                ) or root.endswith('transition_tracker_log') or root.endswith('helpers'):
                    for file in files:
                        if any([file.endswith(i) for i in ['.py', '.dat', '.ui', '.hdf']]):
                            zf.write(os.path.join(root, file), os.path.join(root.replace(os.path.commonprefix([root, src]), ""), file))
            zf.close()


    # ----------------------------
    # NEW: monitor loop that observes current_script and script_history
    # This lets us emit task_updated events without touching run()
    # ----------------------------
    def _monitor_tasks_loop(self):
        """
        Background thread: periodically checks self.current_script and self.script_history
        and updates task status accordingly. Runs as daemon.
        """
        try:
            poll_interval = 0.5
            while not self._monitor_stop_event.is_set():
                try:
                    # Mark running task if current_script exists
                    if hasattr(self, 'current_script') and self.current_script:
                        modname = self.current_script.get('module_name')
                        task = self._task_by_module.get(modname)
                        if task and task.status != Status.RUNNING:
                            #task.status = Status.RUNNING
                            #task.start_time = datetime.datetime.now()
                            try:
                                self.signals.task_updated.emit(task)
                            except Exception:
                                pass

                    # If script_history has grown -> mark the last appended as COMPLETED
                    hist_len = len(self.script_history) if hasattr(self, 'script_history') else 0
                    if hist_len != self._last_script_history_len:
                        # A change occurred: mark any newly finished tasks
                        # iterate over new entries
                        for idx in range(self._last_script_history_len, hist_len):
                            try:
                                item = self.script_history[idx]
                                if isinstance(item, dict):
                                    modname = item.get('module_name')
                                else:
                                    # fallback attempt
                                    modname = getattr(item, 'module_name', None)
                                if modname:
                                    task = self._task_by_module.get(modname)
                                    if task:
                                        task.status = Status.COMPLETED
                                        task.end_time = datetime.datetime.now()
                                        if task.start_time:
                                            task.accumulated_time = (task.end_time - task.start_time).total_seconds()
                                        try:
                                            self.signals.task_updated.emit(task)
                                        except Exception:
                                            pass
                            except Exception:
                                pass
                        self._last_script_history_len = hist_len

                    # If queue cleared unexpectedly: mark queued tasks as ABORTED when queue empty but tasks remain
                    try:
                        qsize = self.q.qsize() if hasattr(self, 'q') else 0
                        if qsize == 0:
                            # check for tasks still marked QUEUED
                            for task in list(self.tasks):
                                if task.status == Status.QUEUED:
                                    # if script_queue has no matching ScriptQueueStep, consider it removed
                                    has_in_script_queue = False
                                    try:
                                        for step in self.script_queue:
                                            # step.name holds module name without prefix in original impl:
                                            # original added ScriptQueueStep(module_name[10:], ...)
                                            # so compare using suffix of module_name
                                            if task.module_name.endswith(step.name):
                                                has_in_script_queue = True
                                                break
                                    except Exception:
                                        pass
                                    if not has_in_script_queue:
                                        task.status = Status.ABORTED
                                        try:
                                            self.signals.task_updated.emit(task)
                                        except Exception:
                                            pass
                    except Exception:
                        pass

                except Exception:
                    # guard against unexpected monitor exceptions
                    try:
                        self.signals.log_output.emit("[QueueLogic monitor] exception in monitor loop\n")
                    except Exception:
                        pass
                # sleep
                self._monitor_stop_event.wait(poll_interval)
        except Exception:
            # final guard
            try:
                self.signals.log_output.emit("[QueueLogic monitor] terminated due to exception\n")
            except Exception:
                pass


    def Delete_measurement(self,tid):
        #self.cun.save_dir
        tsk=self._find_task_by_id(tid)
        try:
            shutil.rmtree(tsk.saved_dir)
        except Exception as e:
            print("could not remove folder:", e)

    # ----------------------------
    # GUI control methods
    # ----------------------------
    def _find_task_by_id(self, tid):
        return self._task_by_id.get(tid)

    def _find_task_by_module(self, module_name):
        return self._task_by_module.get(module_name)

    def pause_task(self, task_id):
        """
        GUI requested pause: attempt to pause the running task if possible
        or mark queued task as PAUSED.
        This does not forcibly stop the underlying measurement unless it supports manual pause.
        """
        try:
            task = self._find_task_by_id(task_id)
            if not task:
                return
            if task.status == Status.RUNNING:
                # if running and nuclear has manual_pause attribute, set it
                try:
                    if hasattr(self, 'cun') and hasattr(self.cun, 'manual_pause'):
                        self.cun.manual_pause = True
                        task.status = Status.PAUSED
                        self.signals.task_updated.emit(task)
                        return
                except Exception:
                    pass
                # otherwise simply mark paused (the measurement may not actually pause)
                task.status = Status.PAUSED
                try:
                    self.signals.task_updated.emit(task)
                except Exception:
                    pass
            elif task.status == Status.QUEUED:
                task.status = Status.PAUSED
                try:
                    self.signals.task_updated.emit(task)
                except Exception:
                    pass
        except Exception:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    def start_task(self, task_or_obj):
        """
        GUI requested start: either resume a paused task, or force a queued task to the front of queue.
        The GUI passes either a ScriptTask or a task id (str).
        """
        try:
            if isinstance(task_or_obj, str):
                task = self._find_task_by_id(task_or_obj)
            else:
                task = task_or_obj
            if not task:
                return

            if task.status == Status.PAUSED:
                # resume: simply mark as queued and re-add to queue if necessary
                task.status = Status.QUEUED
                self.cun.manual_pause=False
                try:
                    self.signals.task_updated.emit(task)
                except Exception:
                    pass
                # ensure module_name present in q; if absent, re-add it
                present = False
                try:
                    for qitem in list(self.q.queue):
                        if qitem.get('module_name') == task.module_name:
                            present = True
                            break
                except Exception:
                    pass
                if not present:
                    # re-enqueue
                    try:
                        self.q.put({'module_name': task.module_name, 'pd': task.pd})
                        # restore script_queue entry as original code expects it
                        try:
                            # the original script_queue stores the short name (module_name[10:])
                            shortname = task.module_name[10:]
                            self.script_queue.append(ScriptQueueStep(shortname, task.pd))
                        except Exception:
                            pass
                    except Exception:
                        pass
            elif task.status == Status.QUEUED:
                # move to front by reconstructing queue: pop all, put the one first, then others
                try:
                    items = []
                    try:
                        while True:
                            items.append(self.q.get_nowait())
                    except Exception:
                        pass
                    # put selected task first then requeue others
                    new_items = []
                    found = False
                    for it in items:
                        if it.get('module_name') == task.module_name and not found:
                            new_items.append(it)
                            found = True
                        else:
                            new_items.append(it)
                    # push back
                    for it in new_items:
                        self.q.put(it)
                    # update status
                    task.status = Status.QUEUED
                    try:
                        self.signals.task_updated.emit(task)
                    except Exception:
                        pass
                except Exception:
                    pass
        except Exception:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    def abort_task(self, task_id):
        """
        GUI requested abort: try to remove from queue or request stop if running.
        This will attempt to remove queued entries and for running tasks set global stop request.
        """
        try:
            task = self._find_task_by_id(task_id)
            if not task:
                return

            self.set_stop_request()

            if task.status == Status.RUNNING:
                # request stop of the current running thread (this stops all scripts)
                try:
                    self.set_stop_request()
                except Exception:
                    pass
                task.status = Status.ABORTED
                try:
                    self.signals.task_updated.emit(task)
                except Exception:
                    pass
            elif task.status in (Status.QUEUED, Status.PAUSED):
                # remove matching entries from queue
                try:
                    items = []
                    try:
                        while True:
                            items.append(self.q.get_nowait())
                    except Exception:
                        pass
                    # requeue everything except the aborted task
                    for it in items:
                        if it.get('module_name') == task.module_name:
                            # skip this entry (effectively abort)
                            continue
                        self.q.put(it)
                    # remove any script_queue entries matching the short name
                    try:
                        shortname = task.module_name[10:]
                        # find and remove first matching ScriptQueueStep
                        for i, step in enumerate(list(self.script_queue.list)):
                            if step.name == shortname:
                                del self.script_queue.list[i]
                                break
                    except Exception:
                        pass
                    task.status = Status.ABORTED
                    try:
                        self.signals.task_updated.emit(task)
                    except Exception:
                        pass
                except Exception:
                    pass
        except Exception:
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)

    # End of class queue_logic
