# -*- coding: utf-8 -*-
"""
This file contains a Qudi logic module for controlling scans of the
fourth analog output channel.  It was originally written for
scanning laser frequency, but it can be used to control any parameter
in the experiment that is voltage controlled.  The hardware
range is typically -10 to +10 V.

Qudi is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Qudi is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Qudi. If not, see <http://www.gnu.org/licenses/>.

Copyright (c) the Qudi Developers. See the COPYRIGHT.txt file at the
top-level directory of this distribution and at <https://github.com/Ulm-IQO/qudi/>
"""

from cgitb import enable
from collections import OrderedDict
import datetime
import matplotlib.pyplot as plt
import numpy as np
import time
import os

from core.connector import Connector
from core.statusvariable import StatusVar
from core.util.mutex import Mutex
from logic import save_logic
from logic.generic_logic import GenericLogic
from qtpy import QtCore
from PyQt5 import QtTest
import logging; logger = logging.getLogger(__name__)

import base64
import hashlib

# for analysis:
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit

from core.configoption import ConfigOption

# from logic.laserscanner.ple_default_values_and_widget_functions import ple_default_values_and_widget_functions as ple_default
# from gui.laserscanner.connectors_and_set_default import initialize_connections_and_defaultvalue

# from logic.confocal_logic import ConfocalLogic
# from logic.save_logic import SaveLogic
# from logic.biaslogic import BiasLogic
# from logic.fit_logic import FitLogic
# from logic.TopticaDLCLogic import CTLLogic, DLCProLogic
# from logic.wavemeter_logger_logic import WavemeterLoggerLogic
# from hardware.national_instruments_x_series import NationalInstrumentsXSeries
from hardware.swabian_instruments.timetagger import TimeTaggerInterface
from TimeTagger import TimeTagStreamBuffer
from logic.save_logic import SaveLogic
# from qudi.util.generic_measurement import GenericMeasurement
import numpy as np
import time
import logging

import pyqtgraph as pg

logger = logging.getLogger(__name__)


# from interface.filtered_counts_interface import FilteredCounterInterface


class TimeHistLogic(GenericLogic):
    """
    Qudi-Messklasse zur Zählratenaufnahme mit Schwellenwert-Filterung
    und Histogramm-Erstellung aus TimeTagger Counterdaten.
    """
    counter_device = Connector(interface='TimeTaggerInterface')
    savelogic = Connector(interface='SaveLogic')

    hist_updated = QtCore.Signal()
    trace_updated = QtCore.Signal()


    # Parameter
    channel = StatusVar('channel', default=0)
    binwidth = StatusVar('binwidth', default=20_000_000) # in ps
    n_values = StatusVar('n_values', default=2_000_000)
    threshold = StatusVar('threshold', default=5)
    n_filter = StatusVar('n_filter', default=10)
    plot_length = StatusVar('plot_length', default=100_000)
    Filename = StatusVar('Filename', "Filename")


    def __init__(self,  **kwargs):
        super().__init__( **kwargs)

        self.threadlock = Mutex()


        # TimeTagger Objekte
        # self.counter = None
        self.stream = None
        self.buffer = None

        # Ergebnisdaten
        self.raw_data = np.array([0])
        self.t_raw_data = np.array([0])
        self.filtered_data = np.array([0])
        self.histogram = np.array([0])
        self.bin_edges = np.array([0, 1])
        self.histogram_raw = np.array([0])
        self.bin_edges_raw = np.array([0, 1])
        self.time_zero = None
        self.live_interval_ms = 500
        
        

    def on_activate(self):
        """ Initialisation performed during activation of the module.
        """
        self._time_tagger:TimeTaggerInterface=self.counter_device()
        self._save_logic:SaveLogic = self.savelogic()
        self._time_tagger.setup_TT()

        # Timer für Live-Modus
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update_live_data)
        self._live_running = False


    def on_deactivate(self):
        pass
    
    


    def setup_time_tagger(self,**kwargs):
        self._time_tagger._stream.update(**kwargs)
        # self._time_tagger._counter.update(**kwargs)
        return self._time_tagger.stream()
        # return self._time_tagger.counter()
    
    # def prepare(self):
    #     """Initialisierung von TimeTagger und Counter."""
    #     self.counter = self.setup_time_tagger(
    #             channels=[self.channel],
    #             binwidth=self.binwidth,
    #             n_values=self.n_values)
    #     logger.info("Counter initialisiert.")
    #     # self.counter.startFor(capture_duration=self.n_values * self.binwidth,clear=True)

    def prepare(self):
        """Initialisierung von TimeTagger und Stream."""
        self.stream = self.setup_time_tagger(
                channels=[self.channel],
                n_max_events = self.n_values)
        self.buffer = TimeTagStreamBuffer()
        self.buffer.size = self.n_values
        # self.stream.getData(self.buffer)
        self.timestamps = np.array([], dtype=np.int64)
        self.time_zero = None
        self.filtered_data = np.array([], dtype=np.int64)
        self.histogram = np.array([0])
        self.bin_edges = np.array([0, 1])
        self.histogram_raw = np.array([0])
        self.bin_edges_raw = np.array([0, 1])
        logger.info("TimeTagStream initialisiert.")

    def start_live(self, new):
        if new or self.stream is None:
            self.prepare()
        self._live_running = True
        QtCore.QTimer.singleShot(0, self._start_timer_safe)

    def _start_timer_safe(self):
        # Diese Methode wird garantiert im GUI-Thread ausgeführt
        self.timer.start(self.live_interval_ms)
        logger.info("Live-Messung gestartet.")

    def stop_live(self):
        self._live_running = False
        QtCore.QTimer.singleShot(0, self._stop_timer_safe)

    def _stop_timer_safe(self):
        self.timer.stop()
        logger.info("Live-Messung gestoppt.")

    def run(self):
        pass
        # """Messung durchführen, Daten filtern, Histogramm erstellen."""
        # wait_time = (self.n_values * self.binwidth) / 1e12 + 1  # in Sekunden

        

        # logger.info(f"Warte {wait_time:.1f} s auf Daten...")

        # start = time.time()
        # while time.time() < start+wait_time*3:
        #     QtCore.QCoreApplication.processEvents()
        #     time.sleep(0.01)  # kleines Intervall, blockiert nicht zu hart
        #     self.raw_data = self.counter.getData()[0]
        #     self._apply_filter_and_histogram()
        
        # # QtTest.QTest.qSleep(500) 
        # self.raw_data = self.counter.getData()[0]
        # self._apply_filter_and_histogram()
        
        # # self.run()

    # def _apply_filter_and_histogram(self):
    #     """Filtert und erstellt Histogramm basierend auf aktuellen Parametern."""
    #     if self.raw_data is None:
    #         logger.warning("Keine Rohdaten vorhanden.")
    #         return

    #     mask = np.array([
    #         np.any(self.raw_data[max(0, i - self.n_filter):i] > self.threshold)
    #         for i in range(len(self.raw_data))
    #     ])
    #     self.filtered_data = self.raw_data[mask]

    #     self.histogram, self.bin_edges = np.histogram(
    #         self.filtered_data,
    #         bins=np.arange(0, np.max(self.filtered_data) + 2, 1)
    #     )
    #     self.histogram_raw, self.bin_edges_raw = np.histogram(
    #         self.raw_data,
    #         bins=np.arange(0, np.max(self.raw_data) + 2, 1)
    #     )

    #     self.hist_updated.emit()
    #     self.trace_updated.emit()


    def _update_live_data(self):
        if not self._live_running:
            return

        # buffer = TimeTagStreamBuffer()
        self.stream.getData(self.buffer)
        tags = np.array(self.buffer.getTimestamps(), dtype=np.int64)
        # print(tags)

        # tags = self.stream.getData()[self.channel]  # ns-Zeitstempel (int64)
        if tags is None or len(tags) < 2:
            return
        if self.time_zero == None:
            self.time_zero = tags[0]

        tags -= self.time_zero

        # Anfügen an gespeicherte Zeitstempel
        self.timestamps = np.concatenate([self.timestamps, tags])


        # Zählen der Ereignisse in festen Zeit-Bins
        min_t = self.timestamps[0]
        max_t = self.timestamps[-1] + self.binwidth
        bin_edges = np.arange(min_t, max_t + 1, self.binwidth)
        counts_per_bin, _ = np.histogram(self.timestamps, bins=bin_edges)

        self.raw_data = counts_per_bin
        self.t_raw_data = np.linspace(0,max_t-min_t,len(counts_per_bin))/1e12


        # Schwellenwertfilter anwenden (alle N vorherigen deltas > threshold)
        # mask = np.array([
        #     np.any(counts_per_bin[max(0, i - self.n_filter):i] > self.threshold)
        #     for i in range(len(counts_per_bin))
        # ])
        is_over_threshold = (counts_per_bin >= self.threshold).astype(np.uint8)
        window = np.convolve(is_over_threshold, np.ones(self.n_filter, dtype=np.uint8), mode='valid')
        mask = np.concatenate([np.zeros(self.n_filter, dtype=bool), window[:-1] > 0])

        new_filtered = counts_per_bin[mask]

        # Anfügen und Histogramm berechnen
        self.filtered_data = new_filtered
        if len(self.filtered_data) > 0:
            bins = np.arange(np.min(self.filtered_data), np.max(self.filtered_data)+1,1)-0.5
            self.histogram, self.bin_edges = np.histogram(self.filtered_data,bins=bins)
            bins = np.arange(np.min(counts_per_bin), np.max(counts_per_bin)+1,1)-0.5
            self.histogram_raw, self.bin_edges_raw = np.histogram(counts_per_bin,bins=bins)
        else:
            self.histogram = np.array([0])
            self.bin_edges = np.array([0, 1])
            self.histogram_raw = np.array([0])
            self.bin_edges_raw = np.array([0, 1])

        self.hist_updated.emit()
        self.trace_updated.emit()

    

    # def refilter(self, threshold=None, n_filter=None):
    #     """Wendet neuen Filter auf vorhandene Rohdaten an."""
    #     if threshold is not None:
    #         self.threshold = threshold
    #     if n_filter is not None:
    #         self.n_filter = n_filter

    #     logger.info(f"Refilter mit Threshold={self.threshold}, N={self.n_filter}")
    #     self._apply_filter_and_histogram()

    def get_result(self):
        """Rückgabe der verarbeiteten Daten."""
        return {
            'raw_data': self.raw_data,
            't_raw_data': self.t_raw_data,
            # 'filtered_data': self.filtered_data,
            'filtered_data': self.filtered_data,
            'histogram': self.histogram,
            'bin_edges': self.bin_edges
        }
    
    def refilter(self, threshold=None, n_filter=None):
        """Filterung neu anwenden auf alle gespeicherten Zeitdifferenzen."""
        if threshold is not None:
            self.threshold = threshold
        if n_filter is not None:
            self.n_filter = n_filter

        # Zählen der Ereignisse in festen Zeit-Bins
        min_t = self.timestamps[0]
        max_t = self.timestamps[-1] + self.binwidth
        bin_edges = np.arange(min_t, max_t + 1, self.binwidth)
        counts_per_bin, _ = np.histogram(self.timestamps, bins=bin_edges)

        self.raw_data = counts_per_bin
        self.t_raw_data = np.linspace(0,max_t-min_t,len(counts_per_bin))/1e12


        # Schwellenwertfilter anwenden (alle N vorherigen deltas > threshold)
        # mask = np.array([
        #     np.any(counts_per_bin[max(0, i - self.n_filter):i] > self.threshold)
        #     for i in range(len(counts_per_bin))
        # ])
        is_over_threshold = (counts_per_bin >= self.threshold).astype(np.uint8)
        window = np.convolve(is_over_threshold, np.ones(self.n_filter, dtype=np.uint8), mode='valid')
        mask = np.concatenate([np.zeros(self.n_filter, dtype=bool), window[:-1] > 0])
        new_filtered = counts_per_bin[mask]

        # Anfügen und Histogramm berechnen
        self.filtered_data = new_filtered
        if len(self.filtered_data) > 0:
            bins = np.arange(np.min(self.filtered_data), np.max(self.filtered_data)+1,1)-0.5
            self.histogram, self.bin_edges = np.histogram(self.filtered_data,bins=bins)
            bins = np.arange(np.min(counts_per_bin), np.max(counts_per_bin)+1,1)-0.5
            self.histogram_raw, self.bin_edges_raw = np.histogram(counts_per_bin,bins=bins)
        else:
            self.histogram = np.array([0])
            self.bin_edges = np.array([0, 1])
            self.histogram_raw = np.array([0])
            self.bin_edges_raw = np.array([0, 1])

        self.hist_updated.emit()
        self.trace_updated.emit()

    

    def save(self, tag):
        self.Filename = tag
        filepath  = self._save_logic.get_path_for_module(module_name='TimeTraceHistogram')
        if len(tag) > 0:
            filelabel  = tag + '_Timetrace_Histogram'
            filelabel_raw  = tag + '_Timetrace_Histogram_Raw_Data'
            filelabel_tag  = tag + '_Timetrace_Histogram_Raw_Tags'
        else:
            filelabel  = 'Timetrace_Histogram'
            filelabel_raw  = tag + 'Timetrace_Histogram_Raw_Data'
            filelabel_tag  = tag + 'Timetrace_Histogram_Raw_Tags'
        timestamp = datetime.datetime.now()

        """Speichert Histogrammdaten als CSV."""
        if self.histogram is not None:
            data = OrderedDict()
            data['bin_edges, Zaehlrate'] = self.bin_edges[:-1]+0.5
            data['histogram'] = self.histogram

            data2 = OrderedDict()
            data2["Raw Data"] = self.raw_data

            data3 = OrderedDict()
            data3["Raw Time Tags"] = self.timestamps

            parameters = OrderedDict()
            parameters['binwidth (ps)'] = self.binwidth
            parameters['n_filter'] = self.n_filter
            parameters['n_values'] = self.n_values
            parameters['threshold'] = self.threshold

            self._save_logic.save_data(
                data,
                filepath=filepath,
                parameters=parameters,
                filelabel=filelabel,
                fmt='%.6e',
                delimiter='\t',
                timestamp=timestamp
            )
            self._save_logic.save_data(
                data2,
                filepath=filepath,
                parameters=parameters,
                filelabel=filelabel_raw,
                fmt='%.6e',
                delimiter='\t',
                timestamp=timestamp
            )
            self._save_logic.save_data(
                data3,
                filepath=filepath,
                parameters=parameters,
                filelabel=filelabel_tag,
                fmt='%.6e',
                delimiter='\t',
                timestamp=timestamp
            )
            logger.info(f"Daten gespeichert unter {filepath}")
        else:
            logger.warning("Keine Histogrammdaten zum Speichern vorhanden.")

    # def cleanup(self):
    #     if self.counter is not None:
    #         self.counter.clear()
    #     logger.info("TimeTagger freigegeben.")

    #     # if self.plot_widget is not None:
    #     #     self.plot_widget.close()

    def cleanup(self):
        if self.stream is not None:
            self.stream.clear()
        if self.tagger is not None:
            self.tagger.free()
        if self.plot_widget is not None:
            self.plot_widget.close()
        logger.info("Aufgeräumt.")
        


    