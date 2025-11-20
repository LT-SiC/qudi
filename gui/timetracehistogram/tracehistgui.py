# -*- coding: utf-8 -*-
"""
This file contains the Qudi GUI module to operate the voltage (laser) scanner.

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

from ast import Pass
import numpy as np
import os
import pyqtgraph as pg

from collections import OrderedDict
from core.connector import Connector
from gui.colordefs import ColorScaleInferno
from gui.guibase import GUIBase
from gui.guiutils import ColorBar
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets
from qtpy import uic
import time 
from PyQt5 import QtTest
from PyQt5.QtWidgets import QFileDialog

from gui.timetracehistogram.tracehist_connectors_and_set_default import initialize_connections_and_defaultvalue
from logic.timetracehistogram_logic.timehist_logic import TimeHistLogic
# from hardware.Keysight_AWG_M8190.pym8190a import valid_ch_dict
# from logic.laserscanner.laser_scanner_logic import LaserScannerLogic


class TraceHistMainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        # Get the path to the *.ui file
        this_dir = os.path.dirname(__file__)
        ui_file = os.path.join(this_dir, 'ui_tracehist_gui.ui')

        # Load it
        super(TraceHistMainWindow, self).__init__()
        uic.loadUi(ui_file, self)
        self.show()


# UI_FILE = os.path.join(os.path.dirname(__file__), 'filtered_counter.ui')


class FilteredCounterGui(GUIBase, initialize_connections_and_defaultvalue):

    timehistlogic = Connector(interface='TimeHistLogic')

    def __init__(self,  **kwargs):
        super().__init__( **kwargs)

       

    def start_measurement(self):
        # self.logic.prepare()
        # self.logic.run()
        self.logic.start_live(new=True)

    def resubmit_measurement(self):
        # self.logic.prepare()
        # self.logic.run()
        print("resubmit clicked!")
        self.logic.start_live(new=False)

    def stop_measurement(self):
        # self.logic.prepare()
        # self.logic.run()
        self.logic.stop_live()


    def on_activate(self):
        self._mw = TraceHistMainWindow()
        self.logic: TimeHistLogic = self.timehistlogic()
        # Plotting
        self.hist_plot_widget = None
        self.trace_plot_widget = None
        self.plot_hist = None
        self.plot_hist_raw = None
        self.plot_trace = None


        # self._setupcontrollogic: SetupControlLogic = self.setupcontrollogic()
        self.initialize_connections_and_defaultvalues()
        self._init_plot()
        self.logic.hist_updated.connect(self._update_hist_plot, QtCore.Qt.QueuedConnection)
        self.logic.trace_updated.connect(self._update_trace_plot, QtCore.Qt.QueuedConnection)
        self.show()


    def _update_hist_plot(self):
        if self.logic.histogram is not None and self.logic.bin_edges is not None:
            self.plot_hist.setData(x=self.logic.bin_edges, y=self.logic.histogram)
            self._mw.hist_plot_widget.removeItem(self.plot_hist_raw)
            if self._mw.raw_data_checkBox.isChecked():
                self.plot_hist_raw.setData(x=self.logic.bin_edges_raw, y=self.logic.histogram_raw)
                self._mw.hist_plot_widget.addItem(self.plot_hist_raw)
            

    def _update_trace_plot(self):
        if self.logic.raw_data is not None:
            self.plot_trace.setData(x=self.logic.t_raw_data[-self.logic.plot_length:], y=self.logic.raw_data[-self.logic.plot_length:])

    
    def on_deactivate(self):
        # self.logic.on_deactivate()
        self._mw.close()

    def show(self):
        """Make window visible and put it above all other windows. """
        self._mw.show()
        self._mw.activateWindow()
        self._mw.raise_()

    def update_filter_params(self):
        self.logic.threshold = self._mw.thresholdSpinBox.value()
        self.logic.n_filter = self._mw.nFilterSpinBox.value()

    def update_collection_params(self):
        self.logic.n_values = self._mw.datapoints_spinBox.value()
        self.logic.binwidth = int(self._mw.binwidth_spinBox.value()*1e3)
    
    def update_plot_params(self):
        self.logic.plot_length = self._mw.window_size_spinBox.value()

    def refilter(self):
        self.logic.refilter()
        
    def save_histogram(self):
        # path, _ = QFileDialog.getSaveFileName(self, "Histogramm speichern", "", "CSV-Dateien (*.csv)")
        # if path:
        self.logic.save(tag=self._mw.Filename_LineEdit.text())

    def _init_plot(self):
        self.plot_hist = pg.PlotDataItem(
            self.logic.bin_edges, self.logic.histogram,stepMode=True, fillLevel=0, brush=(0,0,255,75))
        self.plot_hist_raw = pg.PlotDataItem(
            self.logic.bin_edges_raw, self.logic.histogram_raw,stepMode=True, fillLevel=0, brush=(255,0,0,100))
        self.plot_trace = pg.PlotDataItem(np.linspace(0,self.logic.n_values*self.logic.binwidth/1e12,len(self.logic.raw_data)), self.logic.raw_data)

        self.plot_hist.setZValue(1)
        self.plot_hist_raw.setZValue(0)
       
        self._mw.hist_plot_widget.addItem(self.plot_hist)#
        self._mw.hist_plot_widget.addItem(self.plot_hist_raw)#
        self._mw.hist_plot_widget.showGrid(x=True, y=True, alpha=0.8)
        self._mw.hist_plot_widget.setLabel(axis='left', text='Häufigkeit', units='')
        self._mw.hist_plot_widget.setLabel(axis='bottom', text='Zählrate', units='')
        self._mw.hist_plot_widget.setTitle("Live Histogramm gefilterter Zählraten")
        
        self._mw.trace_plot_widget.addItem(self.plot_trace)#
        self._mw.trace_plot_widget.showGrid(x=True, y=True, alpha=0.8)
        self._mw.trace_plot_widget.setLabel(axis='left', text='Counts', units='')
        self._mw.trace_plot_widget.setLabel(axis='bottom', text='Time', units='')
        self._mw.trace_plot_widget.setTitle("Live Time Tags")

