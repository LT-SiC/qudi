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

from gui.laserscanner.connectors_and_set_default import initialize_connections_and_defaultvalue as ple_default_functions

from hardware.Keysight_AWG_M8190.pym8190a import valid_ch_dict
from logic.laserscanner.laser_scanner_logic import LaserScannerLogic


class VoltScanMainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        # Get the path to the *.ui file
        this_dir = os.path.dirname(__file__)
        ui_file = os.path.join(this_dir, 'ui_laserscanner_gui.ui')

        # Load it
        super(VoltScanMainWindow, self).__init__()
        uic.loadUi(ui_file, self)
        self.show()


class LoadDialog(QtWidgets.QDialog):
    """ Dialog to load data """
    save_path = ""
    sigLoadPLE = QtCore.Signal(str)
    def __init__(self, parent, title="Select all PLE .dat files", text="Loading..."):
        super().__init__(parent)
        this_dir = os.path.dirname(__file__)
        ui_file = os.path.join(this_dir, 'ui_load.ui')
        uic.loadUi(ui_file, self)
    
        self.setWindowTitle(title)
        self.setWindowModality(QtCore.Qt.WindowModal)
        self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating)
        self.load_toolButton.clicked.connect(self.choose_folder_for_load)
        
        self.accepted.connect(self.load)    

        # self.rejected.connect()

    
    def choose_folder_for_load(self):
        self.file_path = QtWidgets.QFileDialog.getOpenFileNames()[0]
        # printdebug(self.debug, f'Chosen filepath is {dir_path}')
        self.load_path_lineEdit.setText(self.file_path)
        return self.file_path
   
   
    def load(self):
        self.sigLoadPLE.emit(self.file_path)



class VoltScanGui(GUIBase, ple_default_functions):
   
    # declare connectors
    voltagescannerlogic1 = Connector(interface='LaserScannerLogic')
    
    sigStartScan = QtCore.Signal()
    sigStopScan = QtCore.Signal()
    sigChangeVoltage = QtCore.Signal(float)
    sigChangeRange = QtCore.Signal(list)
    sigChangeResolution = QtCore.Signal(float)
    sigChangeSpeed = QtCore.Signal(float)
    sigChangeARCFactor = QtCore.Signal(float)
    sigChangeWavelength = QtCore.Signal(float)
    sigChangeLines = QtCore.Signal(int)
    sigChangeMaxLines = QtCore.Signal(int)
    sigSaveMeasurement = QtCore.Signal(str, list, list)

    sigCheckEmission = QtCore.Signal()
    sigResetARC = QtCore.Signal()

    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)

    def on_deactivate(self):
        """ Reverse steps of activation

        @return int: error code (0:OK, -1:error)
        """

        self._mw.close()
        return 0

    def on_activate(self):

        self._mw = VoltScanMainWindow()
        self._voltscan_logic: LaserScannerLogic = self.voltagescannerlogic1()
        self.initialize_connections_and_defaultvalues()

        #self._voltscan_logic.SigClock.connect(self.Update_Runtime, QtCore.Qt.QueuedConnection)
  
        # Get the image from the logic
        self.scan_matrix_image = pg.ImageItem(
            self._voltscan_logic.plot_matrix,
            axisOrder='row-major')

        self.scan_matrix_image.setRect(
            QtCore.QRectF(
                self._voltscan_logic.scan_range[0],
                0,
                self._voltscan_logic.scan_range[1] - self._voltscan_logic.scan_range[0],
                self._voltscan_logic.number_of_repeats)
        )

        self.scan_image = pg.PlotDataItem(
            self._voltscan_logic.plot_x,
            self._voltscan_logic.plot_y)

        self.scan_fit_image = pg.PlotDataItem(
            self._voltscan_logic.plot_x,
            self._voltscan_logic.plot_y,
            pen=pg.mkPen(pg.mkColor(255, 0, 0), style=QtCore.Qt.SolidLine))

        # Add the display item to the xy and xz VieWidget, which was defined in
        # the UI file.
        self._mw.ple_data_PlotWidget.addItem(self.scan_image)#
        self._mw.ple_data_PlotWidget.addItem(self.scan_fit_image)#
        #self._mw.voltscan_ViewWidget.addItem(self.scan_fit_image)
        self._mw.ple_data_PlotWidget.showGrid(x=True, y=True, alpha=0.8)
        self._mw.ple_data_PlotWidget.setLabel(axis='left', text='Fluorescence', units='Counts')
        self._mw.ple_data_PlotWidget.setLabel(axis='bottom', text='Detuning', units='GHz')

        self._mw.ple_matrix_PlotWidget.addItem(self.scan_matrix_image)
        self._mw.ple_matrix_PlotWidget.setLabel(axis='left', text='# Scan', units='')
        self._mw.ple_matrix_PlotWidget.setLabel(axis='bottom', text='Scan voltage', units='V')

        #self._mw.voltscan2_ViewWidget.addItem(self.scan_image2)
        #self._mw.voltscan2_ViewWidget.addItem(self.scan_fit_image)
        #self._mw.voltscan2_ViewWidget.showGrid(x=True, y=True, alpha=0.8)

        # Get the colorscales at set LUT
        my_colors = ColorScaleInferno()

        self.scan_matrix_image.setLookupTable(my_colors.lut)
        #self.scan_matrix_image2.setLookupTable(my_colors.lut)

        # Configuration of the Colorbar
        self.scan_cb = ColorBar(my_colors.cmap_normed, 100, 0, 100000)

        #adding colorbar to ViewWidget
        self._mw.voltscan_cb_ViewWidget.addItem(self.scan_cb) 
        self._mw.voltscan_cb_ViewWidget.hideAxis('bottom')
        self._mw.voltscan_cb_ViewWidget.hideAxis('left') 
        self._mw.voltscan_cb_ViewWidget.setLabel('right', 'Fluorescence', units='c/s') 

        # Connect the buttons and inputs for colorbar
        self._mw.voltscan_cb_manual_RadioButton.clicked.connect(self.refresh_matrix)
        self._mw.voltscan_cb_centiles_RadioButton.clicked.connect(self.refresh_matrix)

        # set initial values


        # Update the input/displayed numbers if the cursor has left the field:


        # self._mw.Repump_checkBox.stateChanged.connect(self.change_repump_state)
        # self._mw.A2_checkBox.stateChanged.connect(self.change_A2_state)
        # self._mw.A1_checkBox.stateChanged.connect(self.change_A1_state)

        # self._mw.MW1FrequencyDoubleSpinBox.valueChanged.connect(self.change_MW1_freq)
        # self._mw.MW2FrequencyDoubleSpinBox.valueChanged.connect(self.change_MW2_freq)
        # self._mw.MW3FrequencyDoubleSpinBox.valueChanged.connect(self.change_MW3_freq)
        # self._mw.MW1PowerDoubleSpinBox.valueChanged.connect(self.change_MW1_power)
        # self._mw.MW2PowerDoubleSpinBox.valueChanged.connect(self.change_MW2_power)
        # self._mw.MW3PowerDoubleSpinBox.valueChanged.connect(self.change_MW3_power)

        # self.updateButtonsEnabled()
        # self._mw.MW1_on_Button.clicked.connect(self.changeMW1State)
        # self._mw.MW2_on_Button.clicked.connect(self.changeMW2State)
        # self._mw.MW3_on_Button.clicked.connect(self.changeMW3State)

        # self._mw.MW1PowerDoubleSpinBox.setValue(-20)
        # self._mw.MW2PowerDoubleSpinBox.setValue(-20)
        # self._mw.MW3PowerDoubleSpinBox.setValue(-20)
        # self._mw.MW1FrequencyDoubleSpinBox.setValue(70)
        # self._mw.MW2FrequencyDoubleSpinBox.setValue(70)
        # self._mw.MW3FrequencyDoubleSpinBox.setValue(70)

        # #
        # self._mw.voltscan_cb_max_InputWidget.valueChanged.connect(self.refresh_matrix)
        # self._mw.voltscan_cb_min_InputWidget.valueChanged.connect(self.refresh_matrix)
        # self._mw.voltscan_cb_high_centile_InputWidget.valueChanged.connect(self.refresh_matrix)
        # self._mw.voltscan_cb_low_centile_InputWidget.valueChanged.connect(self.refresh_matrix)

        # # Connect signals
        self._voltscan_logic.sigUpdatePlots.connect(self.refresh_matrix)
        self._voltscan_logic.sigUpdatePlots.connect(self.refresh_plot)
        self._voltscan_logic.sigUpdatePlots.connect(self.refresh_lines)
        self._voltscan_logic.sigScanFinished.connect(self.scan_stopped)
        self._voltscan_logic.sigScanStarted.connect(self.scan_started)
        self._voltscan_logic.sigScanRangeAdjustment.connect(self.update_scan_range_ComboBox)
        self._voltscan_logic.sigScanRangeChanged.connect(self.update_scan_range_DoubleSpinBox)
        self._voltscan_logic.sigPLEfactorChanged.connect(self.update_factor_V_to_GHz_LineEdit)
        self._voltscan_logic.sigLockLaserChanged.connect(self.update_ple_Lock_Laser_CheckBox)

        self._voltscan_logic.ple_progressBar.connect(self.update_ProgressBar)
        self._voltscan_logic.emissionChanged.connect(self.update_LaserEmissionBar)
        self._voltscan_logic.ARCFactorChanged.connect(self.update_ARCFactorSpinBox)
        self._voltscan_logic.sigUpdateWavlength.connect(self.refresh_wavelength)


        #self.sigStartScan.connect(self._voltscan_logic.start_scanning)
        #self.sigStopScan.connect(self._voltscan_logic.stop_scanning)
        self.sigChangeVoltage.connect(self._voltscan_logic.set_voltage)
        self.sigChangeRange.connect(self._voltscan_logic.set_scan_range)
        self.sigChangeSpeed.connect(self._voltscan_logic.set_scan_speed)
        self.sigChangeARCFactor.connect(self._voltscan_logic.set_arc_factor)
        self.sigChangeWavelength.connect(self._voltscan_logic.set_ctl_wavelength)
        self.sigChangeLines.connect(self._voltscan_logic.set_show_lines)
        self.sigChangeMaxLines.connect(self._voltscan_logic.set_scan_lines)
        self.sigChangeResolution.connect(self._voltscan_logic.set_resolution)
        self.sigSaveMeasurement.connect(self._voltscan_logic.save_data)

        self.sigCheckEmission.connect(self._voltscan_logic.check_emission)
        self.sigResetARC.connect(self._voltscan_logic.set_arc_factor)

        ### for loading
        self._load_dialog = LoadDialog(self._mw)
        self._load_dialog.sigLoadPLE.connect(self.load_ple_data)

        # Set up ScanRange combobox
        scan_ranges = ['Single Peak', 'Double Peak', 'Selected Range']
        for n, ch in enumerate(scan_ranges):
            self._mw.ple_ScanRanges_ComboBox.addItem(str(ch), n)

        self._mw.ple_ScanRanges_ComboBox.activated.connect(self.select_scan_range)
        self._mw.ple_Stop_Button.setEnabled(False)

        self._mw.resolutionSpinBox.setValue(self._voltscan_logic.stepsize)

        # for CTL option:
        laser_choice = ['DLC Pro CTL', 'DLC Pro 917']
        channel_choice = ['Fine In 1', 'Fine In 2',"Fast In 3","Fast In 4"]
        tuning_choice = ["PiezoControl","MotorControl"]
        for n, ch in enumerate(laser_choice):
            self._mw.laser_comboBox.addItem(str(ch), n)
        self._mw.laser_comboBox.activated.connect(self.select_laser)
        for n, ch in zip([0,1,2,4],channel_choice):
            self._mw.input_channel_comboBox.addItem(str(ch), n)
        self._mw.input_channel_comboBox.activated.connect(self.select_channel)
        for n, ch in enumerate(tuning_choice):
            self._mw.tuning_mode_comboBox.addItem(str(ch), n)
        self._mw.tuning_mode_comboBox.activated.connect(self.select_mode)


        self._mw.show()

    def show(self):
        """Make window visible and put it above all other windows. """
        self._mw.show()
        self._mw.activateWindow()
        self._mw.raise_()

    def updateButtonsEnabled(self):
        # """ Logic told us to update our button states, so set the buttons accordingly. """
        self._mw.MW1_on_Button.setCheckable(True)
        self._mw.MW2_on_Button.setCheckable(True)
        self._mw.MW3_on_Button.setCheckable(True)
    
    # def run_stop(self, is_checked):
    #     """ Manages what happens if scan is started/stopped """
    #     print('Start scanning')
    #     if is_checked:
    #         self.sigStartScan.emit()
    #         #self._mw.voltscan_ViewWidget.removeItem(self.scan_fit_image) CHANGE
    #         #self._mw.voltscan2_ViewWidget.removeItem(self.scan_fit_image)
    #     else:
    #         self.sigStopScan.emit()

    def scan_started(self):
        self.disable_scan_actions()
        self._voltscan_logic.curr_freq = 0.0
        self._voltscan_logic.current_freq = ''
        self.refresh_current_freq()

    def scan_stopped(self):
        self.enable_scan_actions()
        if self._voltscan_logic.lock_laser:
            perform_fit = self._voltscan_logic.PerformFit
            self._voltscan_logic.PerformFit = True
        self.refresh_plot()
        self.refresh_matrix()
        self.refresh_lines()
        if self._voltscan_logic.lock_laser:
            self._voltscan_logic.PerformFit = perform_fit
        self.refresh_current_freq()


    # for laoding
    def load_ple_data(self, files):
        success = True
        try:
            f1 = [f for f in files if f.endswith("data.dat")][0]
        except IndexError:
            print("Missing '_data.dat' file!")
            success = False
        try:
            f2 = [f for f in files if f.endswith("raw_freqs.dat")][0]
        except IndexError:
            print("Missing '_raw_freqs.dat' file!")
            success = False
        try:
            f3 = [f for f in files if f.endswith("raw_trace.dat")][0]
        except IndexError:
            print("Missing '_raw_trace.dat' file!")
            success = False
        
        if success:
            print("Files are here! Now try laoding...")
            self._voltscan_logic.sigUpdatePlots.emit()
            pass
        else:
            print("Loading failed!")
        # try:
        #     xy_data = np.loadtxt(file, comments='#')
        #     self._scanning_logic.xy_image[:,:,3] = xy_data
        #     self._scanning_logic.signal_xy_image_updated.emit()
        # except:
        #     print("Error in load xy scan! You might have to adjust the current confocal size/resolution onto the the values of required file (It just updates countdata not scan area)!")


        
    def refresh_current_freq(self):
        self._mw.current_freq_Label.setText(str(self._voltscan_logic.current_freq))
    # def change_current_arc_factor(self):
    #     self._mw.current_arc_factor_label.setText(str(self._voltscan_logic.arc_factor_label))
    def update_ARCFactorSpinBox(self,val):
        self._mw.ARCFactorSpinBox.setValue(val)
    # def update_WavelengthdoubleSpinBox(self,val):
    #     self._mw.WavelengthdoubleSpinBox.setValue(val)

    def refresh_plot(self):
        """ Refresh the xy-plot image """
        self.scan_image.setData(self._voltscan_logic.plot_x_frequency, self._voltscan_logic.plot_y)
        self._mw.ple_data_PlotWidget.removeItem(self.scan_fit_image)
        if self._voltscan_logic.PerformFit:
            self._mw.ple_data_PlotWidget.addItem(self.scan_fit_image)
            print("PLE GUI PERFOMING GAUSSIAN FIT")
            try:
                interpolated_x_data, fit_data, result = self._voltscan_logic.do_gaussian_fit()
                self._mw.ple_Contrast_Fit_Label.setText(self._voltscan_logic.Contrast_Fit)
                self._mw.ple_Frequencies_Fit_Label.setText(self._voltscan_logic.Frequencies_Fit_GHz)
                self._mw.ple_Linewidths_Fit_Label.setText(self._voltscan_logic.Linewidths_Fit_GHz)

                self.scan_fit_image.setData(interpolated_x_data*self._voltscan_logic.factor_V_to_GHz , fit_data) # 0.22 if FeedForward in turned on
            except:
                self.log.warning('Gaussian fit not successful...')

    def refresh_matrix(self):
        """ Refresh the xy-matrix image """
        self.scan_matrix_image.setImage(self._voltscan_logic.plot_matrix, axisOrder='row-major')
        self.scan_matrix_image.setRect(
            QtCore.QRectF(
                self._voltscan_logic.scan_range[0],
                0,
                self._voltscan_logic.scan_range[1] - self._voltscan_logic.scan_range[0],
                self._voltscan_logic.number_of_repeats)
            )
        self.refresh_scan_colorbar()

        scan_image_data = self._voltscan_logic.plot_matrix

        # If "Centiles" is checked, adjust colour scaling automatically to centiles.
        # Otherwise, take user-defined values.
        if self._mw.voltscan_cb_centiles_RadioButton.isChecked():
            low_centile = self._mw.voltscan_cb_low_centile_InputWidget.value()
            high_centile = self._mw.voltscan_cb_high_centile_InputWidget.value()

            cb_min = np.percentile(scan_image_data, low_centile)
            cb_max = np.percentile(scan_image_data, high_centile)
        else:
            cb_min = self._mw.voltscan_cb_min_InputWidget.value()
            cb_max = self._mw.voltscan_cb_max_InputWidget.value()

        # Now update image with new color scale, and update colorbar
        self.scan_matrix_image.setImage(
            image=scan_image_data,
            levels=(cb_min, cb_max),
            axisOrder='row-major')
 
        self.refresh_scan_colorbar()

    def refresh_scan_colorbar(self):
        """ Update the colorbar to a new scaling."""

        # If "Centiles" is checked, adjust colour scaling automatically to centiles.
        # Otherwise, take user-defined values.
        if self._mw.voltscan_cb_centiles_RadioButton.isChecked():
            low_centile = self._mw.voltscan_cb_low_centile_InputWidget.value()
            high_centile = self._mw.voltscan_cb_high_centile_InputWidget.value()

            cb_min = np.percentile(self.scan_matrix_image.image, low_centile)
            cb_max = np.percentile(self.scan_matrix_image.image, high_centile)
        else:
            cb_min = self._mw.voltscan_cb_min_InputWidget.value()
            cb_max = self._mw.voltscan_cb_max_InputWidget.value()

        self.scan_cb.refresh_colorbar(cb_min, cb_max)
        # self._mw.voltscan_cb_ViewWidget.update() CHANGE

    def refresh_lines(self):
        self._mw.elapsed_lines_DisplayWidget.display(self._voltscan_logic._scan_counter_up)

    def refresh_wavelength(self):
        self._mw.wavelength_DisplayWidget.display(self._voltscan_logic.ctl_wavelength)

    def change_voltage(self,tmp=0):
        self.sigChangeVoltage.emit(self._mw.constDoubleSpinBox.value())

    def change_start_volt(self,tmp=0):
        self.sigChangeRange.emit([
            self._mw.startDoubleSpinBox.value(),
            self._mw.stopDoubleSpinBox.value()
        ])

    def change_speed(self,tmp=0):
        self.sigChangeSpeed.emit(self._mw.speedDoubleSpinBox.value())
    def change_arc_factor(self,tmp=0):
        self.sigChangeARCFactor.emit(self._mw.ARCFactorSpinBox.value())
    def change_wavelength(self,tmp=0):
        self.sigChangeWavelength.emit(self._mw.WavelengthdoubleSpinBox.value())

    def change_stop_volt(self,tmp=0):
        self.sigChangeRange.emit([
            self._mw.startDoubleSpinBox.value(),
            self._mw.stopDoubleSpinBox.value()
        ])

    def change_lines(self,tmp=0):
        self.sigChangeLines.emit(self._mw.linesSpinBox.value())
    
    def change_maxlines(self,tmp=0):
        self.sigChangeMaxLines.emit(self._mw.maxlinesSpinBox.value())

    def change_resolution(self,tmp=0):
        self.sigChangeResolution.emit(self._mw.resolutionSpinBox.value())

    def get_matrix_cb_range(self):
        """
        Determines the cb_min and cb_max values for the matrix plot
        """
        matrix_image = self.scan_matrix_image.image

        # If "Manual" is checked or the image is empty (all zeros), then take manual cb range.
        # Otherwise, calculate cb range from percentiles.
        if self._mw.voltscan_cb_manual_RadioButton.isChecked() or np.max(matrix_image) < 0.1:
            cb_min = self._mw.voltscan_cb_min_InputWidget.value()
            cb_max = self._mw.voltscan_cb_max_InputWidget.value()
        else:
            # Exclude any zeros (which are typically due to unfinished scan)
            matrix_image_nonzero = matrix_image[np.nonzero(matrix_image)]

            # Read centile range
            low_centile = self._mw.voltscan_cb_low_centile_InputWidget.value()
            high_centile = self._mw.voltscan_cb_high_centile_InputWidget.value()

            cb_min = np.percentile(matrix_image_nonzero, low_centile)
            cb_max = np.percentile(matrix_image_nonzero, high_centile)

        cb_range = [cb_min, cb_max]
        return cb_range

    def select_scan_range(self, index):
        """ The automated scan range adjustment was changed.

            @param index int: index of selected channel item in combo box
        """
        self._voltscan_logic.scan_range_adjustment = int(self._mw.ple_ScanRanges_ComboBox.itemData(index, QtCore.Qt.UserRole))
        
    def update_scan_range_ComboBox(self, value):
        self._mw.ple_ScanRanges_ComboBox.setCurrentIndex(value)
    
    def update_scan_range_DoubleSpinBox(self, value_min, value_max):
        self._mw.startDoubleSpinBox.setValue(value_min)
        self._mw.stopDoubleSpinBox.setValue(value_max)

    def update_factor_V_to_GHz_LineEdit(self, value):
        self._mw.factor_V_to_GHz_LineEdit.setText(str(value))
    
    def update_ple_Lock_Laser_CheckBox(self, val):
        self._mw.ple_Lock_Laser_CheckBox.setChecked(val)

    def update_ProgressBar(self, val, scan_dur):
        # print("Update ProgressBar (GUI): ", val, scan_dur)
        self._mw.ple_progressBar.setRange(0, scan_dur)
        self._mw.ple_progressBar.setValue(val)

    def update_LaserEmissionBar(self, val_ctl, val_dlc):
        if self._voltscan_logic.laser_choice == 0:
            val = val_ctl
        elif self._voltscan_logic.laser_choice == 1:
            val = val_dlc
        self._mw.laserstatus_progressBar.setRange(0, 1)
        self._mw.laserstatus_progressBar.setValue(val)
    
    def select_laser(self, index):
        self._voltscan_logic.laser_choice = int(self._mw.laser_comboBox.itemData(index, QtCore.Qt.UserRole))
        i = self._voltscan_logic.laser_choice
        print(f"PLE laser changed to {i} : {dict({0:'CTL',1:'DLC Pro'})[i]}")
        self.sigCheckEmission.emit()
        self.sigResetARC.emit()
        if i == 0:
            self._mw.tuning_mode_comboBox.setEnabled(True)
            self._mw.WavelengthdoubleSpinBox.setEnabled(True)
        elif i == 1:
            self._mw.tuning_mode_comboBox.setEnabled(False)
            self._mw.tuning_mode_comboBox.setCurrentIndex(0)
            self._mw.WavelengthdoubleSpinBox.setEnabled(False)
            self._mw.WavelengthdoubleSpinBox.setEnabled(False)
            self.select_mode(0)

    
    def select_channel(self, index):
        self._voltscan_logic.input_channel = int(self._mw.input_channel_comboBox.itemData(index, QtCore.Qt.UserRole))
        i = self._voltscan_logic.input_channel
        print(f"PLE voltage channel changed to {i} : {dict({0:'Fine In 1',1:'Fine In 2',2:'Fast In 3',4:'Fast In 4'})[i]}")
        self.sigCheckEmission.emit()
        self.sigResetARC.emit()

    def select_mode(self, index):
        self._voltscan_logic.tuning_mode = int(self._mw.tuning_mode_comboBox.itemData(index, QtCore.Qt.UserRole))
        i = self._voltscan_logic.tuning_mode
        print(f"PLE tuning methode changed to {i} : {dict({0:'PiezoControl',1:'MotorControl'})[i]}")
        self.sigCheckEmission.emit()
        self.sigResetARC.emit()
        if i == 0:
            self._mw.current_arc_factor_label.setText(str("ARC Factor (V/V):"))
        elif i == 1:
            self._mw.current_arc_factor_label.setText(str("ARC Factor (nm/V):"))

    
    def disable_scan_actions(self):
        self._mw.ple_Run_Button.setEnabled(False)
        self._mw.ple_Stop_Button.setEnabled(True)
        self._mw.ple_Continue_Button.setEnabled(False)
        self._mw.ple_Load_Button.setEnabled(False)
        self._mw.toggle_emission_Button.setEnabled(False)
        self._mw.factor_V_to_GHz_pushButton.setEnabled(False)
    
    def enable_scan_actions(self):
        self._mw.ple_Run_Button.setEnabled(True)
        self._mw.ple_Stop_Button.setEnabled(False)
        self._mw.ple_Continue_Button.setEnabled(True)
        self._mw.ple_Load_Button.setEnabled(True)
        self._mw.toggle_emission_Button.setEnabled(True)
        self._mw.factor_V_to_GHz_pushButton.setEnabled(True)
