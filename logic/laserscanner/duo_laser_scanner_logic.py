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
from gui.laser import laser
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

from logic.laserscanner.duo_ple_default_values_and_widget_functions import ple_default_values_and_widget_functions as ple_default
from gui.laserscanner.duo_connectors_and_set_default import initialize_connections_and_defaultvalue

from logic.confocal_logic import ConfocalLogic
from logic.save_logic import SaveLogic
from logic.fit_logic import FitLogic
from logic.TopticaDLCLogic import CTLLogic, DLCProLogic
from logic.wavemeter_logger_logic import WavemeterLoggerLogic
from hardware.national_instruments_x_series import NationalInstrumentsXSeries



class LaserScannerLogic(GenericLogic, ple_default):

    """This logic module controls scans of DC voltage on the fourth analog
    output channel of the NI Card.  It collects countrate as a function of voltage.
    """
    sig_data_updated = QtCore.Signal()

    #---------declare connectors-------#
    confocalscanner1 = Connector(interface='ConfocalScannerInterface')
    savelogic = Connector(interface='SaveLogic')
    mcas_holder = Connector(interface='McasDictHolderInterface')
    fitlogic = Connector(interface='FitLogic')
    ctllogic=Connector('CTLLogic')
    dlcprologic=Connector('DLCProLogic')
    wavemeterlogic= Connector(interface="WavemeterLoggerLogic")
    #-------------------------------#
    
    # Laser 1:
    Filename = StatusVar('Filename', "Filename")
    _scan_range_adjustment = StatusVar('scan_range_adjustment', 1)
    scan_range = StatusVar('scan_range', [-3, 3])
    resolution = StatusVar(default=100)
    stepsize = StatusVar(default=0.01)
    number_of_lines = StatusVar(default=10)
    number_of_repeats = StatusVar(default=10)
    NumberOfPeaks = StatusVar('NumberOfPeaks', default=2)
    _scan_speed = StatusVar('scan_speed', 0.1)
    _static_v = StatusVar('goto_voltage', 0)
    factor_V_to_GHz = StatusVar('factor_V_to_GHz', 1)
    current_freq = StatusVar('current_freq', "")
    fc = StatusVar('fits', None)
    enable_A1 = StatusVar('enable_A1', False)
    enable_A2 = StatusVar('enable_A2', True)
    enable_Repump = StatusVar('enable_Repump', False)
    enable_PulsedRepump = StatusVar('enable_PulsedRepump', True)
    enable_CTL = StatusVar('enable_CTL', False)
    enable_PulsedCTL = StatusVar('enable_PulsedCTL', False)
    enable_Blue = StatusVar('enable_Blue', False)
    enable_PulsedBlue = StatusVar('enable_PulsedBlue', False)
    RepumpWhenIonized = StatusVar('RepumpWhenIonized', False)
    MW1_Freq = StatusVar('MW1_freq', 70)
    MW2_Freq = StatusVar('MW2_freq', 140)
    MW3_Freq = StatusVar('MW3_freq', 210)
    _MW1_Power = StatusVar('MW1_Power', -21)
    MW2_Power = StatusVar('MW2_Power', -21)
    MW3_Power = StatusVar('MW3_Power', -21)
    enable_MW1 = StatusVar('enable_MW1', True)
    enable_MW2 = StatusVar('enable_MW2', True)
    enable_MW3 = StatusVar('enable_MW3', False)

    arc_factor = StatusVar('arc_factor', 10.)
    arc_factor_ctl_pc = StatusVar('arc_factor_ctl_pc', 10.)
    arc_factor_ctl_mc = StatusVar('arc_factor_ctl_mc', 0.01)
    arc_factor_dlcpro = StatusVar('arc_factor_dlcpro', 15.)
    ctl_wavelength = StatusVar('ctl_wavelength', 916.489390)
    laser_choice = StatusVar('laser_choice', 1)
    input_channel = StatusVar('input_channel', 0)
    tuning_mode = StatusVar('tuning_mode', 0)
    ple_progress = StatusVar('ple_progress', 0)
    laserstatus = StatusVar('laserstatus', 0)
    automatic_calibration = StatusVar('automatic_calibration', True)
    discard = StatusVar('discard', False)
    discard_direction = StatusVar('discard_direction', 1)
    discard_threshold = StatusVar('discard_threshold', 100.)

    
    # Laser 2:
    Filename_2 = StatusVar('Filename_2', "Filename_2")
    _scan_range_adjustment_2 = StatusVar('scan_range_adjustment_2', 1)
    scan_range_2 = StatusVar('scan_range_2', [-3, 3])
    resolution_2 = StatusVar(default=100)
    stepsize_2 = StatusVar(default=0.01)
    number_of_lines_2 = StatusVar(default=10)
    number_of_repeats_2 = StatusVar(default=10)
    NumberOfPeaks_2 = StatusVar('NumberOfPeaks_2', default=2)
    _scan_speed_2 = StatusVar('scan_speed_2', 0.1)
    _static_v_2 = StatusVar('goto_voltage_2', 0)
    factor_V_to_GHz_2 = StatusVar('factor_V_to_GHz_2', 1)
    current_freq_2 = StatusVar('current_freq_2', "")
    fc_2 = StatusVar('fits_2', None)
    enable_A1_2 = StatusVar('enable_A1_2', False)
    enable_A2_2 = StatusVar('enable_A2_2', True)
    enable_Repump_2 = StatusVar('enable_Repump_2', False)
    enable_PulsedRepump_2 = StatusVar('enable_PulsedRepump_2', True)
    enable_CTL_2 = StatusVar('enable_CTL_2', False)
    enable_PulsedCTL_2 = StatusVar('enable_PulsedCTL_2', False)
    enable_Blue_2 = StatusVar('enable_Blue_2', False)
    enable_PulsedBlue_2 = StatusVar('enable_PulsedBlue_2', False)
    RepumpWhenIonized_2 = StatusVar('RepumpWhenIonized_2', False)
    MW1_Freq_2 = StatusVar('MW1_freq_2', default=70)
    MW2_Freq_2 = StatusVar('MW2_freq_2', default=140)
    MW3_Freq_2 = StatusVar('MW3_freq_2', default=210)
    MW1_Power_2 = StatusVar('MW1_Power_2', default=-21)
    MW2_Power_2 = StatusVar('MW2_Power_2', default=-21)
    MW3_Power_2 = StatusVar('MW3_Power_2', default=-21)
    enable_MW1_2 = StatusVar('enable_MW1_2', default=True)
    enable_MW2_2 = StatusVar('enable_MW2_2', default=True)
    enable_MW3_2 = StatusVar('enable_MW3_2', default=False)

    arc_factor_2 = StatusVar('arc_factor_2', 10.)
    arc_factor_ctl_pc_2 = StatusVar('arc_factor_ctl_pc_2', 10.)
    arc_factor_ctl_mc_2 = StatusVar('arc_factor_ctl_mc_2', 0.01)
    arc_factor_dlcpro_2 = StatusVar('arc_factor_dlcpro_2', 15.)
    ctl_wavelength_2 = StatusVar('ctl_wavelength_2', 916.489390)
    laser_choice_2 = StatusVar('laser_choice_2', 2)
    input_channel_2 = StatusVar('input_channel_2', 0)
    tuning_mode_2 = StatusVar('tuning_mode_2', 0)
    ple_progress_2 = StatusVar('ple_progress_2', 0)
    laserstatus_2 = StatusVar('laserstatus_2', 0)
    automatic_calibration_2 = StatusVar('automatic_calibration_2', True)
    discard_2 = StatusVar('discard_2', False)
    discard_direction_2 = StatusVar('discard_direction_2', 1)
    discard_threshold_2 = StatusVar('discard_threshold_2', 100.)


    ple_locked = StatusVar('ple_locked', False)
    ple_locked_2 = StatusVar('ple_locked_2', False)

    _ch = ConfigOption('default_channel', default=1)


    sigChangeVoltage = QtCore.Signal(float, int)
    sigVoltageChanged = QtCore.Signal(float)
    sigScanNextLine = QtCore.Signal()
    sigScanNextLine_2 = QtCore.Signal()
    sigUpdatePlots = QtCore.Signal(int)
    sigScanFinished = QtCore.Signal(int)
    sigScanStarted = QtCore.Signal(int)
    sigFitPerformed =  QtCore.Signal(str)
    sigFitPerformed_2 =  QtCore.Signal(str)
    SigIonized = QtCore.Signal()            # not used currently
    sigScanRangeAdjustment = QtCore.Signal(int,int)
    sigScanRangeChanged = QtCore.Signal(float, float,int)
    sigLockLaserChanged = QtCore.Signal(bool,int)
    sigPLEfactorChanged = QtCore.Signal(float,int)

    ARCFactorChanged = QtCore.Signal(float, int)
    ple_progressBar = QtCore.Signal(float, float, int)
    updateProgressBarSignal = QtCore.Signal(int)
    emissionChanged = QtCore.Signal(float, float)
    sigUpdateWavlength = QtCore.Signal(int)
    
    # to cut the scan line into smaller parts:
    mode=0
    xy_coords = [0,0]
    
    def __init__(self, **kwargs):
        """ Create VoltageScanningLogic object with connectors.

          @param dict kwargs: optional parameters
        """
        super().__init__(**kwargs)

        # locking for thread safety
        self.threadlock = Mutex()
        self.stopRequested = False
        self.stopRequested_2 = False
        self.AbortRequested=False
        self.local_counts=[]
        self.local_counts_2=[]
        self.MW1_Power=-21
        self.volt_list=[] #track PLE
        self.timstap_list=[] #timestamp for PLE track
        self.accepted_list=[]
        self.hashed = False
        self.ionized = False
        self.laser_at_position = False
        self.measurement_running = False
 
        

        self.fit_x = []
        self.fit_x_2 = []
        self.fit_y = []
        self.fit_y_2 = []
        self.plot_x = []
        self.plot_x_2 = []
        self.plot_x_frequency=[]
        self.plot_x_frequency_2=[]
        self.plot_y = []
        self.plot_y_2 = []
        self.bias_voltages = []
        self.bias_voltages_2 = []

        self.adv_mode = 'SING'
        self.peak_min_counts=300
        
        

    def on_activate(self):
        """ Initialisation performed during activation of the module.
        """
        self._scanning_device: NationalInstrumentsXSeries = self.confocalscanner1()
        self._save_logic: SaveLogic = self.savelogic()
        self._awg = self.mcas_holder()
        self._wavemeterlogic: WavemeterLoggerLogic = self.wavemeterlogic()
        self._fit_logic: FitLogic = self.fitlogic()
        self._ctl_logic: CTLLogic = self.ctllogic()
        self._dlcpro_logic: DLCProLogic = self.dlcprologic()
        self.a_range = [self._scanning_device.get_position_range()[3],self._scanning_device.get_position_range()[2]]
        laser_index = 1

        # Initialise the current position of all four scanner channels.
        self.current_position = self._scanning_device.get_scanner_position()

        # initialise the range for scanning
        self.set_scan_range(self.scan_range, 1)
        self.set_scan_range(self.scan_range_2, 2)

        # Keep track of the current static voltage even while a scan may cause the real-time
        # voltage to change.
        self.goto_voltage(self._static_v, 1)
        self.goto_voltage(self._static_v_2, 2)

        # Sets connections between signals and functions
        self.sigChangeVoltage.connect(self._change_voltage, QtCore.Qt.QueuedConnection)
        self.sigScanNextLine.connect(self._do_next_line, QtCore.Qt.QueuedConnection)
        self.sigScanNextLine_2.connect(self._do_next_line_2, QtCore.Qt.QueuedConnection)
        self.sigFitPerformed.connect(self.goto_fitted_peak, QtCore.Qt.QueuedConnection)
        self.sigFitPerformed_2.connect(self.goto_fitted_peak_2, QtCore.Qt.QueuedConnection)

        self.updateProgressBarSignal.connect(self.update_ProgressBar, QtCore.Qt.QueuedConnection)

        # Initialization of internal counter for scanning
        self._scan_counter_up = 0
        self._scan_counter_up_2 = 0
        # Keep track of scan direction
        self.upwards_scan = True

        # calculated number of points in a scan, depends on speed and max step size
        self._num_of_steps = 50  # initialising.  This is calculated for a given ramp.

        #############################

        # TODO: allow configuration with respect to measurement duration
        self.acquire_time = 20  # seconds

        # default values for clock frequency and slowness
        # slowness: steps during retrace line
        self.set_resolution(self.stepsize,1)
        self.set_resolution(self.stepsize_2,2)
        self._goto_speed = 10  # 0.01  # volt / second
        self._goto_speed_2 = 10  # 0.01  # volt / second
        self.set_scan_speed(self._scan_speed,1)
        self.set_scan_speed(self._scan_speed_2,2)
        self._smoothing_steps = 0  # steps to accelerate between 0 and scan_speed
        self._max_step = 0.01  # volt

        ##############################

        # Initialie data matrix
        self._initialise_data_matrix(100,1)
        self._initialise_data_matrix(100,2)
        self._wavemeterlogic.start_scanning() #the array of frequencies will be updated
        self.start_stop_timestamps=[]
        self.timestamp_list=[]
        self.f_start_end=[]
        self.xy_data=[] # will contain the frequency and the counts associated with this frequency for each trace measured# ok is not used anymore
        self.range_defined=False
        
        self.start_stop_timestamps_2=[]
        self.timestamp_list_2=[]
        self.f_start_end_2=[]
        self.xy_data_2=[] # will contain the frequency and the counts associated with this frequency for each trace measured# ok is not used anymore
        self.range_defined_2=False

        self.tmp_storage=[]
        self.measured_frequencies=[]
        self.start_list=[]
        self.stop_list=[]
        
        self.tmp_storage_2=[]
        self.measured_frequencies_2=[]
        self.start_list_2=[]
        self.stop_list_2=[]

        self.interpolated_x_data=np.linspace(0,1,100)
        self.interpolated_x_data_2=np.linspace(0,1,100)
        self.measurement_running = False
        self.measurement_running_2 = False
        self.factor_V_to_GHz = self._scanning_device._scanner_position_ranges[3][1]
        self.factor_V_to_GHz_2 = self._scanning_device._scanner_position_ranges[2][1]
        self.laser_choice = 1
        self.laser_choice_2 = 2
        self.tuning_mode = 0
        self.tuning_mode_2 = 0
        self.check_emission()
        print("DuoPLE activation done!")

    def on_deactivate(self):
        """ Deinitialisation performed during deactivation of the module.
        """
        self.stopRequested = True
        self.stopRequested_2 = True
        self.AbortRequested = False
        self.measurement_running = False
        self.measurement_running_2 = False

    
    @property
    def scan_range_adjustment(self):
        return self._scan_range_adjustment
    
    @scan_range_adjustment.setter
    def scan_range_adjustment(self, val):
        try:
            if val <=2:
                self._scan_range_adjustment = int(val)
            else:
                self.log.error('Unkown input for scan range.')
        except: 
            if val == 'Single Peak':
                self._scan_range_adjustment = 0
            elif val == 'Double Peak':
                self._scan_range_adjustment = 1
            elif val == 'Selected Range':
                self._scan_range_adjustment = 2
            else:
                self.log.error('Unkown input for scan range.')
        self.sigScanRangeAdjustment.emit(self._scan_range_adjustment,1)

    @property
    def scan_range_adjustment_2(self):
        return self._scan_range_adjustment_2
    
    @scan_range_adjustment_2.setter
    def scan_range_adjustment_2(self, val):
        try:
            if val <=2:
                self._scan_range_adjustment_2 = int(val)
            else:
                self.log.error('Unkown input for scan range.')
        except: 
            if val == 'Single Peak':
                self._scan_range_adjustment_2 = 0
            elif val == 'Double Peak':
                self._scan_range_adjustment_2 = 1
            elif val == 'Selected Range':
                self._scan_range_adjustment_2 = 2
            else:
                self.log.error('Unkown input for scan range.')
        self.sigScanRangeAdjustment.emit(self._scan_range_adjustment_2,2)
    
    @property
    def lock_laser(self):
        return self._lock_laser
    
    @lock_laser.setter
    def lock_laser(self, val):
        self._lock_laser = val
        print("sigLockLaserChanged emitted from logic")
        #self.sigLockLaserChanged.emit(self.lock_laser)

    @QtCore.Slot(float)
    def goto_voltage(self, volts=None, laser_index=1):
        """Forwarding the desired output voltage to the scanning device.

        @param float volts: desired voltage (volts)

        @return int: error code (0:OK, -1:error)
        """
        # Changes the respective value
        if volts is not None:
            if laser_index == 1:
                self._static_v = volts
            elif laser_index == 2:
                self._static_v_2 = volts

        # Checks if the scanner is still running
        if (self.module_state() == 'locked'
                or self._scanning_device.module_state() == 'locked'):
            self.log.error('Cannot goto, because scanner is locked!')
            return -1
        else:
            self.sigChangeVoltage.emit(volts, laser_index)
            return 0


    def _change_voltage(self, new_voltage, laser_index=1):
        """ Threaded method to change the hardware voltage for a goto.
        @return int: error code (0:OK, -1:error)
        """
        print(f"DuoPLE{laser_index}: ",new_voltage)
        ramp_scan = self._generate_ramp(self.get_current_voltage(laser_index), new_voltage, 0.75, laser_index)
        print(f"DuoPLE{laser_index}: changed _change_voltage in laserscannerlogic" )
        print(f"DuoPLE{laser_index}: ",ramp_scan)
        if len(ramp_scan[0])==1:
            ramp_scan=np.hstack((ramp_scan,ramp_scan))
        elif len(ramp_scan[0])<1:
            print(f"DuoPLE{laser_index}: Ramp not corrently initialized. Maybe laser is not at desired position.")
            return 0

        self._initialise_scanner()
        ignored_counts = self._scan_line(ramp_scan) # was removed from original code once. But why? We try to just put back.
        self._close_scanner()
        self.sigVoltageChanged.emit(new_voltage)
        return 0

    def _goto_during_scan(self, voltage=None, laser_index=1):
        if voltage is None:
            return -1

        #if abs(self.get_current_voltage()-voltage)<0.01: #avoid a strange situation where we get an error
        #    goto_ramp = self._generate_ramp(self.get_current_voltage(), voltage+0.05, 0.75)

        #    ignored_counts = self._scan_line(goto_ramp)

        goto_ramp = self._generate_ramp(self.get_current_voltage(laser_index), voltage, 0.75, laser_index)
        if len(goto_ramp[0])==1:
            # this fix works only in this case, it is not meant to work when another channel than the last one (the 4th)
            goto_ramp=np.hstack((goto_ramp,goto_ramp))
            goto_ramp[3,1]=voltage
        
        ignored_counts = self._scan_line(goto_ramp)

        return 0
        

    def set_clock_frequency(self, clock_frequency, laser_index=1):
        """Sets the frequency of the clock

        @param int clock_frequency: desired frequency of the clock

        @return int: error code (0:OK, -1:error)
        """
        if laser_index == 1:
            self._clock_frequency = float(clock_frequency)
        elif laser_index == 2:
            self._clock_frequency_2 = float(clock_frequency)
        # checks if scanner is still running
        if self.module_state() == 'locked':
            return -1
        else:
            return 0

    def set_resolution(self, stepsize, laser_index=1):
        """ Calculate clock rate from scan speed and desired number of pixels """
        if laser_index == 1:
            self.stepsize = stepsize
            scan_range = abs(self.scan_range[1] - self.scan_range[0])
            if stepsize == 0:
                print("DuoPLE1: Stepsize = 0 is invalid.")
                stepsize = 0.01
            self.resolution = int(scan_range/stepsize)
            duration = scan_range / self._scan_speed
            new_clock = self.resolution / duration
        elif laser_index == 2:
            self.stepsize_2 = stepsize
            scan_range = abs(self.scan_range_2[1] - self.scan_range_2[0])
            if stepsize == 0:
                print("DuoPLE2: Stepsize = 0 is invalid.")
                stepsize = 0.01
            self.resolution_2 = int(scan_range/stepsize)
            duration = scan_range / self._scan_speed_2
            new_clock = self.resolution_2 / duration
        return self.set_clock_frequency(new_clock, laser_index)

    # def set_resolution(self, resolution):
    #     """ Calculate clock rate from scan speed and desired number of pixels """
    #     self.resolution = resolution
    #     scan_range = abs(self.scan_range[1] - self.scan_range[0])
    #     duration = scan_range / self._scan_speed
    #     new_clock = resolution / duration
    #     return self.set_clock_frequency(new_clock)

    def set_scan_range(self, scan_range, laser_index=1):
        """ Set the scan range """
        r_max = np.clip(scan_range[1], self.a_range[laser_index-1][0], self.a_range[laser_index-1][1])
        r_min = np.clip(scan_range[0], self.a_range[laser_index-1][0], r_max)
        if laser_index == 1:
            self.scan_range = [r_min, r_max]
        elif laser_index == 2:
            self.scan_range_2 = [r_min, r_max]

    def set_voltage(self, volts, laser_index=1):
        """ Set the channel idle voltage """
        if laser_index == 1:
            self._static_v = np.clip(volts, self.a_range[laser_index-1][0], self.a_range[laser_index-1][1])
            self.goto_voltage(self._static_v, laser_index)
        elif laser_index == 2:
            self._static_v_2 = np.clip(volts, self.a_range[laser_index-1][0], self.a_range[laser_index-1][1])
            self.goto_voltage(self._static_v_2, laser_index)

    def set_scan_speed(self, scan_speed, laser_index=1):
        """ Set scan speed in volt per second """
        if laser_index == 1:
            self._scan_speed = np.clip(scan_speed, 1e-9, 1e6)
            self._goto_speed = self._scan_speed
        elif laser_index == 2:
            self._scan_speed_2 = np.clip(scan_speed, 1e-9, 1e6)
            self._goto_speed_2 = self._scan_speed_2
            

    def set_arc_factor(self, arc_factor=None):
        # """ Set scan speed in volt per second """
        if arc_factor:
            self.arc_factor = arc_factor
        else:
            if self.laser_choice == 0:
                if self.tuning_mode == 0:
                    self.arc_factor = self.arc_factor_ctl_pc
                elif self.tuning_mode == 1:
                    self.arc_factor = self.arc_factor_ctl_mc
            elif self.laser_choice == 1:
                self.arc_factor = self.arc_factor_dlcpro
        
        self.ARCFactorChanged.emit(self.arc_factor,self.laser_choice)
        
        self.write_arc_factor(self.arc_factor)

    def write_arc_factor(self, arc_factor):
            if self.laser_choice == 0:
                if self.tuning_mode == 0:
                    self._ctl_logic.ext_factor(arc_factor)
                elif self.tuning_mode == 1:
                    self._ctl_logic.remote_factor(arc_factor)
            elif self.laser_choice == 1:
                self._dlcpro_logic.ext_factor(arc_factor)

    def set_ctl_wavelength(self, ctl_wavelength):
        # """ Set scan speed in volt per second """
        self._ctl_logic.wavelength(ctl_wavelength)
        QtTest.QTest.qSleep(1000)
        self.ctl_wavelength = self._ctl_logic._laser.wavelength_act
        self.sigUpdateWavlength.emit(1)

    def set_show_lines(self, show_lines, laser_index=1):
        if laser_index == 1:
            self.number_of_lines = int(np.clip(show_lines, 1, 1e6))
        elif laser_index == 2:
            self.number_of_lines_2 = int(np.clip(show_lines, 1, 1e6))
    
    def set_scan_lines(self, scan_lines, laser_index=1):
        if laser_index == 1:
            self.number_of_repeats = int(np.clip(scan_lines, 1, 1e6))
        elif laser_index == 2:
            self.number_of_repeats_2 = int(np.clip(scan_lines, 1, 1e6))

    def _initialise_data_matrix(self, scan_length, laser_index=1):
        """ Initializing the matrix plot. """
        if laser_index == 1:
            self.scan_matrix = np.zeros((self.number_of_repeats, scan_length))
            self.plot_matrix = np.zeros((self.number_of_lines, scan_length))
            self.plot_x = np.linspace(self.scan_range[0], self.scan_range[1], scan_length)
            self.plot_x_frequency=self.plot_x*self.factor_V_to_GHz #1000 MHz equals 0.22 V on the PLE x range with FeedForward on # 1000 MHz equals 0.30 V on the PLE x range without FeedForward
            self.plot_y = np.zeros(scan_length)
            self.fit_data = np.zeros(scan_length)
            self.fit_x = np.linspace(self.scan_range[0], self.scan_range[1], scan_length)
            self.fit_y = np.zeros(scan_length)
        elif laser_index == 2:
            # laser 2:
            self.scan_matrix_2 = np.zeros((self.number_of_repeats_2, scan_length))
            self.plot_matrix_2 = np.zeros((self.number_of_lines_2, scan_length))
            self.plot_x_2 = np.linspace(self.scan_range_2[0], self.scan_range_2[1], scan_length)
            self.plot_x_frequency_2=self.plot_x_2*self.factor_V_to_GHz_2 #1000 MHz equals 0.22 V on the PLE x range with FeedForward on # 1000 MHz equals 0.30 V on the PLE x range without FeedForward
            self.plot_y_2 = np.zeros(scan_length)
            self.fit_data_2 = np.zeros(scan_length)
            self.fit_x_2 = np.linspace(self.scan_range_2[0], self.scan_range_2[1], scan_length)
            self.fit_y_2 = np.zeros(scan_length)
        return

    def get_current_voltage(self, laser_index=1):
        """returns current voltage of hardware device(atm NIDAQ 4th output)"""
        if laser_index == 1:
            return self._scanning_device.get_scanner_position()[3]
        elif laser_index == 2:
            return self._scanning_device.get_scanner_position()[2]

    def _initialise_scanner(self, laser_index=1):
        """Initialise the clock and locks for a scan"""
        self.module_state.lock()
        self._scanning_device.module_state.lock()

        if laser_index == 1:
            returnvalue = self._scanning_device.set_up_scanner_clock(
                clock_frequency=self._clock_frequency)
        elif laser_index == 2:
            returnvalue = self._scanning_device.set_up_scanner_clock(
                clock_frequency=self._clock_frequency_2)
            
        if returnvalue < 0:
            self._scanning_device.module_state.unlock()
            self.module_state.unlock()
            self.set_position('scanner')
            return -1

        returnvalue = self._scanning_device.set_up_scanner()
        if returnvalue < 0:
            self._scanning_device.module_state.unlock()
            self.module_state.unlock()
            self.set_position('scanner')
            return -1

        return 0

    def start_scanning(self, v_min=None, v_max=None):
        """Setting up the scanner device and starts the scanning procedure
        @return int: error code (0:OK, -1:error)
        """
        self._awg.mcas_dict.stop_awgs()
        self.laser_at_position = False
        self.measurement_running = True
        # print(self._scanning_device.module_state()) #self._scanning_device.module_state() is not the same as _optimizer_logic.module_state.current! So _s_d.module_state is "idle" even when confocal refocus runs.
        # if self._scanning_device.module_state() == 'locked':
        #     logger.error('Nicard is in module state "locked".')
        #     raise Exception    

        print("Start DuoPLE1 scan...")
        self.ple_progressBar.emit(0,100,1)

        # for laser integration:
        self.ctl_wavelength = self._ctl_logic._laser.wavelength_act
        self.sigUpdateWavlength.emit(1)
        self.check_emission()
        if self.ctl_emission == False:
            print("DuoPLE1: CAREFULL! Resonant laser disabled!")
            return
        # QtTest.QTest.qSleep(200)
        if self.tuning_mode == 0:
            self._ctl_logic.remote_enabled(False)
            self._ctl_logic.pc_enabled(True)
        elif self.tuning_mode == 1:
            self._ctl_logic.pc_enabled(False)
            self._ctl_logic.remote_enabled(True)

        # check if nidaq is not already used
        if not(self._scanning_device._scanner_counter_daq_tasks==[] or self._scanning_device._scanner_clock_daq_task==None):
        #if not(self._scanning_device._scanner_counter_daq_tasks==[] or self._scanning_device._scanner_analog_daq_task==None or self._scanning_device._scanner_clock_daq_task==None or self._scanning_device._clock_daq_task==[] or self._scanning_device._counter_daq_tasks ==None):
            print("DuoPLE1: Another scanner is running. Please try pressing 'Run' again. I will continue with current scanner task")
            return

        QtTest.QTest.qSleep(200)
        self.trace_seq(1)
        if self.enable_PulsedRepump:
            #self.ps.stream(seq=[[int(self.RepumpDuration*1e3),["repump"],0,0],[int(self.RepumpDecay*1e3),[],0,0]],n_runs=1) #self.RepumpDuration is in µs
            QtTest.QTest.qSleep(200)
            self._awg.mcas_dict.stop_awgs()
            QtTest.QTest.qSleep(200)
            self.setup_repump(1)
            self._awg.mcas_dict['ple_repump'].run()
        if self.enable_PulsedCTL:
            QtTest.QTest.qSleep(200)
            self._awg.mcas_dict.stop_awgs()
            QtTest.QTest.qSleep(200)
            self.setup_ctl(1)
            self._awg.mcas_dict['ple_ctl'].run()
        if self.enable_PulsedBlue:
            QtTest.QTest.qSleep(200)
            self._awg.mcas_dict.stop_awgs()
            QtTest.QTest.qSleep(200)
            self.setup_blue(1)
            self._awg.mcas_dict['ple_blue'].run()
        self.current_position = self._scanning_device.get_scanner_position() #Never used

        if v_min is not None:
            self.scan_range[0] = v_min
        else:
            v_min = self.scan_range[0]
        if v_max is not None:
            self.scan_range[1] = v_max
        else:
            v_max = self.scan_range[1]

        # # in case of motor scan we have to switch the voltages:
        # if (self.laser_choice == 0) and (self.tuning_mode == 1):
        #     v_n = v_min.copy()
        #     v_x = v_max.copy()
        #     v_min = v_x
        #     v_max = v_n
        

        self._scan_counter_up = 0
        self.upwards_scan = True

        self.set_resolution(self.stepsize,1)
        
        self._upwards_ramp = self._generate_ramp(v_min, v_max, self._scan_speed, 1)
        self._downwards_ramp = self._generate_ramp(v_max, v_min, 0.75, 1)
        
        self.local_counts = []
        self.timestamp_list=[]
        self.f_start_end=[]
        self.xy_data=[]
        self.range_defined=False

        self._initialise_data_matrix(len(self._upwards_ramp[3]),1)
        
        # Lock and set up scanner
        returnvalue = self._initialise_scanner()
        if returnvalue < 0:
            # TODO: error message
            logging.error('The scanner was not initialised')
            return -1

        self.stopped=False
        self.start_stop_timestamps=[]
        self.start_list=[]
        self.stop_list=[]
        # initialize wavemeter data
        self.starttime = time.time()

        self._acqusition_start_time = time.time()
        self.setup_seq(sequence_name=self.curr_sequence_name)
        self._awg.mcas_dict[self.curr_sequence_name].run()
        self.sigScanNextLine.emit()
        self.sigScanStarted.emit(1)
        return 0
    
    def start_scanning_2(self, v_min=None, v_max=None):
        """Setting up the scanner device and starts the scanning procedure
        @return int: error code (0:OK, -1:error)
        """
        self._awg.mcas_dict.stop_awgs()
        self.laser_at_position = False
        self.measurement_running_2 = True
        # print(self._scanning_device.module_state()) #self._scanning_device.module_state() is not the same as _optimizer_logic.module_state.current! So _s_d.module_state is "idle" even when confocal refocus runs.
        # if self._scanning_device.module_state() == 'locked':
        #     logger.error('Nicard is in module state "locked".')
        #     raise Exception    

        print("Start DuoPLE2 scan...")
        self.ple_progressBar.emit(0,100,2)

        # for laser integration:
        self.ctl_wavelength = self._ctl_logic._laser.wavelength_act
        self.sigUpdateWavlength.emit(2)
        self.check_emission()
        if self.dlcpro_emission == False:
            print("DuoPLE2: CAREFULL! Resonant laser disabled!")
            return
        self._dlcpro_logic.ext_enabled(True)

        # check if nidaq is not already used
        if not(self._scanning_device._scanner_counter_daq_tasks==[] or self._scanning_device._scanner_clock_daq_task==None):
        #if not(self._scanning_device._scanner_counter_daq_tasks==[] or self._scanning_device._scanner_analog_daq_task==None or self._scanning_device._scanner_clock_daq_task==None or self._scanning_device._clock_daq_task==[] or self._scanning_device._counter_daq_tasks ==None):
            print("DuoPLE2: Another scanner is running. Please try pressing 'Run' again. I will continue with current scanner task")
            return

        QtTest.QTest.qSleep(200)
        self.trace_seq(2)
        if self.enable_PulsedRepump_2:
            #self.ps.stream(seq=[[int(self.RepumpDuration*1e3),["repump"],0,0],[int(self.RepumpDecay*1e3),[],0,0]],n_runs=1) #self.RepumpDuration is in µs
            QtTest.QTest.qSleep(200)
            self._awg.mcas_dict.stop_awgs()
            QtTest.QTest.qSleep(200)
            self.setup_repump(2)
            self._awg.mcas_dict['ple_repump'].run()
        if self.enable_PulsedCTL_2:
            QtTest.QTest.qSleep(200)
            self._awg.mcas_dict.stop_awgs()
            QtTest.QTest.qSleep(200)
            self.setup_ctl(2)
            self._awg.mcas_dict['ple_ctl'].run()
        if self.enable_PulsedBlue_2:
            QtTest.QTest.qSleep(200)
            self._awg.mcas_dict.stop_awgs()
            QtTest.QTest.qSleep(200)
            self.setup_blue(2)
            self._awg.mcas_dict['ple_blue'].run()
        self.current_position = self._scanning_device.get_scanner_position() #Never used

        if v_min is not None:
            self.scan_range_2[0] = v_min
        else:
            v_min = self.scan_range_2[0]
        if v_max is not None:
            self.scan_range_2[1] = v_max
        else:
            v_max = self.scan_range_2[1]

        # # in case of motor scan we have to switch the voltages:
        # if (self.laser_choice == 0) and (self.tuning_mode == 1):
        #     v_n = v_min.copy()
        #     v_x = v_max.copy()
        #     v_min = v_x
        #     v_max = v_n
        

        self._scan_counter_up_2 = 0
        self.upwards_scan = True

        self.set_resolution(self.stepsize_2,2)
        
        self._upwards_ramp = self._generate_ramp(v_min, v_max, self._scan_speed, 2)
        self._downwards_ramp = self._generate_ramp(v_max, v_min, 0.75, 2)
        
        self.local_counts_2 = []
        self.timestamp_list_2=[]
        self.f_start_end_2=[]
        self.xy_data_2=[]
        self.range_defined_2=False

        self._initialise_data_matrix(len(self._upwards_ramp[3]),2)
        
        # Lock and set up scanner
        returnvalue = self._initialise_scanner()
        if returnvalue < 0:
            # TODO: error message
            logging.error('The scanner was not initialised')
            return -1

        self.stopped=False
        self.start_stop_timestamps_2=[]
        self.start_list_2=[]
        self.stop_list_2=[]
        # initialize wavemeter data
        self.starttime_2 = time.time()

        self._acqusition_start_time_2 = time.time()
        self.setup_seq(sequence_name=self.curr_sequence_name)
        self._awg.mcas_dict[self.curr_sequence_name].run()
        self.sigScanNextLine_2.emit()
        self.sigScanStarted.emit(2)
        return 0

    def stop_scanning(self):
        """Stops the scan

        @return int: error code (0:OK, -1:error)
        """
        with self.threadlock:
            print("threadlock laserscanning")
            if self.module_state() == 'locked':
                print("i'm locked")
                self._close_scanner() # hope this will not destroy something
                self._initialise_scanner()
                self.stopRequested = True
                self.stopRequested_2 = True
                #self._do_next_line()
        return 0

    def abort_scanning(self):
        with self.threadlock:
            if self.module_state() == 'locked':
                self.AbortRequested = True
        return 0


    def _close_scanner(self):
        """Close the scanner and unlock"""
        with self.threadlock:
            self.kill_scanner()
            self.stopRequested = False
            self.stopRequested_2 = False
            self.AbortRequested = False
            if self.module_state.can('unlock'):
                self.module_state.unlock()

    def _do_next_line(self):
        """ If stopRequested then finish the scan, otherwise perform next repeat of the scan line
        """
        self.updateProgressBarSignal.emit(1)
        # stops scanning
        if (self.stopRequested or (self._scan_counter_up >= self.number_of_repeats and self.number_of_repeats!=0) or ((time.time()-self.starttime) >= self.stoptime and self.stoptime!=0)):
            self._goto_during_scan(self._static_v,laser_index=1)
            self._close_scanner()
            # self.sigScanFinished.emit(1)
            # self.local_counts=[]
            # self.slice_number=0
            self._awg.mcas_dict.stop_awgs()
            self.local_counts=[]

            self.measured_frequencies=[]
            self.measured_frequencies=np.array(self._wavemeterlogic._hardware_pull._parentclass._wavelength_data.copy())
            if not self._wavemeterlogic._hardware_pull._parentclass._wavelength_data == []:
                self.measured_frequencies = np.array(list(zip(self.measured_frequencies[:,0] - self.reset_wavemeter_time,self.measured_frequencies[:,1])))
            # only emits here if no laser locking is on
            time.sleep(0.1)
            # if not self.lock_laser or (self.lock_laser and self.laser_at_position):
            self.measurement_running = False
            print("DuoPLE1: sigScanFinished emit")
            self.sigScanFinished.emit(1)
            if not self._wavemeterlogic._hardware_pull._parentclass._wavelength_data == []:
                if self.automatic_calibration:
                    try:
                        self._estimate_ratio()
                    except TypeError as e:
                        print("DuoPLE1: ",e)
            else:
                print("DuoPLE1: COULD NOT ESTIMATE RATIO- CAREFULL - WAVEMETER TRAGGING IS WAS NOT RUNNING! YOU MIGHT HAVE TO RESTART QUDI!")
            return

        if self._scan_counter_up == 0:
            # move from current voltage to ende of scan range, to simulate similar conditions for the 1st scan compared to the following. Also improves conversion estimation.
            # self._goto_during_scan(self.scan_range[1],laser_index=1) # this seems to be the origin of shifting PLE laser, but why? it worked before...
            # QtTest.QTest.qSleep(1000)
            if not self._wavemeterlogic._hardware_pull._parentclass._wavelength_data == []:
                self.reset_wavemeter_time = self._wavemeterlogic._hardware_pull._parentclass._wavelength_data[-1][0]
            else:
                print("DuoPLE1: CAREFULL - WAVEMETER TRAGGING IS NOT RUNNING! YOU MIGHT HAVE TO RESTART QUDI!")
                print("DuoPLE1: CAREFULL - WAVEMETER TRAGGING IS NOT RUNNING! YOU MIGHT HAVE TO RESTART QUDI!")
            self._wavemeterlogic._hardware_pull._parentclass._wavelength_data=[]
            

        if self.upwards_scan:
            # print("running seq: ", self.curr_sequence_name) # Does it affect the sequence if it is started repeatedly?
            #self._awg.mcas_dict.stop_awgs()
            if (self.enable_PulsedRepump and not self.RepumpWhenIonized) or (self.enable_PulsedRepump and self.RepumpWhenIonized and self.ionized) or (self.enable_PulsedCTL) or (self.enable_Blue): 
                # only need to setup sequence again, if repump was used before.
                self.setup_seq(sequence_name=self.curr_sequence_name)
            self._awg.mcas_dict[self.curr_sequence_name].run()
            
            fstart=self._wavemeterlogic._wavemeter_device.get_current_wavelength(kind ='air', ch = self._wavemeterlogic._wavemeter_device.channel1)
            self.start_list.append(time.time()-self._acqusition_start_time)
            counts = self._scan_line(self._upwards_ramp)
            fend=self._wavemeterlogic._wavemeter_device.get_current_wavelength(kind ='air', ch = self._wavemeterlogic._wavemeter_device.channel1)
            self.f_start_end.append([fstart,fend,len(counts)])

            #get the timestamp from the wavemeter
            self.timestamp_list.append(self._scanning_device.timestamp_list)
                
            self.local_counts=self.local_counts+list(counts)
            if self.mode==0:
                freqs=np.concatenate([np.linspace(fstart,fend,length) for fstart,fend,length in self.f_start_end])
            else:
                # calculate the frequencies from the wavelengths measured
                freq_data=self._wavemeterlogic._hardware_pull._parentclass._wavelength_data.copy()
                wavelengths=np.array(freq_data)[:,1]
                times=np.array(freq_data)[:,0]
                freqs=np.concatenate([np.interp(Ts,xp=times,fp=wavelengths) for Ts in self.timestamp_list])

            cts=self.local_counts
            self.stop_list.append(time.time()-self._acqusition_start_time)
            #TODO get the real fmin,fmax,points
            if not self.range_defined:
                self.fmin=min(freqs)
                self.fmax=max(freqs)
                self.range_defined=True

            self.f_range=np.linspace(self.fmin,self.fmax,len(self.local_counts)) #this will be the x axis for all the scans #do not calculate it here, t least not fmin and fmax
            self.xy_data.append([freqs,cts])

            # because freqs might have a lesser span than fmin->fmax, we artificially create the two extrema data points for the interpolation to work
            if min(freqs)>self.fmin:
                freqs=np.concatenate(([self.fmin],freqs))
                cts=np.concatenate(([0],cts))
                
            if max(freqs)<self.fmax:
                freqs=np.concatenate((freqs,[self.fmax]))
                cts=np.concatenate((cts,[0]))

            self.interpolated_cts=np.interp(self.f_range, xp=freqs, fp=cts)
            
            #self.tmp_storage.append(self.f_start_end.copy())
            self.timestamp_list=[] #reinitialize the timestamp_list
            self.f_start_end=[]

            #old code
            if self.scan_matrix.shape[0]< self.number_of_repeats:
                add=np.zeros((int(self.number_of_repeats-self.scan_matrix.shape[0]),self.scan_matrix.shape[1]))
                self.scan_matrix=np.vstack((self.scan_matrix,add))
            elif self.scan_matrix.shape[0]> self.number_of_repeats and self.number_of_repeats!=0:
                # self.scan_matrix=self.scan_matrix[:int(self.number_of_lines)]
                pass
            if self.plot_matrix.shape[0]< self.number_of_lines:
                add=np.zeros((int(self.number_of_lines-self.plot_matrix.shape[0]),self.plot_matrix.shape[1]))
                self.plot_matrix=np.vstack((self.plot_matrix,add))
            elif self.plot_matrix.shape[0]> self.number_of_lines and self.number_of_lines!=0:
                self.plot_matrix=self.plot_matrix[:int(self.number_of_lines)]

            # adds new values to the beginning and shifts others 1 behind
            self.scan_matrix[1:]=self.scan_matrix[0:-1]
            self.scan_matrix[0]=self.local_counts
            self.plot_matrix[1:]=self.plot_matrix[0:-1]
            self.plot_matrix[0]=self.local_counts
        
            #self.scan_matrix[self._scan_counter_up] =self.local_counts # Here occurs an error "cannot copy sequence with size 21 to array axis with dimension 20". This only occured, when variables where changed while the programm is running. 
            self.plot_y = self.plot_y + np.array(self.local_counts)
        
            # # new code
            # self.scan_matrix[self._scan_counter_up] =self.interpolated_cts
            # self.plot_y = self.plot_y + np.array(self.interpolated_cts)
            self.check_if_ionized(self.scan_matrix[0])
            self.local_counts=[]
            self._scan_counter_up += 1
            self.upwards_scan = False
            
        else: #retrace
            self.sigUpdatePlots.emit(1)
            if (self.enable_PulsedRepump and not self.RepumpWhenIonized) or (self.enable_PulsedRepump and self.RepumpWhenIonized and self.ionized):
                self._awg.mcas_dict.stop_awgs()
                QtTest.QTest.qSleep(200)
                self.setup_repump(1)
                self._awg.mcas_dict['ple_repump'].run()
            if self.enable_PulsedCTL:
                self._awg.mcas_dict.stop_awgs()
                QtTest.QTest.qSleep(200)
                self.setup_ctl(1)
                self._awg.mcas_dict['ple_ctl'].run()
            if self.enable_PulsedBlue:
                self._awg.mcas_dict.stop_awgs()
                QtTest.QTest.qSleep(200)
                self.setup_blue(1)
                self._awg.mcas_dict['ple_blue'].run()
            counts = self._scan_line(self._downwards_ramp)
            self.upwards_scan = True
            self.start_stop_timestamps=[]
        self.sigScanNextLine.emit()
        self.updateProgressBarSignal.emit(1)
    # Laser 2:
    def _do_next_line_2(self):
        """ If stopRequested then finish the scan, otherwise perform next repeat of the scan line
        """
        self.updateProgressBarSignal.emit(2)
        # stops scanning
        if (self.stopRequested_2 or (self._scan_counter_up_2 >= self.number_of_repeats_2 and self.number_of_repeats_2!=0) or ((time.time()-self.starttime_2) >= self.stoptime_2 and self.stoptime_2!=0)):
            self._goto_during_scan(self._static_v_2,laser_index=2)
            self._close_scanner()
            # self.sigScanFinished.emit(2)
            # self.slice_number=0
            self._awg.mcas_dict.stop_awgs()
            self.local_counts_2=[]

            self.measured_frequencies_2=[]
            self.measured_frequencies_2=np.array(self._wavemeterlogic._hardware_pull._parentclass._wavelength_data.copy())
            if not self._wavemeterlogic._hardware_pull._parentclass._wavelength_data == []:
                self.measured_frequencies_2 = np.array(list(zip(self.measured_frequencies_2[:,0] - self.reset_wavemeter_time,self.measured_frequencies_2[:,1])))
            # only emits here if no laser locking is on
            time.sleep(0.1)
            # if not self.lock_laser or (self.lock_laser and self.laser_at_position):
            self.measurement_running_2 = False
            print("DuoPLE2: sigScanFinished emit")
            self.sigScanFinished.emit(2)
            if not self._wavemeterlogic._hardware_pull._parentclass._wavelength_data == []:
                if self.automatic_calibration:
                    try:
                        self._estimate_ratio()
                    except TypeError as e:
                        print(e)
            else:
                print("DuoPLE2: COULD NOT ESTIMATE RATIO- CAREFULL - WAVEMETER TRAGGING IS WAS NOT RUNNING! YOU MIGHT HAVE TO RESTART QUDI!")
            return

        if self._scan_counter_up_2 == 0:
            # move from current voltage to ende of scan range, to simulate similar conditions for the 1st scan compared to the following. Also improves conversion estimation.
            self._goto_during_scan(self.scan_range[1],laser_index=2)
            if not self._wavemeterlogic._hardware_pull._parentclass._wavelength_data == []:
                self.reset_wavemeter_time = self._wavemeterlogic._hardware_pull._parentclass._wavelength_data[-1][0]
            else:
                print("DuoPLE2: CAREFULL - WAVEMETER TRAGGING IS NOT RUNNING! YOU MIGHT HAVE TO RESTART QUDI!")
                print("DuoPLE2: CAREFULL - WAVEMETER TRAGGING IS NOT RUNNING! YOU MIGHT HAVE TO RESTART QUDI!")
            self._wavemeterlogic._hardware_pull._parentclass._wavelength_data=[]
            

        if self.upwards_scan:
            # print("running seq: ", self.curr_sequence_name) # Does it affect the sequence if it is started repeatedly?
            #self._awg.mcas_dict.stop_awgs()
            if (self.enable_PulsedRepump_2 and not self.RepumpWhenIonized_2) or (self.enable_PulsedRepump_2 and self.RepumpWhenIonized_2 and self.ionized) or (self.enable_PulsedCTL_2) or (self.enable_Blue_2): 
                # only need to setup sequence again, if repump was used before.
                self.setup_seq(sequence_name=self.curr_sequence_name)
            self._awg.mcas_dict[self.curr_sequence_name].run()
            
            fstart=self._wavemeterlogic._wavemeter_device.get_current_wavelength(kind ='air', ch = self._wavemeterlogic._wavemeter_device.channel1)
            self.start_list_2.append(time.time()-self._acqusition_start_time_2)
            counts = self._scan_line(self._upwards_ramp)
            fend=self._wavemeterlogic._wavemeter_device.get_current_wavelength(kind ='air', ch = self._wavemeterlogic._wavemeter_device.channel1)
            self.f_start_end_2.append([fstart,fend,len(counts)])

            #get the timestamp from the wavemeter
            self.timestamp_list_2.append(self._scanning_device.timestamp_list)
                
            self.local_counts_2=self.local_counts_2+list(counts)
            if self.mode==0:
                freqs=np.concatenate([np.linspace(fstart,fend,length) for fstart,fend,length in self.f_start_end_2])
            else:
                # calculate the frequencies from the wavelengths measured
                freq_data=self._wavemeterlogic._hardware_pull._parentclass._wavelength_data.copy()
                wavelengths=np.array(freq_data)[:,1]
                times=np.array(freq_data)[:,0]
                freqs=np.concatenate([np.interp(Ts,xp=times,fp=wavelengths) for Ts in self.timestamp_list_2])

            cts=self.local_counts_2
            self.stop_list_2.append(time.time()-self._acqusition_start_time_2)
            #TODO get the real fmin,fmax,points
            if not self.range_defined_2:
                self.fmin=min(freqs)
                self.fmax=max(freqs)
                self.range_defined_2=True

            self.f_range_2=np.linspace(self.fmin,self.fmax,len(self.local_counts_2)) #this will be the x axis for all the scans #do not calculate it here, t least not fmin and fmax
            self.xy_data_2.append([freqs,cts])

            # because freqs might have a lesser span than fmin->fmax, we artificially create the two extrema data points for the interpolation to work
            if min(freqs)>self.fmin:
                freqs=np.concatenate(([self.fmin],freqs))
                cts=np.concatenate(([0],cts))
                
            if max(freqs)<self.fmax:
                freqs=np.concatenate((freqs,[self.fmax]))
                cts=np.concatenate((cts,[0]))

            self.interpolated_cts_2=np.interp(self.f_range_2, xp=freqs, fp=cts)
            
            #self.tmp_storage.append(self.f_start_end.copy())
            self.timestamp_list_2=[] #reinitialize the timestamp_list
            self.f_start_end_2=[]

            #old code
            if self.scan_matrix_2.shape[0]< self.number_of_repeats_2:
                add=np.zeros((int(self.number_of_repeats_2-self.scan_matrix_2.shape[0]),self.scan_matrix_2.shape[1]))
                self.scan_matrix_2=np.vstack((self.scan_matrix_2,add))
            elif self.scan_matrix_2.shape[0]> self.number_of_repeats_2 and self.number_of_repeats_2!=0:
                # self.scan_matrix=self.scan_matrix[:int(self.number_of_lines)]
                pass
            if self.plot_matrix_2.shape[0]< self.number_of_lines_2:
                add=np.zeros((int(self.number_of_lines_2-self.plot_matrix_2.shape[0]),self.plot_matrix_2.shape[1]))
                self.plot_matrix_2=np.vstack((self.plot_matrix_2,add))
            elif self.plot_matrix_2.shape[0]> self.number_of_lines_2 and self.number_of_lines_2!=0:
                self.plot_matrix_2=self.plot_matrix_2[:int(self.number_of_lines_2)]

            # adds new values to the beginning and shifts others 1 behind
            self.scan_matrix_2[1:]=self.scan_matrix_2[0:-1]
            self.scan_matrix_2[0]=self.local_counts_2
            self.plot_matrix_2[1:]=self.plot_matrix_2[0:-1]
            self.plot_matrix_2[0]=self.local_counts_2
        
            #self.scan_matrix[self._scan_counter_up] =self.local_counts_2 # Here occurs an error "cannot copy sequence with size 21 to array axis with dimension 20". This only occured, when variables where changed while the programm is running. 
            self.plot_y_2 = self.plot_y_2 + np.array(self.local_counts_2)
        
            # # new code
            # self.scan_matrix[self._scan_counter_up] =self.interpolated_cts
            # self.plot_y = self.plot_y + np.array(self.interpolated_cts)
            self.check_if_ionized(self.scan_matrix_2[0])
            self.local_counts_2=[]
            self._scan_counter_up_2 += 1
            self.upwards_scan = False
            
        else: #retrace
            self.sigUpdatePlots.emit(2)
            if (self.enable_PulsedRepump_2 and not self.RepumpWhenIonized_2) or (self.enable_PulsedRepump_2 and self.RepumpWhenIonized_2 and self.ionized):
                self._awg.mcas_dict.stop_awgs()
                QtTest.QTest.qSleep(200)
                self.setup_repump(2)
                self._awg.mcas_dict['ple_repump'].run()
            if self.enable_PulsedCTL_2:
                self._awg.mcas_dict.stop_awgs()
                QtTest.QTest.qSleep(200)
                self.setup_ctl(2)
                self._awg.mcas_dict['ple_ctl'].run()
            if self.enable_PulsedBlue_2:
                self._awg.mcas_dict.stop_awgs()
                QtTest.QTest.qSleep(200)
                self.setup_blue(2)
                self._awg.mcas_dict['ple_blue'].run()
            counts = self._scan_line(self._downwards_ramp)
            self.upwards_scan = True
            self.start_stop_timestamps_2=[]
        self.sigScanNextLine_2.emit()
        self.updateProgressBarSignal.emit(2)

    def check_if_ionized(self, cts):
        self.ionized=not(np.sum(np.asarray(cts)>self.peak_min_counts))
        if self.ionized:
            print("Ionized in PLE")
            self.SigIonized.emit()

    
    @property
    def MW1_Power(self):
        return self._MW1_Power

    @MW1_Power.setter
    def MW1_Power(self,val):
        self._MW1_Power=val

    def trace_seq(self, laser_index=1):
        self.setup_repump(laser_index)
        self.setup_ctl(laser_index)
        self.setup_blue(laser_index)
        hash = base64.b64encode(hashlib.sha1(self.convert_seq_params_to_string(laser_index).encode()).digest())
        #Added self.queue._gated_counter.readout_duration such that the hash recognizes a change in readout duration and will update n_values in the sequence accordingly
        self.curr_sequence_name = "ple_trace_hash_{}".format(hash)
        if self.hashed and not self.curr_sequence_name in self._awg.mcas_dict:
            print("hash in PLE: ", self.curr_sequence_name)
            self._awg.mcas_dict.stop_awgs()
            QtTest.QTest.qSleep(200)
            self.setup_seq(sequence_name=self.curr_sequence_name)

        elif self.hashed == False: 
            self._awg.mcas_dict.stop_awgs()
            QtTest.QTest.qSleep(200)
            self.curr_sequence_name = "ple_trace"
            self.setup_seq(sequence_name=self.curr_sequence_name)
        return
        
    def setup_seq(self, sequence_name):
        self.adv_mode = 'REP'
        self.power = []
        if self.enable_MW1:
            self.power += [self.MW1_Power]
        if self.enable_MW2:
            self.power += [self.MW2_Power]
        if self.enable_MW3:
            self.power += [self.MW3_Power]
        
        self.power = np.asarray(self.power)
        self.power=self.power_to_amp(self.power)
        if np.sum(self.power)>1:
            logger.error("Combined Microwavepower of all active channels too high! Need value below 1. Value of {} was given.", np.sum(self.power))
            raise Exception
            
        # generate a single MW sequence with the needed mw frequencies and play is continuously until the measurement is stopped,
        # either by the stop button, the runtime, or number of sequence repetitions.
        seq = self._awg.mcas(name=sequence_name, ch_dict={"2g": [1, 2], "ps": [1]})
        frequencies = np.array([self.MW1_Freq, self.MW2_Freq, self.MW3_Freq])[[self.enable_MW1, self.enable_MW2, self.enable_MW3]]
        seq.start_new_segment("Microwaves"+str(frequencies), loop_count=500)
        if len(self.power) == 0:
            seq.asc(name="without MW",
                    A1=self.enable_A1,
                    A2=self.enable_A2,
                    repump=self.enable_Repump,
                    CTL=self.enable_CTL,
                    Blue=self.enable_Blue,
                    length_mus=10
                    )
        else:
            seq.asc(name="with MW", pd2g1={"type": "sine", "frequencies": frequencies, "amplitudes": self.power},
                    A1=self.enable_A1,
                    A2=self.enable_A2,
                    gateMW=True,
                    repump=self.enable_Repump,
                    CTL=self.enable_CTL,
                    Blue=self.enable_Blue,
                    length_mus=10
                    )
        
        # if not "PLE_treace" in self._awg.mcas_dict.keys():
        self._awg.mcas_dict.stop_awgs()
        QtTest.QTest.qSleep(200)
        self._awg.mcas_dict[sequence_name] = seq
        #self._awg.mcas_dict.print_info()
        return
    
    # Laser 2:
    def setup_seq_2(self, sequence_name):
        self.adv_mode = 'REP'
        self.power = []
        if self.enable_MW1_2:
            self.power += [self.MW1_Power_2]
        if self.enable_MW2_2:
            self.power += [self.MW2_Power_2]
        if self.enable_MW3_2:
            self.power += [self.MW3_Power_2]
        
        self.power = np.asarray(self.power)
        self.power=self.power_to_amp(self.power)
        if np.sum(self.power)>1:
            logger.error("Combined Microwavepower of all active channels too high! Need value below 1. Value of {} was given.", np.sum(self.power))
            raise Exception
            
        # generate a single MW sequence with the needed mw frequencies and play is continuously until the measurement is stopped,
        # either by the stop button, the runtime, or number of sequence repetitions.
        seq = self._awg.mcas(name=sequence_name, ch_dict={"2g": [1, 2], "ps": [1]})
        frequencies = np.array([self.MW1_Freq_2, self.MW2_Freq_2, self.MW3_Freq_2])[[self.enable_MW1_2, self.enable_MW2_2, self.enable_MW3_2]]
        seq.start_new_segment("Microwaves"+str(frequencies), loop_count=500)
        if len(self.power) == 0:
            seq.asc(name="without MW",
                    A1=self.enable_A1_2,
                    A2=self.enable_A2_2,
                    repump=self.enable_Repump_2,
                    CTL=self.enable_CTL_2,
                    Blue=self.enable_Blue_2,
                    length_mus=10
                    )
        else:
            seq.asc(name="with MW", pd2g1={"type": "sine", "frequencies": frequencies, "amplitudes": self.power},
                    A1=self.enable_A1_2,
                    A2=self.enable_A2_2,
                    gateMW=True,
                    repump=self.enable_Repump_2,
                    CTL=self.enable_CTL_2,
                    Blue=self.enable_Blue_2,
                    length_mus=10
                    )
        
        # if not "PLE_treace" in self._awg.mcas_dict.keys():
        self._awg.mcas_dict.stop_awgs()
        QtTest.QTest.qSleep(200)
        self._awg.mcas_dict[sequence_name] = seq
        #self._awg.mcas_dict.print_info()
        return

    def setup_repump(self, laser_index=1):
        self.adv_mode = 'SING'
        seq = self._awg.mcas(name='ple_repump', ch_dict={"2g": [1, 2], "ps": [1]}, advance_mode=self.adv_mode)

        seq.start_new_segment("Repump", loop_count = 1, advance_mode=self.adv_mode)
        if laser_index == 1:
            seq.asc(repump=self.enable_PulsedRepump, length_mus=self.RepumpDuration)
        elif laser_index == 2:
            seq.asc(repump=self.enable_PulsedRepump_2, length_mus=self.RepumpDuration_2)
        
        # seq.start_new_segment("RepumpDec", advance_mode=self.adv_mode)
        # seq.asc(length_mus=self.RepumpDecay)
        
        self._awg.mcas_dict.stop_awgs()
        QtTest.QTest.qSleep(200)
        self._awg.mcas_dict['ple_repump'] = seq

    def setup_ctl(self, laser_index=1):
        self.adv_mode = 'SING'
        seq = self._awg.mcas(name='ple_ctl', ch_dict={"2g": [1, 2], "ps": [1]}, advance_mode=self.adv_mode)

        seq.start_new_segment("CTL", loop_count = 1, advance_mode=self.adv_mode)
        if laser_index == 1:
            seq.asc(CTL=self.enable_PulsedCTL, length_mus=self.CTLDuration)
        elif laser_index == 2:
            seq.asc(CTL=self.enable_PulsedCTL_2, length_mus=self.CTLDuration_2)
        
        # seq.start_new_segment("RepumpDec", advance_mode=self.adv_mode)
        # seq.asc(length_mus=self.RepumpDecay)
        
        self._awg.mcas_dict.stop_awgs()
        QtTest.QTest.qSleep(200)
        self._awg.mcas_dict['ple_ctl'] = seq

    def setup_blue(self, laser_index=1):
        self.adv_mode = 'SING'
        seq = self._awg.mcas(name='ple_blue', ch_dict={"2g": [1, 2], "ps": [1]}, advance_mode=self.adv_mode)

        seq.start_new_segment("Blue", loop_count = 1, advance_mode=self.adv_mode)
        if laser_index == 1:
            seq.asc(Blue=self.enable_PulsedBlue, length_mus=self.BlueDuration)
        elif laser_index == 2:
            seq.asc(Blue=self.enable_PulsedBlue_2, length_mus=self.BlueDuration_2)
        
        # seq.start_new_segment("RepumpDec", advance_mode=self.adv_mode)
        # seq.asc(length_mus=self.RepumpDecay)
        
        self._awg.mcas_dict.stop_awgs()
        QtTest.QTest.qSleep(200)
        self._awg.mcas_dict['ple_blue'] = seq
        
    def power_to_amp(self, power_dBm, impedance=50):
        power_dBm = np.atleast_1d(power_dBm)
        P_watts = 10**(power_dBm / 10) * 1e-3
        V_rms = np.sqrt(P_watts * impedance)
        V_pp = V_rms * 2 * np.sqrt(2)
        ou = self._awg.mcas_dict.awgs['2g'].ch[1].output_amplitude
        return V_pp / ou
       
    def _generate_ramp(self, voltage1, voltage2, speed, laser_index=1):
        """Generate a ramp vrom voltage1 to voltage2 that
        satisfies the speed, step, smoothing_steps parameters.  Smoothing_steps=0 means that the
        ramp is just linear.

        @param float voltage1: voltage at start of ramp.

        @param float voltage2: voltage at end of ramp.
        """

        # It is much easier to calculate the smoothed ramp for just one direction (upwards),
        # and then to reverse it if a downwards ramp is required.

        v_min = min(voltage1, voltage2)
        v_max = max(voltage1, voltage2)

        if v_min == v_max:
            ramp = np.array([v_min, v_max])
        else:
            # These values help simplify some of the mathematical expressions
            if laser_index == 1:
                linear_v_step = speed / self._clock_frequency
            elif laser_index == 2:
                linear_v_step = speed / self._clock_frequency_2
            smoothing_range = self._smoothing_steps + 1

            # Sanity check in case the range is too short

            # The voltage range covered while accelerating in the smoothing steps
            v_range_of_accel = sum(
                n * linear_v_step / smoothing_range for n in range(0, smoothing_range)
                )

            # Obtain voltage bounds for the linear part of the ramp
            v_min_linear = v_min + v_range_of_accel
            v_max_linear = v_max - v_range_of_accel

            if v_min_linear > v_max_linear:
                self.log.warning(
                    'Voltage ramp too short to apply the '
                    'configured smoothing_steps. A simple linear ramp '
                    'was created instead.')
                num_of_linear_steps = int(np.rint((v_max - v_min) / linear_v_step))
                ramp = np.linspace(v_min, v_max, num_of_linear_steps)

            else:
                #num_of_linear_steps = int(num_of_linear_steps) # FIXME. This was not needed before 13.10.2022. Should be already handled by np.rint
                num_of_linear_steps = int(np.rint((v_max_linear - v_min_linear) / linear_v_step))

                # Calculate voltage step values for smooth acceleration part of ramp
                smooth_curve = np.array(
                    [sum(
                        n * linear_v_step / smoothing_range for n in range(1, N)
                        ) for N in range(1, smoothing_range)
                    ])

                accel_part = v_min + smooth_curve
                decel_part = v_max - smooth_curve[::-1]

                linear_part = np.linspace(v_min_linear, v_max_linear, num_of_linear_steps)

                ramp = np.hstack((accel_part, linear_part, decel_part))

        # Reverse if downwards ramp is required
        if voltage2 < voltage1:
            ramp = ramp[::-1]

        # Put the voltage ramp into a scan line for the hardware (4-dimension)
        spatial_pos = self._scanning_device.get_scanner_position()
        print(f"DuoPLE{laser_index}: generate ramp in laserscannerlogic", spatial_pos)

        laser_index = 1
        if laser_index == 1:
            scan_line = np.vstack((
                np.ones((len(ramp), )) * spatial_pos[0],
                np.ones((len(ramp), )) * spatial_pos[1],
                np.ones((len(ramp), )) * spatial_pos[2],
                ramp
                ))
        elif laser_index == 2:
            scan_line = np.vstack((
                np.ones((len(ramp), )) * spatial_pos[0],
                np.ones((len(ramp), )) * spatial_pos[1],
                ramp,
                np.ones((len(ramp), )) * spatial_pos[3]
                ))
        return scan_line


    def _scan_line(self, line_to_scan=None):
        """do a single voltage scan from voltage1 to voltage2
        """
        # makes sure that there is more than one value for each channel, as the task require to have that
        if len(line_to_scan[0]) == 1:
            line_to_scan = np.tile(line_to_scan, (1,2))

        if line_to_scan is None:
            self.log.error('Voltage scanning logic needs a line to scan!')
            return -1
        try:
            # scan of a single line
            counts_on_scan_line = self._scanning_device.scan_line(line_to_scan)
            return counts_on_scan_line.transpose()[0]

        except Exception as e:
            self.log.error('The scan went wrong, killing the scanner.')
            self.stop_scanning()
            self.sigScanNextLine.emit()
            raise e

    def kill_scanner(self):
        """Closing the scanner device.

        @return int: error code (0:OK, -1:error)
        """
        try:
            self._scanning_device.close_scanner()
            self._scanning_device.close_scanner_clock()
        except Exception as e:
            self.log.exception('Could not even close the scanner, giving up.')
            raise e
        try:
            if self._scanning_device.module_state.can('unlock'):
                self._scanning_device.module_state.unlock()
        except:
            self.log.exception('Could not unlock scanning device.')
            return -1
        return 0

    def goto_fitted_peak(self):
        if self.lock_laser and self.module_state() != 'locked':
            freqs=np.array(self.Frequencies_Fit[:-1].replace("V","").split(";")[:-1]).astype(float)
            peak_volt=max(freqs)
            print("max counts of last row: ", np.max(self.xy_data[0][1]))
            # last scan ionized?
            not_ionized = (np.max(self.xy_data[0][1])>self.peak_min_counts)
            print("max counts", np.max(self.xy_data[0][1]))
            print("peak_min_counts", self.peak_min_counts)
            if (peak_volt<self.scan_range[1] and peak_volt>self.scan_range[0]):# and not_ionized: # check if there was a peak within the scan range, and last scan was not ionized
                self.volt_list.append(peak_volt)
                self.timstap_list.append(time.time())
                self._static_v = peak_volt
                self.goto_voltage(self._static_v,1)
                #self._wavemeterlogic._wavemeter_device.get_reference_course(self.Frequencies_Fit[-1], channel=2)

                # follow the defect PLE line by applying a voltage to the laser chamber
                #Range=self.scan_range[1]-self.scan_range[0]

                # select scan range depending on ComboBox scan_range_adjustment
                if self.scan_range_adjustment == 0:
                    self.scan_range[0],self.scan_range[1]=peak_volt-0.2,peak_volt+0.2
                if self.scan_range_adjustment == 1:
                    self.scan_range[0],self.scan_range[1]=peak_volt-0.5,peak_volt+0.4
                if self.scan_range_adjustment == 2:
                    pass
                    #self.scan_range[0],self.scan_range[1]=peak_volt-0.5*range,peak_volt+0.5*range
                self.sigScanRangeChanged.emit(self.scan_range[0],self.scan_range[1],1)
                self.laser_at_position = True
            else: 
                print("No PLE found in range. Retry with bigger scan range...")
                self.scan_range[0],self.scan_range[1]=self.scan_range[0]-1,self.scan_range[1]+1
                self.scan_range=list(np.clip(self.scan_range,-3,3))
                self.sigScanRangeChanged.emit(self.scan_range[0],self.scan_range[1],1)
                self.start_scanning()
        self.stopped=True
        print('goto_fitted_peak done')
        try:
            self.curr_freq = 299792458/ self._wavemeterlogic._wavemeter_device.get_current_wavelength(kind ='vac', ch = self._wavemeterlogic._wavemeter_device.channel1) # GHz
            self.current_freq = f"{self.curr_freq:.4f}"
            print(f"Laser now at {self.current_freq:.4f} GHz")
        except:
            pass
        return 0
    
    def goto_fitted_peak_2(self):
        if self.lock_laser_2 and self.module_state() != 'locked':
            freqs=np.array(self.Frequencies_Fit_2[:-1].replace("V","").split(";")[:-1]).astype(float)
            peak_volt=max(freqs)
            print("max counts of last row: ", np.max(self.xy_data_2[0][1]))
            # last scan ionized?
            not_ionized = (np.max(self.xy_data_2[0][1])>self.peak_min_counts)
            print("max counts", np.max(self.xy_data_2[0][1]))
            print("peak_min_counts", self.peak_min_counts)
            if (peak_volt<self.scan_range_2[1] and peak_volt>self.scan_range_2[0]):# and not_ionized: # check if there was a peak within the scan range, and last scan was not ionized
                self.volt_list.append(peak_volt)
                self.timstap_list.append(time.time())
                self._static_v_2 = peak_volt
                self.goto_voltage(self._static_v_2,2)
                #self._wavemeterlogic._wavemeter_device.get_reference_course(self.Frequencies_Fit[-1], channel=2)

                # follow the defect PLE line by applying a voltage to the laser chamber
                #Range=self.scan_range[1]-self.scan_range[0]

                # select scan range depending on ComboBox scan_range_adjustment
                if self.scan_range_adjustment_2 == 0:
                    self.scan_range_2[0],self.scan_range_2[1]=peak_volt-0.2,peak_volt+0.2
                if self.scan_range_adjustment_2 == 1:
                    self.scan_range_2[0],self.scan_range_2[1]=peak_volt-0.5,peak_volt+0.4
                if self.scan_range_adjustment_2 == 2:
                    pass
                    #self.scan_range[0],self.scan_range[1]=peak_volt-0.5*range,peak_volt+0.5*range
                self.sigScanRangeChanged.emit(self.scan_range_2[0],self.scan_range_2[1],2)
                self.laser_at_position = True
            else: 
                print("No PLE found in range. Retry with bigger scan range...")
                self.scan_range_2[0],self.scan_range_2[1]=self.scan_range_2[0]-1,self.scan_range_2[1]+1
                self.scan_range_2=list(np.clip(self.scan_range_2,-3,3))
                self.sigScanRangeChanged.emit(self.scan_range_2[0],self.scan_range_2[1],2)
                self.start_scanning_2()
        self.stopped=True
        print('goto_fitted_peak done')
        try:
            self.curr_freq = 299792458/ self._wavemeterlogic._wavemeter_device.get_current_wavelength(kind ='vac', ch = self._wavemeterlogic._wavemeter_device.channel1) # GHz
            self.current_freq = f"{self.curr_freq:.4f}"
            print(f"Laser now at {self.current_freq:.4f} GHz")
        except:
            pass
        return 0


    def save_data(self, tag=None, colorscale_range=None, percentile_range=None):
        """ Save the counter trace data and writes it to a file.

        @return int: error code (0:OK, -1:error)
        """
        if tag is None:
            tag = ''

        self._saving_stop_time = time.time()

        filepath  = self._save_logic.get_path_for_module(module_name='LaserScanning')
        timestamp = None # datetime.datetime.now()

        if len(tag) > 0:
            filelabel  = tag + '_PLE_data_1'
            filelabel2 = tag + '_PLE_data_raw_trace_1'
            filelabel22 = tag + '_PLE_data_plot_trace_1'
            # filelabel3 = tag + '_PLE_data_raw freqs vs cts'
            filelabel4 = tag + '_PLE_data_raw_freqs_1'
            filelabel5 = tag + '_PLE_bias_voltages_1'
        else:
            filelabel  = 'PLE_data_1'
            filelabel2 = 'PLE_data_raw_trace_1'
            filelabel22 = 'PLE_data_plot_trace_1'
            # filelabel3 = 'PLE_data_raw freqs vs cts'
            filelabel4 = 'PLE_data_raw_freqs_1'
            filelabel5 = tag + 'PLE_bias_voltages_1'
        
        # prepare the data in a dict or in an OrderedDict:
        data = OrderedDict()
        data['frequency (Hz)'] = self.plot_x_frequency
        data['trace count data (counts/s)'] = self.plot_y
        data['scanvoltage (V)'] = self.plot_x

        data2 = OrderedDict()
        data2['count data (counts/s)'] = self.scan_matrix[:self._scan_counter_up, :]
        data22 = OrderedDict()
        data22['count data (counts/s)'] = self.plot_matrix[:self._scan_counter_up, :]

        data3 =OrderedDict()
        data3["freqs vs cts per scan"]= self.xy_data

        data4 = OrderedDict()
        data4["freqs"] = self.measured_frequencies

        data5 = OrderedDict()
        data5["bias voltages"] = self.bias_voltages
        data5["bias voltages 2"] = self.bias_voltages_2

        parameters = OrderedDict()
        parameters['Number of frequency sweeps (#)'] = self._scan_counter_up
        parameters['Start Voltage (V)'] = self.scan_range[0]
        parameters['Stop Voltage (V)'] = self.scan_range[1]
        parameters['Scan speed [V/s]'] = self._scan_speed
        parameters['Clock Frequency (Hz)'] = self._clock_frequency
        parameters['ContrastFit'] = self.Contrast_Fit
        parameters['FrequenciesFit'] = self.Frequencies_Fit
        parameters['LinewidthsFit'] = self.Linewidths_Fit
        parameters['MW1 Power (dBm)'] = self.MW1_Power
        parameters['MW2 Power (dBm)'] = self.MW2_Power
        parameters['MW3 Power (dBm)'] = self.MW3_Power
        parameters['MW1 freq (MHz)'] = self.MW1_Freq
        parameters['MW2 freq (MHz)'] = self.MW2_Freq
        parameters['MW3 freq (MHz)'] = self.MW3_Freq
        parameters['PLE voltage-factor (V to GHz)'] = self.factor_V_to_GHz
        parameters['Current laser'] = self.laser_choice
        parameters['Current input channel'] = self.input_channel
        parameters['Current tuning mode'] = self.tuning_mode
        parameters['Current ARC factor'] = self.arc_factor
        parameters['Current CTL wavelength'] = self.ctl_wavelength
        with open(os.getcwd()+"\\current_temperature.txt","r") as f:
            self.temperature = float(np.loadtxt(f,skiprows=3))
        parameters['Sample temperature (K)'] = self.temperature
        parameters['Defect position x/y (m/m)'] = self.xy_coords


        fig = self.draw_figure(
            self.plot_matrix,
            self.plot_x,
            self.plot_y,
            self.interpolated_x_data,
            self.fit_data,
            cbar_range=colorscale_range,
            percentile_range=percentile_range)

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
            filelabel=filelabel2,
            fmt='%.6e',
            delimiter='\t',
            timestamp=timestamp,
            plotfig=fig
        )

        self._save_logic.save_data(
            data22,
            filepath=filepath,
            parameters=parameters,
            filelabel=filelabel22,
            fmt='%.6e',
            delimiter='\t',
            timestamp=timestamp
        )

        # self._save_logic.save_data(
        #     data3,
        #     filepath=filepath3,
        #     parameters=parameters,
        #     filelabel=filelabel3,
        #     fmt='%.6e',
        #     delimiter='\t',
        #     timestamp=timestamp,
        #     plotfig=fig
        # )

        
        self._save_logic.save_data(
            data4,
            filepath=filepath,
            parameters=parameters,
            filelabel=filelabel4,
            fmt='%.15e',
            delimiter='\t',
            timestamp=timestamp
        )
        self._save_logic.save_data(
            data5,
            filepath=filepath,
            parameters=parameters,
            filelabel=filelabel5,
            fmt='%.15e',
            delimiter='\t',
            timestamp=timestamp
        )
        self.log.info('Laser Scan saved to:\n{0}'.format(filepath))
        return 0
    
    def save_data_2(self, tag=None, colorscale_range=None, percentile_range=None):
        """ Save the counter trace data and writes it to a file.

        @return int: error code (0:OK, -1:error)
        """
        if tag is None:
            tag = ''

        self._saving_stop_time = time.time()

        filepath  = self._save_logic.get_path_for_module(module_name='LaserScanning')
        timestamp = None # datetime.datetime.now()

        if len(tag) > 0:
            filelabel  = tag + '_PLE_data_2'
            filelabel2 = tag + '_PLE_data_raw_trace_2'
            filelabel22 = tag + '_PLE_data_plot_trace_2'
            # filelabel3 = tag + '_PLE_data_raw freqs vs cts'
            filelabel4 = tag + '_PLE_data_raw_freqs_2'
            filelabel5 = tag + '_PLE_bias_voltages_2'
        else:
            filelabel  = 'PLE_data_2'
            filelabel2 = 'PLE_data_raw_trace_2'
            filelabel22 = 'PLE_data_plot_trace_2'
            # filelabel3 = 'PLE_data_raw freqs vs cts'
            filelabel4 = 'PLE_data_raw_freqs_2'
            filelabel5 = tag + 'PLE_bias_voltages_2'
        
        # prepare the data in a dict or in an OrderedDict:
        data = OrderedDict()
        data['frequency (Hz)'] = self.plot_x_frequency_2
        data['trace count data (counts/s)'] = self.plot_y_2
        data['scanvoltage (V)'] = self.plot_x_2

        data2 = OrderedDict()
        data2['count data (counts/s)'] = self.scan_matrix_2[:self._scan_counter_up_2, :]
        data22 = OrderedDict()
        data22['count data (counts/s)'] = self.plot_matrix_2[:self._scan_counter_up_2, :]

        data3 =OrderedDict()
        data3["freqs vs cts per scan"]= self.xy_data_2

        data4 = OrderedDict()
        data4["freqs"] = self.measured_frequencies_2

        data5 = OrderedDict()
        data5["bias voltages"] = self.bias_voltages
        data5["bias voltages 2"] = self.bias_voltages_2

        parameters = OrderedDict()
        parameters['Number of frequency sweeps (#)'] = self._scan_counter_up_2
        parameters['Start Voltage (V)'] = self.scan_range_2[0]
        parameters['Stop Voltage (V)'] = self.scan_range_2[1]
        parameters['Scan speed [V/s]'] = self._scan_speed_2
        parameters['Clock Frequency (Hz)'] = self._clock_frequency_2
        parameters['ContrastFit'] = self.Contrast_Fit_2
        parameters['FrequenciesFit'] = self.Frequencies_Fit_2
        parameters['LinewidthsFit'] = self.Linewidths_Fit_2
        parameters['MW1 Power (dBm)'] = self.MW1_Power_2
        parameters['MW2 Power (dBm)'] = self.MW2_Power_2
        parameters['MW3 Power (dBm)'] = self.MW3_Power_2
        parameters['MW1 freq (MHz)'] = self.MW1_Freq_2
        parameters['MW2 freq (MHz)'] = self.MW2_Freq_2
        parameters['MW3 freq (MHz)'] = self.MW3_Freq_2
        parameters['PLE voltage-factor (V to GHz)'] = self.factor_V_to_GHz_2
        parameters['Current laser'] = self.laser_choice_2
        parameters['Current input channel'] = self.input_channel_2
        parameters['Current tuning mode'] = self.tuning_mode_2
        parameters['Current ARC factor'] = self.arc_factor_2
        parameters['Current CTL wavelength'] = self.ctl_wavelength_2
        with open(os.getcwd()+"\\current_temperature.txt","r") as f:
            self.temperature = float(np.loadtxt(f,skiprows=3))
        parameters['Sample temperature (K)'] = self.temperature
        parameters['Defect position x/y (m/m)'] = self.xy_coords


        fig = self.draw_figure(
            self.plot_matrix_2,
            self.plot_x_2,
            self.plot_y_2,
            self.interpolated_x_data_2,
            self.fit_data_2,
            cbar_range=colorscale_range,
            percentile_range=percentile_range)

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
            filelabel=filelabel2,
            fmt='%.6e',
            delimiter='\t',
            timestamp=timestamp,
            plotfig=fig
        )

        self._save_logic.save_data(
            data22,
            filepath=filepath,
            parameters=parameters,
            filelabel=filelabel22,
            fmt='%.6e',
            delimiter='\t',
            timestamp=timestamp
        )

        # self._save_logic.save_data(
        #     data3,
        #     filepath=filepath3,
        #     parameters=parameters,
        #     filelabel=filelabel3,
        #     fmt='%.6e',
        #     delimiter='\t',
        #     timestamp=timestamp,
        #     plotfig=fig
        # )

        
        self._save_logic.save_data(
            data4,
            filepath=filepath,
            parameters=parameters,
            filelabel=filelabel4,
            fmt='%.15e',
            delimiter='\t',
            timestamp=timestamp
        )
        self._save_logic.save_data(
            data5,
            filepath=filepath,
            parameters=parameters,
            filelabel=filelabel5,
            fmt='%.15e',
            delimiter='\t',
            timestamp=timestamp
        )
        self.log.info('Laser Scan saved to:\n{0}'.format(filepath))
        return 0
    

    def draw_figure(self, matrix_data, freq_data, count_data, fit_freq_vals, fit_count_vals, cbar_range=None, percentile_range=None):
        """ Draw the summary figure to save with the data.

        @param: list cbar_range: (optional) [color_scale_min, color_scale_max].
                                 If not supplied then a default of data_min to data_max
                                 will be used.

        @param: list percentile_range: (optional) Percentile range of the chosen cbar_range.

        @return: fig fig: a matplotlib figure object to be saved to file.
        """

        # If no colorbar range was given, take full range of data
        if cbar_range is None:
            cbar_range = np.array([np.min(matrix_data), np.max(matrix_data)])
        else:
            cbar_range = np.array(cbar_range)

        prefix = ['', 'k', 'M', 'G', 'T']
        prefix_index = 0

        # Rescale counts data with SI prefix
        while np.max(count_data) > 1000:
            count_data = count_data / 1000
            fit_count_vals = fit_count_vals / 1000
            prefix_index = prefix_index + 1

        counts_prefix = prefix[prefix_index]

        # Rescale frequency data with SI prefix
        prefix_index = 0

        while np.max(freq_data) > 1000:
            freq_data = freq_data / 1000
            fit_freq_vals = fit_freq_vals / 1000
            prefix_index = prefix_index + 1

        mw_prefix = prefix[prefix_index]

        # Rescale matrix counts data with SI prefix
        prefix_index = 0

        while np.max(matrix_data) > 1000:
            matrix_data = matrix_data / 1000
            cbar_range = cbar_range / 1000
            prefix_index = prefix_index + 1

        cbar_prefix = prefix[prefix_index]

        # Use qudi style
        plt.style.use(self._save_logic.mpl_qd_style)

        # Create figure
        fig, (ax_mean, ax_matrix) = plt.subplots(nrows=2, ncols=1)

        ax_mean.plot(freq_data*self.factor_V_to_GHz, count_data, linestyle=':', linewidth=0.5)

        # Do not include fit curve if there is no fit calculated.
        if max(fit_count_vals) > 0:
            ax_mean.plot(fit_freq_vals*self.factor_V_to_GHz, fit_count_vals, marker='None')

        ax_mean.set_ylabel('Fluorescence (' + counts_prefix + 'c/s)')
        ax_mean.set_xlabel('Detuning (' + mw_prefix + 'GHz)')
        ax_mean.set_xlim(np.min(freq_data*self.factor_V_to_GHz), np.max(freq_data*self.factor_V_to_GHz))

        matrixplot = ax_matrix.imshow(
            matrix_data,
            cmap=plt.get_cmap('inferno'),  # reference the right place in qd
            origin='lower',
            vmin=cbar_range[0],
            vmax=cbar_range[1],
            extent=[
                np.min(freq_data),
                np.max(freq_data),
                0,
                self.number_of_lines
                ],
            aspect='auto',
            interpolation='nearest')

        ax_matrix.set_xlabel('Scan voltage (V)')
        ax_matrix.set_ylabel('Scan #')

        # Adjust subplots to make room for colorbar & upper xlabel
        fig.subplots_adjust(right=0.8,hspace=0.3)

        # Add colorbar axis to figure
        cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])

        # Draw colorbar
        cbar = fig.colorbar(matrixplot, cax=cbar_ax)
        cbar.set_label('Fluorescence (' + cbar_prefix + 'c/s)')

        # remove ticks from colorbar for cleaner image
        cbar.ax.tick_params(which='both', length=0)

        # If we have percentile information, draw that to the figure
        if percentile_range is not None:
            cbar.ax.annotate(str(percentile_range[0]),
                             xy=(-0.3, 0.0),
                             xycoords='axes fraction',
                             horizontalalignment='right',
                             verticalalignment='center',
                             rotation=90
                             )
            cbar.ax.annotate(str(percentile_range[1]),
                             xy=(-0.3, 1.0),
                             xycoords='axes fraction',
                             horizontalalignment='right',
                             verticalalignment='center',
                             rotation=90
                             )
            cbar.ax.annotate('(percentile)',
                             xy=(-0.3, 0.5),
                             xycoords='axes fraction',
                             horizontalalignment='right',
                             verticalalignment='center',
                             rotation=90
                             )

        return fig
    
    def do_gaussian_fit(self):
        x_data=self.plot_x.astype(np.float)
        #x_data=self.plot_x_frequency.astype(np.float)
        y_data=self.plot_y.astype(np.float)
        if self.NumberOfPeaks==1:
            model,params=self._fit_logic.make_gaussian_model()

            result = self._fit_logic.make_gaussian_fit(
                                x_axis=x_data,
                                data=y_data,
                                units='Hz',
                                estimator=self._fit_logic.estimate_gaussian_peak
                                )

        elif self.NumberOfPeaks==2:
            model,params=self._fit_logic.make_gaussiandouble_model()

            result = self._fit_logic.make_gaussiandouble_fit(
                                x_axis=x_data,
                                data=y_data,
                                units='Hz',
                                estimator=self._fit_logic.estimate_gaussiandouble_peak
                                )
        elif self.NumberOfPeaks==3:
            model,params=self._fit_logic.make_gaussiantriple_model()

            result = self._fit_logic.make_gaussiantriple_fit(
                                x_axis=x_data,
                                data=y_data,
                                units='Hz',
                                estimator=self._fit_logic.estimate_gaussiantriple_peak
                                )


            logger.warning("function 3 gaussian peaks not implemeted")

        self.interpolated_x_data=np.linspace(x_data.min(),x_data.max(),len(x_data)*5)
        self.fit_data = model.eval(x=self.interpolated_x_data, params=result.params)
        
        #using own fitlogic
        # fit_func=self._fit_logic.make_n_gauss_function(self.NumberOfPeaks)
        # result=fit_func.fit(x_data,y_data)
        # self.fit_data=fit_func(interpolated_x_data,*result["result"].x)

        self.Contrast_Fit:str=''
        self.Frequencies_Fit:str=''
        self.Linewidths_Fit:str=''
        self.Frequencies_Fit_GHz:str=''
        self.Linewidths_Fit_GHz:str=''


        for i in range(self.NumberOfPeaks):
            try:
                self.Contrast_Fit=self.Contrast_Fit+str(round(result.params[("g"+str(i)+"_")*(self.NumberOfPeaks!=1)+"amplitude"].value,3))+"; " # because 1 peak and 2 peak gaussian fit dont give the same result keywords, we add the 'gi_' part (missing in the 1 peak case) by multiplying the string by 1 if paeks!=1 and remove it if peaks=1.
                Frequencies_Fit=str(round(result.params[("g"+str(i)+"_")*(self.NumberOfPeaks!=1)+"center"].value,5))+"V; "
                self.Frequencies_Fit=self.Frequencies_Fit+Frequencies_Fit
                Linewidths_Fit=str(round(result.params[("g"+str(i)+"_")*(self.NumberOfPeaks!=1)+"fwhm"].value,3))+"V; " #TODO convert linewidth from V to MHz
                self.Linewidths_Fit=self.Linewidths_Fit+Linewidths_Fit #TODO convert linewidth from V to MHz
                self.Frequencies_Fit_GHz=self.Frequencies_Fit_GHz+Frequencies_Fit+str(round(result.params[("g"+str(i)+"_")*(self.NumberOfPeaks!=1)+"center"].value*self.factor_V_to_GHz,3))+"GHz; "
                self.Linewidths_Fit_GHz=self.Linewidths_Fit_GHz+Linewidths_Fit+str(round(result.params[("g"+str(i)+"_")*(self.NumberOfPeaks!=1)+"fwhm"].value*self.factor_V_to_GHz,3))+"GHz; " #TODO convert linewidth from V to MHz
            except:
                self.log.warning('Error occured at fitting.')

        self.sigFitPerformed.emit(self.Frequencies_Fit)
        return self.interpolated_x_data,self.fit_data,result
    
    def do_gaussian_fit_2(self):
        x_data=self.plot_x_2.astype(np.float)
        #x_data=self.plot_x_frequency.astype(np.float)
        y_data=self.plot_y_2.astype(np.float)
        if self.NumberOfPeaks==1:
            model,params=self._fit_logic.make_gaussian_model()

            result = self._fit_logic.make_gaussian_fit(
                                x_axis=x_data,
                                data=y_data,
                                units='Hz',
                                estimator=self._fit_logic.estimate_gaussian_peak
                                )

        elif self.NumberOfPeaks==2:
            model,params=self._fit_logic.make_gaussiandouble_model()

            result = self._fit_logic.make_gaussiandouble_fit(
                                x_axis=x_data,
                                data=y_data,
                                units='Hz',
                                estimator=self._fit_logic.estimate_gaussiandouble_peak
                                )
        elif self.NumberOfPeaks==3:
            model,params=self._fit_logic.make_gaussiantriple_model()

            result = self._fit_logic.make_gaussiantriple_fit(
                                x_axis=x_data,
                                data=y_data,
                                units='Hz',
                                estimator=self._fit_logic.estimate_gaussiantriple_peak
                                )


            logger.warning("function 3 gaussian peaks not implemeted")

        self.interpolated_x_data_2=np.linspace(x_data.min(),x_data.max(),len(x_data)*5)
        self.fit_data_2 = model.eval(x=self.interpolated_x_data_2, params=result.params)
        
        #using own fitlogic
        # fit_func=self._fit_logic.make_n_gauss_function(self.NumberOfPeaks)
        # result=fit_func.fit(x_data,y_data)
        # self.fit_data=fit_func(interpolated_x_data,*result["result"].x)

        self.Contrast_Fit_2:str=''
        self.Frequencies_Fit_2:str=''
        self.Linewidths_Fit_2:str=''
        self.Frequencies_Fit_GHz_2:str=''
        self.Linewidths_Fit_GHz_2:str=''


        for i in range(self.NumberOfPeaks_2):
            try:
                self.Contrast_Fit_2=self.Contrast_Fit_2+str(round(result.params[("g"+str(i)+"_")*(self.NumberOfPeaks_2!=1)+"amplitude"].value,3))+"; " # because 1 peak and 2 peak gaussian fit dont give the same result keywords, we add the 'gi_' part (missing in the 1 peak case) by multiplying the string by 1 if paeks!=1 and remove it if peaks=1.
                Frequencies_Fit=str(round(result.params[("g"+str(i)+"_")*(self.NumberOfPeaks_2!=1)+"center"].value,5))+"V; "
                self.Frequencies_Fit_2=self.Frequencies_Fit_2+Frequencies_Fit
                Linewidths_Fit=str(round(result.params[("g"+str(i)+"_")*(self.NumberOfPeaks_2!=1)+"fwhm"].value,3))+"V; " #TODO convert linewidth from V to MHz
                self.Linewidths_Fit_2=self.Linewidths_Fit_2+Linewidths_Fit #TODO convert linewidth from V to MHz
                self.Frequencies_Fit_GHz_2=self.Frequencies_Fit_GHz_2+Frequencies_Fit+str(round(result.params[("g"+str(i)+"_")*(self.NumberOfPeaks_2!=1)+"center"].value*self.factor_V_to_GHz_2,3))+"GHz; "
                self.Linewidths_Fit_GHz_2=self.Linewidths_Fit_GHz_2+Linewidths_Fit+str(round(result.params[("g"+str(i)+"_")*(self.NumberOfPeaks_2!=1)+"fwhm"].value*self.factor_V_to_GHz_2,3))+"GHz; " #TODO convert linewidth from V to MHz
            except:
                self.log.warning('Error occured at fitting.')

        self.sigFitPerformed_2.emit(self.Frequencies_Fit_2)
        return self.interpolated_x_data_2,self.fit_data_2,result
    
    def convert_seq_params_to_string(self, laser_index=1):
        if laser_index == 1:
            return str(self.MW1_Power)+str(self.MW2_Power)+str(self.MW3_Power)+str(self.MW1_Freq)+str(self.MW2_Freq)+str(self.MW3_Freq)+str(self.enable_MW1)+str(self.enable_MW2)+str(self.enable_MW3)+str(self.enable_A1)+str(self.enable_A2)+str(self.enable_Repump)+str(self.enable_PulsedRepump)+str(self.enable_CTL)+str(self.enable_PulsedCTL)+str(self.enable_Blue)+str(self.enable_PulsedBlue)+str(self.lock_laser)+str(self.RepumpDuration)+str(self.RepumpDecay)+str(self.CTLDuration)+str(self.CTLDecay)+str(self.BlueDuration)+str(self.BlueDecay)
        elif laser_index == 2:
            return str(self.MW1_Power_2)+str(self.MW2_Power_2)+str(self.MW3_Power_2)+str(self.MW1_Freq_2)+str(self.MW2_Freq_2)+str(self.MW3_Freq_2)+str(self.enable_MW1_2)+str(self.enable_MW2_2)+str(self.enable_MW3_2)+str(self.enable_A1_2)+str(self.enable_A2_2)+str(self.enable_Repump_2)+str(self.enable_PulsedRepump_2)+str(self.enable_CTL_2)+str(self.enable_PulsedCTL_2)+str(self.enable_Blue_2)+str(self.enable_PulsedBlue_2)+str(self.lock_laser_2)+str(self.RepumpDuration_2)+str(self.RepumpDecay_2)+str(self.CTLDuration_2)+str(self.CTLDecay_2)+str(self.BlueDuration_2)+str(self.BlueDecay_2)
    

    def toggle_emission(self):
        if self.laser_choice == 0:
            if (self._ctl_logic.emission() == False) and (self._ctl_logic.CCemission() == True):
                print("Emission button is not activated! Enable manually!")
                pass
            elif (self._ctl_logic.emission() == False) and (self._ctl_logic.CCemission() == False):
                self._ctl_logic.CCemission(True)
                print("Emission of CTL turned ON!")
            elif (self._ctl_logic.emission() == True) and (self._ctl_logic.CCemission() == True):
                self._ctl_logic.CCemission(False)
                print("Emission of CTL turned OFF!")

        elif self.laser_choice == 1:
            if (self._dlcpro_logic.emission() == False) and (self._dlcpro_logic.CCemission() == True):
                print("Emission button is not activated! Enable manually!")
                pass
            elif (self._dlcpro_logic.emission() == False) and (self._dlcpro_logic.CCemission() == False):
                self._dlcpro_logic.CCemission(True)
                print("Emission of DLC Pro turned ON!")
            elif (self._dlcpro_logic.emission() == True) and (self._dlcpro_logic.CCemission() == True):
                self._dlcpro_logic.CCemission(False)
                print("Emission of DLC Pro turned OFF!")

        QtTest.QTest.qSleep(500)  
        self.check_emission()         


    ### From here:
    # Take frequency data from wavemeter and extrapolate it
    # Use this extrapolated data as new x-axis for plot and fit

    # def find_freq_extrema(self, data, testplot=False):
    #     ### Getting minima and maxima with peakfinder (for minima the plot has to be mirrored)
    #     maxima = find_peaks(data - np.mean(data))[0]
    #     minima = find_peaks(np.mean(data) - data)[0]

    #     ## Removing the last element of the minima since it is already related to the way back to 0 of the scan:
    #     #minima = minima[: -1]
    #     #minima = minima[1:]

    #     ## Removing first element of maxima since first line is about to be ignored
    #     maxima = maxima[1:]

    #     if testplot:
    #         plt.plot(data)
    #         plt.plot(maxima, data[maxima], 'o')
    #         plt.plot(minima, data[minima], 'o')
    #         plt.show()

    #     return (maxima, minima)
    
    # def interpolate_freqlines(self, data, minima, maxima, samples):
    #     frequencylist = []
    #     linelist = []
    #     for i, max in enumerate(maxima):
    #         y = data[minima[i]:maxima[i]]
    #         x = np.arange(0, len(y), 1)
    #         f = UnivariateSpline(x, y)
    #         f.set_smoothing_factor(0.35)
    #         freqline = f(np.linspace(0, maxima[i] - minima[i] - 1, samples))
    #         frequencylist.append(freqline)

    #         linelist.append(np.full((samples), i + 1))

    #     frequencylist = np.array(frequencylist)
    #     ## Estimating the mean frequency:
    #     meanfreq = np.mean(frequencylist)
    #     relativelist = (frequencylist - meanfreq) * 1e3

    #     #return (relativelist, linelist, meanfreq)
    #     return (frequencylist*1e3, linelist, meanfreq)
    
    # def average_freqlines(self, freqlines):
    #     frequencymean = np.zeros_like(freqlines)
    #     for i in freqlines:
    #         frequencymean += i
    #     frequencymean = frequencymean / np.shape(freqlines)[0]
    #     return (frequencymean[0])
        
    # def extract_freqlines(self, extrematest=False):
    #     maxima, minima = self.find_freq_extrema(self.measured_frequencies, extrematest)
    #     freqlines, linelist, meanfreq = self.interpolate_freqlines(self.measured_frequencies, minima, maxima, int(self.resolution))
    #     if np.shape(freqlines)[0] > 0:
    #         freqmeanline = self.average_freqlines(freqlines)
    #         self.plot_x_frequency = freqmeanline

    @QtCore.Slot()
    def update_ProgressBar(self, laser_index=1):
        scan_counter,number_of_repeats = self._scan_counter_up,self.number_of_repeats
        # print("Update ProgressBar (logic): ", scan_counter, number_of_repeats)
        # print("Update ProgressBar (logic) 2: ", 1.*scan_counter/number_of_repeats*100, 100)
        self.ple_progressBar.emit(1.*scan_counter/number_of_repeats*100, 100, laser_index)

    def check_emission(self):
        self.ctl_emission = self._ctl_logic.emission()
        self.dlcpro_emission = self._dlcpro_logic.emission()
        self.emissionChanged.emit(self.ctl_emission,self.dlcpro_emission)

    ########### for live PLE analysis with the wavemeter:   
    def find_freq_extrema(self, data, num):
        """finds and returns all desired extrema of each scan

        Args:
            data (array): wavemeter scan data
            num (int): length of applied voltage range

        Raises:
            ValueError: _description_
            ZeroDivisionError: _description_

        Returns:
            (list,list): array of minimas and maximas
        """
        try:
            ### Getting minima and maxima with peakfinder (for minima the plot has to be mirrored)
            maxima = find_peaks(data - np.mean(data),distance=len(data)/num*0.7)[0]
            minima = find_peaks(np.mean(data) - data,distance=len(data)/num*0.7)[0]
            maxima0 = maxima.copy()
            minima0 = minima.copy()

            ### The maxima and minima should not differ a lot from each other, so we say thay should lay within 3 std error deviation
            maxima = maxima[np.mean(data[maxima])-data[maxima] < np.std(data[maxima])*5]
            minima = minima[np.mean(data[minima])-data[minima] > -np.std(data[minima])*5]

            ### if we have to many minina, probably because it counted at the end another one
            if len(minima) > num:
                if minima[-1] >= maxima[-1]:
                    minima = minima[:-1]
            if len(maxima) > num:
                ## Removing first element of maxima since first line is about to be ignored
                if len(maxima)==len(minima)+1:
                    maxima = maxima[1:]
            ### if the is just one maxima missing, then its most likely the last one, as there is no clear peak
            if len(maxima)==len(minima)-1:
                maxima = np.append(maxima,len(data)-1)
            ### it could be that it found the last maxima at the beginning, just fix it here
            if len(maxima)==len(minima):
                if minima[0] >= maxima[0]:
                    maxima = maxima[1:]
                    maxima = np.append(maxima,len(data)-1)

            ### Error if there is an unequal amount of extrema
            if len(maxima) != len(minima):
                raise ValueError
            ### Error if the amount of extrema do not match the amount of sweeps
            elif (num < 40) & (len(maxima) != num):
                raise ZeroDivisionError
            ### Error if the amount of extrema do not match the amount of sweeps, but you might not care for too many scans, then ignore
            elif (num > 40) & (len(maxima) != num):
                print("not all peaks found, but ignored because of to many scans")
            return (maxima, minima)
        except ValueError:
            print(f"min's ({len(minima)}) and max's ({len(maxima)}) have not equal lengths!")
            return ([],[])
        except ZeroDivisionError:
            print("min's and max's are equal but have not the right length!")
            print("Desired: ",num," Found: ",len(maxima))
            return ([],[])
        except IndexError:
            print("IndexError: index 0 is out of bounds for axis 0 with size 0")
            print("Probably empty arrays for min and max, so no extrema found!")
    
    def interpolate_freqlines(self, data, minima, maxima, samples, relative=False):
        """Uses extremas to interpolate each scan individually and rescales the frequency axis

        Args:
            data (array): wavemeter scan data
            minima (list): list of minimas
            maxima (list): list of maximas
            samples (int): length of applied voltage list
            relative (bool, optional): _description_. Defaults to False.

        Returns:
            array: interpolated frequency arrays
            float: central frequency
        """
        frequencylist = []
        linelist = []
        for i, max in enumerate(maxima):
            # try:
            y = data[minima[i]:maxima[i]]
            x = np.arange(0, len(y), 1)
            # f = CubicSpline(x, y, bc_type="clamped")
            f = UnivariateSpline(x, y,k=5)
            f.set_smoothing_factor(0.001)                # which smoothing factor? 0.35
            freqline = f(np.linspace(0, maxima[i] - minima[i] - 1, samples))
            frequencylist.append(freqline)

            linelist.append(np.full((samples), i + 1))  # not sure for what good for
            # except:
            #     pass

        frequencylist = np.array(frequencylist) * 1e3   # *1e3 for GHz instead of THz
        ## Estimating the mean frequency:
        meanfreq = np.mean(frequencylist)
        relativelist = (frequencylist - meanfreq) 

        if relative:
            return (relativelist, linelist, meanfreq)
        else:
            return (frequencylist, linelist, meanfreq)
    
    def average_freqlines(self, freqlines):
        """returns averaged interpolated frequency array

        Args:
            freqlines (_type_): _description_

        Returns:
            _type_: _description_
        """
        frequencymean = np.zeros_like(freqlines)
        for i in freqlines:
            frequencymean += i
        frequencymean = frequencymean / np.shape(freqlines)[0]
        return (frequencymean[0])
    
    def extract_freqlines(self, wavemeter_data, ple_voltage, num, relative=False):
        """finds extrema, interpolates, averages and fits the new frequency array

        Args:
            wavemeter_data (array): y data of ple logging
            ple_voltage (array): voltage array
            num (int): number of scans done
            relative (bool, optional): decides if returned frequency are relative or absolute. Defaults to False.

        Returns:
            freqmeanline (array): _description_
            freqlines (nd-array): _description_
            meanfreq (float): _description_
        """
        def X1(x, a, b):
            return a * x + b
        res = len(ple_voltage)

        maxima, minima = self.find_freq_extrema(wavemeter_data, num)

        freqlines, linelist, meanfreq = self.interpolate_freqlines(wavemeter_data, minima, maxima, res, relative=relative)

        if np.shape(freqlines)[0] > 0:
            freqmeanline = self.average_freqlines(freqlines)
            # global ResX4
            # global ResX1
            # ResX4,cov = curve_fit(X4,ple_voltage,freqmeanline)
            ResX1,cov = curve_fit(X1,ple_voltage,freqmeanline)
            return ResX1
        
    def _estimate_ratio(self):
        print("DuoPLE1: WM Calibration fired!")
        self.success = "Failed!"
        # if not WAVEMETER:
        #     self.success = "no WM found!"
        #     pass
        # else:        
        # in case of motor scan we have to switch the voltages:
        if (self.laser_choice == 1) and (self.tuning_mode == 1):
            res = self.extract_freqlines(self.measured_frequencies[:,1][::-1],self.plot_x,self._scan_counter_up)
        else:
            res = self.extract_freqlines(self.measured_frequencies[:,1],self.plot_x,self._scan_counter_up)
        self.factor_V_to_GHz = float(f"{res[0]:.3f}")
        # self.F0 = res[1]
        print(f"DuoPLE1: Calibration done! New factor_V_to_GHz value is {res[0]}")
        self.success = "Success!"
        if self.success == "Success!":
            self.sigPLEfactorChanged.emit(self.factor_V_to_GHz,1)
            self.plot_x_frequency=self.plot_x*self.factor_V_to_GHz

    def _estimate_ratio_2(self):
        print("DuoPLE2: WM Calibration fired!")
        self.success = "Failed!"
        # if not WAVEMETER:
        #     self.success = "no WM found!"
        #     pass
        # else:        
        # in case of motor scan we have to switch the voltages:
        if (self.laser_choice_2 == 1) and (self.tuning_mode_2 == 1):
            res = self.extract_freqlines(self.measured_frequencies_2[:,1][::-1],self.plot_x_2,self._scan_counter_up_2)
        else:
            res = self.extract_freqlines(self.measured_frequencies_2[:,1],self.plot_x_2,self._scan_counter_up_2)
        self.factor_V_to_GHz_2 = float(f"{res[0]:.3f}")
        # self.F0 = res[1]
        print(f"DuoPLE2: Calibration done! New factor_V_to_GHz value is {res[0]}")
        self.success = "Success!"
        if self.success == "Success!":
            self.sigPLEfactorChanged.emit(self.factor_V_to_GHz_2,2)
            self.plot_x_frequency_2=self.plot_x_2*self.factor_V_to_GHz_2

    def factor_V_to_GHz_pushButton_Clicked(self,on):
        print('DuoPLE1: done something with factor_V_to_GHz_pushButton')
        self._estimate_ratio()
        QtTest.QTest.qSleep(200)
        self.sigUpdatePlots.emit(1)

    def factor_V_to_GHz_pushButton_2_Clicked(self,on):
        print('DuoPLE2: done something with factor_V_to_GHz_pushButton_2')
        self._estimate_ratio_2()
        QtTest.QTest.qSleep(200)
        self.sigUpdatePlots.emit(2)

    
    def discard_counts(self,value):
        self.discard=value==2
        self.sigUpdatePlots.emit(1)
    
    def discard_counts_2(self,value):
        self.discard_2=value==2
        self.sigUpdatePlots.emit(2)