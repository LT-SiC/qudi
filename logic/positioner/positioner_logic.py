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

from core.connector import Connector
from core.statusvariable import StatusVar
from core.util.mutex import Mutex
from logic.generic_logic import GenericLogic
from qtpy import QtCore
from PyQt5 import QtTest
import logging; logger = logging.getLogger(__name__)

import base64
import hashlib

# for analysis:
# from scipy.signal import find_peaks
# from scipy.interpolate import UnivariateSpline
# from scipy.optimize import curve_fit

from core.configoption import ConfigOption

from logic.positioner.positioner_default_values_and_widget_functions import positioner_default_values_and_widget_functions as default
# from gui.attocube.positioners.connectors_and_set_default import initialize_connections_and_defaultvalue

from hardware.Attocube.anc3XX import ANC300, ANC350, AXIS




class PositionControlLogic(GenericLogic, default):

    """This logic module controls the postion of the sample and the fiber by controlling the ANC300 and ANC350 attocube controler
    """

    # declare connectors
    anc300 = Connector(interface='ANC300') # Sample
    anc350 = Connector(interface='ANC350') # Fiber
    
    T_changed_signal = QtCore.Signal(str)
    update_signal = QtCore.Signal(list)
    C_changed_signal = QtCore.Signal(int,float)
    # sig_data_updated = QtCore.Signal()
    # Filename = StatusVar('Filename', "Filename")
    # _scan_range_adjustment = StatusVar('scan_range_adjustment', 1)
    # scan_range = StatusVar('scan_range', [-3, 3])
    # resolution = StatusVar(default=100)
    # stepsize = StatusVar(default=0.01)
    # number_of_lines = StatusVar(default=10)
    # number_of_repeats = StatusVar(default=10)
    # NumberOfPeaks = StatusVar('NumberOfPeaks', default=2)
    # _scan_speed = StatusVar('scan_speed', 0.1)
    # _static_v = StatusVar('goto_voltage', 0)
    # factor_V_to_GHz = StatusVar('factor_V_to_GHz', 1)
    # current_freq = StatusVar('current_freq', "")
    # fc = StatusVar('fits', None)
    # enable_A1 = StatusVar('enable_A1', False)
    # enable_A2 = StatusVar('enable_A2', True)
    # enable_Repump = StatusVar('enable_Repump', False)
    # enable_PulsedRepump = StatusVar('enable_PulsedRepump', True)
    # enable_CTL = StatusVar('enable_CTL', False)
    # enable_PulsedCTL = StatusVar('enable_PulsedCTL', False)
    # enable_Blue = StatusVar('enable_Blue', False)
    # enable_PulsedBlue = StatusVar('enable_PulsedBlue', False)
    # RepumpWhenIonized = StatusVar('RepumpWhenIonized', False)
    # MW1_Freq = StatusVar('MW1_freq', 70)
    # MW2_Freq = StatusVar('MW2_freq', 140)
    # MW3_Freq = StatusVar('MW3_freq', 210)
    # _MW1_Power = StatusVar('MW1_Power', -21)
    # MW2_Power = StatusVar('MW2_Power', -21)
    # MW3_Power = StatusVar('MW3_Power', -21)
    # enable_MW1 = StatusVar('enable_MW1', True)
    # enable_MW2 = StatusVar('enable_MW2', True)
    # enable_MW3 = StatusVar('enable_MW3', False)

    
    def __init__(self, **kwargs):
        """ Create VoltageScanningLogic object with connectors.

          @param dict kwargs: optional parameters
        """
        super().__init__(**kwargs)

        # locking for thread safety
        self.threadlock = Mutex()
     
        
        

    def on_activate(self):
        """ Initialisation performed during activation of the module.
        """
        self._anc300: ANC300 = self.anc300() # Sample
        self._anc350: ANC350 = self.anc350() # Fiber
        self.last_time = 0


    def on_deactivate(self):
        """ Deinitialisation performed during deactivation of the module.
        """
        self.stopRequested = True
        self.AbortRequested = False
        self.measurement_running = False

    def reset_values(self,T:str):
        print("Reset to ",T)
        self.Frequency = self.Frequency_reset[T]
        self.Frequency_2 = self.Frequency_reset[T]
        self.Frequency_3 = self.Frequency_reset[T]
        self.Frequency_4 = self.Frequency_reset[T]
        self.Frequency_5 = self.Frequency_reset[T]
        self.Frequency_6 = self.Frequency_reset[T]
        self.Voltage = self.Voltage_reset[T]
        self.Voltage_2 = self.Voltage_reset[T]
        self.Voltage_3 = self.Voltage_reset[T]
        self.Voltage_4 = self.Voltage_reset[T]
        self.Voltage_5 = self.Voltage_reset[T]
        self.Voltage_6 = self.Voltage_reset[T]
        self.T_changed_signal.emit(T)

    def Capacitance_Button_clicked(self, on):
        print('done something with Capacitance_Button')
        self.Capacitance = self._anc300.measure_cap(AXIS[1]) # x
        self.C_changed_signal.emit(0,self.Capacitance)
    def Capacitance_Button_2_clicked(self, on):
        print('done something with Capacitance_Button')
        self.Capacitance_2 = self._anc300.measure_cap(AXIS[2]) # y
        self.C_changed_signal.emit(1,self.Capacitance_2)
    def Capacitance_Button_3_clicked(self, on):
        print('done something with Capacitance_Button')
        self.Capacitance_3 = self._anc300.measure_cap(AXIS[3]) # z
        self.C_changed_signal.emit(2,self.Capacitance_3)
    def Capacitance_Button_4_clicked(self, on):
        print('done something with Capacitance_Button')
        self.Capacitance_4 = self._anc350.measure_cap(AXIS[4]) # x
        self.C_changed_signal.emit(3,self.Capacitance_4)
    def Capacitance_Button_5_clicked(self, on):
        print('done something with Capacitance_Button')
        self.Capacitance_5 = self._anc350.measure_cap(AXIS[5]) # y
        self.C_changed_signal.emit(4,self.Capacitance_5)
    def Capacitance_Button_6_clicked(self, on):
        print('done something with Capacitance_Button')
        self.Capacitance_6 = self._anc350.measure_cap(AXIS[6]) # z
        self.C_changed_signal.emit(5,self.Capacitance_6)

    def Voltage_doubleSpinBox_changed(self, value):
        print('done something with Voltage_doubleSpinBox')
        self.Voltage = value
        self._anc300.voltage(AXIS[1],value)
    def Voltage_doubleSpinBox_2_changed(self, value):
        print('done something with Voltage_doubleSpinBox_2')
        self.Voltage_2 = value
        self._anc300.voltage(AXIS[2],value)
    def Voltage_doubleSpinBox_3_changed(self, value):
        print('done something with Voltage_doubleSpinBox_3')
        self.Voltage_3 = value
        self._anc300.voltage(AXIS[3],value)
    def Voltage_doubleSpinBox_4_changed(self, value):
        print('done something with Voltage_doubleSpinBox_4')
        self.Voltage_4 = value
        self._anc350.voltage(AXIS[4],value)
    def Voltage_doubleSpinBox_5_changed(self, value):
        print('done something with Voltage_doubleSpinBox_5')
        self.Voltage_5 = value
        self._anc350.voltage(AXIS[5],value)
    def Voltage_doubleSpinBox_6_changed(self, value):
        print('done something with Voltage_doubleSpinBox_6')
        self.Voltage_6 = value
        self._anc350.voltage(AXIS[6],value)

    def Frequency_doubleSpinBox_changed(self, value):
        print('done something with Frequency_doubleSpinBox')
        self.Frequency = value
        self._anc300.frequency(AXIS[1],value)
    def Frequency_doubleSpinBox_2_changed(self, value):
        print('done something with Frequency_doubleSpinBox_2')
        self.Frequency_2 = value
        self._anc300.frequency(AXIS[2],value)
    def Frequency_doubleSpinBox_3_changed(self, value):
        print('done something with Frequency_doubleSpinBox_3')
        self.Frequency_3 = value
        self._anc300.frequency(AXIS[3],value)
    def Frequency_doubleSpinBox_4_changed(self, value):
        print('done something with Frequency_doubleSpinBox_4')
        self.Frequency_4 = value
        self._anc350.frequency(AXIS[4],value)
    def Frequency_doubleSpinBox_5_changed(self, value):
        print('done something with Frequency_doubleSpinBox_5')
        self.Frequency_5 = value
        self._anc350.frequency(AXIS[5],value)
    def Frequency_doubleSpinBox_6_changed(self, value):
        print('done something with Frequency_doubleSpinBox_6')
        self.Frequency_6 = value
        self._anc350.frequency(AXIS[6],value)
    
    def single_down_pushButton_clicked(self, on):
        # print('done something with single_down_pushButton')
        self.perform_step(1,+1)
    def single_up_pushButton_clicked(self, on):
        # print('done something with single_up_pushButton')
        self.perform_step(1,-1)
    def single_down_pushButton_2_clicked(self, on):
        # print('done something with single_down_pushButton_2')
        self.perform_step(2,+1)
    def single_up_pushButton_2_clicked(self, on):
        # print('done something with single_up_pushButton_2')
        self.perform_step(2,-1)
    def single_down_pushButton_3_clicked(self, on):
        # print('done something with single_down_pushButton_3')
        self.perform_step(3,-1)
    def single_up_pushButton_3_clicked(self, on):
        # print('done something with single_up_pushButton_3')
        self.perform_step(3,+1)
    def single_down_pushButton_4_clicked(self, on):
        # print('done something with single_down_pushButton_4')
        self.perform_step(4,+1)
    def single_up_pushButton_4_clicked(self, on):
        # print('done something with single_up_pushButton_4')
        self.perform_step(4,-1)
    def single_down_pushButton_5_clicked(self, on):
        # print('done something with single_down_pushButton_5')
        self.perform_step(5,-1)
    def single_up_pushButton_5_clicked(self, on):
        # print('done something with single_up_pushButton_5')
        self.perform_step(5,+1)
    def single_down_pushButton_6_clicked(self, on):
        # print('done something with single_down_pushButton_6')
        self.perform_step(6,-1)
    def single_up_pushButton_6_clicked(self, on):
        # print('done something with single_up_pushButton_6')
        self.perform_step(6,+1)


    def perform_step(self,axis,direction):
        if axis < 4:
            self._anc300.perform_step(AXIS[axis],direction)
        else:
            self._anc350.perform_step(AXIS[axis],direction)
        self.anything_done()
    
    def move(self,axis,direction):
        if axis < 4:
            self._anc300.move(AXIS[axis],direction)
        else:
            self._anc350.move(AXIS[axis],direction)
        self.anything_done()
    
    def stop(self,axis):
        if axis < 4:
            self._anc300.stop(AXIS[axis])
        else:
            self._anc350.stop(AXIS[axis])
        self.anything_done()

    def anything_done(self):
        if time.time() - self.last_time > 10*60:
            p1 = []
            p2 = []
            for i in range(1,6+1):
                if i < 4:
                    p1 += [self._anc300.voltage(AXIS[i])]
                    p2 += [self._anc300.frequency(AXIS[i])]
                else:
                    p1 += [self._anc350.voltage(AXIS[i])]
                    p2 += [self._anc350.frequency(AXIS[i])]
            self.last_check = time.time()
            self.update_signal.emit([p1,p2])
        else:
            pass



    def cont_down_pushButton_clicked(self, on):
        # print('done something with cont_down_pushButton')
        self.enable_cont_S_x_down = on
        if on:
            self.move(1,"+")
        else:
            self.stop(1)
    def cont_up_pushButton_clicked(self, on):
        # print('done something with cont_up_pushButton')
        self.enable_cont_S_x_up = on
        if on:
            self.move(1,"-")
        else:
            self.stop(1)
    def cont_down_pushButton_2_clicked(self, on):
        # print('done something with cont_down_pushButton_2')
        self.enable_cont_S_y_down = on
        if on:
            self.move(2,"+")
        else:
            self.stop(2)
    def cont_up_pushButton_2_clicked(self, on):
        # print('done something with cont_up_pushButton_2')
        self.enable_cont_S_y_up = on
        if on:
            self.move(2,"-")
        else:
            self.stop(2)
    def cont_down_pushButton_3_clicked(self, on):
        # print('done something with cont_down_pushButton_3')
        self.enable_cont_S_z_down = on
        if on:
            self.move(3,"-")
        else:
            self.stop(3)
    def cont_up_pushButton_3_clicked(self, on):
        # print('done something with cont_up_pushButton_3')
        self.enable_cont_S_z_up = on
        if on:
            self.move(3,"+")
        else:
            self.stop(3)
    def cont_down_pushButton_4_clicked(self, on):
        # print('done something with cont_down_pushButton_4')
        self.enable_cont_F_x_down = on
        if on:
            self.move(4,"+")
        else:
            self.stop(4)
    def cont_up_pushButton_4_clicked(self, on):
        # print('done something with cont_up_pushButton_4')
        self.enable_cont_F_x_up = on
        if on:
            self.move(4,"-")
        else:
            self.stop(4)
    def cont_down_pushButton_5_clicked(self, on):
        # print('done something with cont_down_pushButton_5')
        self.enable_cont_F_y_down = on
        if on:
            self.move(5,"-")
        else:
            self.stop(5)
    def cont_up_pushButton_5_clicked(self, on):
        # print('done something with cont_up_pushButton_5')
        self.enable_cont_F_y_up = on
        if on:
            self.move(5,"+")
        else:
            self.stop(5)
    def cont_down_pushButton_6_clicked(self, on):
        # print('done something with cont_down_pushButton_6')
        self.enable_cont_F_z_down = on
        if on:
            self.move(6,"-")
        else:
            self.stop(6)
    def cont_up_pushButton_6_clicked(self, on):
        # print('done something with cont_up_pushButton_6')
        self.enable_cont_F_z_up = on
        if on:
            self.move(6,"+")
        else:
            self.stop(6)
