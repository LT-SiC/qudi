# -*- coding: utf-8 -*-

"""
This file contains a gui for the biascontroll.

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

import numpy as np
import os
import pyqtgraph as pg
import time

from core.connector import Connector
from gui.colordefs import QudiPalettePale as palette
from gui.guibase import GUIBase
from PyQt5 import QtTest
from interface.simple_laser_interface import ControlMode, ShutterState, LaserState
from qtpy import QtCore
from qtpy import QtWidgets
from qtpy import uic

from logic.biaslogic import BiasLogic


class BiasControlWindow(QtWidgets.QMainWindow):
    """ Create the Main Window based on the *.ui file. """

    def __init__(self):
        # Get the path to the *.ui file
        this_dir = os.path.dirname(__file__)
        ui_file = os.path.join(this_dir, 'bias_control.ui')

        # Load it
        super().__init__()
        uic.loadUi(ui_file, self)
        self.show()

class BiasControlGUI(GUIBase):

    ## declare connectors
    biaslogic = Connector(interface='BiasLogic')
    #pulsedlogic= Connector(interface='SetupControlLogic')


    def __init__(self, config, **kwargs):
        
        super().__init__(config=config, **kwargs)
        

    
    def on_activate(self):
        """ Definition and initialisation of the GUI plus staring the measurement.
        """
        
        self._bias_logic:BiasLogic = self.biaslogic() 



        self._bias_logic.same_length.connect(self.update_IndicatorBar)

        #####################
        # Configuring the dock widgets
        # Use the inherited class 'CounterMainWindow' to create the GUI window
        self._mw = BiasControlWindow()
        # Setup dock widgets
        self._mw.setDockNestingEnabled(True)

        self._mw.MinV_doubleSpinBox.setValue(self._bias_logic.voltage_min)
        self._mw.MaxV_doubleSpinBox.setValue(self._bias_logic.voltage_max)
        self._mw.Amp_doubleSpinBox.setValue(self._bias_logic.amplification)
        self._mw.StartV_doubleSpinBox.setValue(self._bias_logic.startV)
        self._mw.StopV_doubleSpinBox.setValue(self._bias_logic.stopV)
        self._mw.StepV_doubleSpinBox.setValue(self._bias_logic.stepV)
        self._mw.RepN_doubleSpinBox.setValue(self._bias_logic.reps)
        self._mw.correction_doubleSpinBox.setValue(self._bias_logic.v_offset_correction)
        self._mw.constV_doubleSpinBox.setValue(self._bias_logic.constV)
        
        self._mw.MinV_doubleSpinBox.valueChanged.connect(self.voltage_min_Edited)
        self._mw.MaxV_doubleSpinBox.valueChanged.connect(self.voltage_max_Edited)
        self._mw.Amp_doubleSpinBox.valueChanged.connect(self.amplification_Edited)
        self._mw.StartV_doubleSpinBox.valueChanged.connect(self.startV_Edited)
        self._mw.StopV_doubleSpinBox.valueChanged.connect(self.stopV_Edited)
        self._mw.StepV_doubleSpinBox.valueChanged.connect(self.stepV_Edited)
        self._mw.RepN_doubleSpinBox.valueChanged.connect(self.reps_Edited)
        self._mw.correction_doubleSpinBox.valueChanged.connect(self.v_offset_correction_Edited)
        self._mw.constV_doubleSpinBox.valueChanged.connect(self.constV_Edited)

        self._mw.Custom_lineEdit.textEdited.connect(self.Custom_lineEdit_textEdited)
        self._mw.Custom_lineEdit.setText(str(self._bias_logic.custom_seq))

        self._mw.Reverse_checkBox.stateChanged.connect(self.Reverse_checkBox_StateChanged)
        self._mw.Reverse_checkBox.setChecked(self._bias_logic.reverse)
        self._mw.settings_checkBox.stateChanged.connect(self.settings_checkBox_StateChanged)
        # self._mw.settings_checkBox.setChecked(self._bias_logic.)
        
        self._mw.SetVoltage_pushButton.clicked.connect(self.SetVoltage_pushButton_clicked)
        
        
        
        # Voltage 2
        self._mw.MinV_doubleSpinBox_2.setValue(self._bias_logic.voltage_min_2) #
        self._mw.MaxV_doubleSpinBox_2.setValue(self._bias_logic.voltage_max_2)
        self._mw.Amp_doubleSpinBox_2.setValue(self._bias_logic.amplification_2)
        self._mw.StartV_doubleSpinBox_2.setValue(self._bias_logic.startV_2)
        self._mw.StopV_doubleSpinBox_2.setValue(self._bias_logic.stopV_2)#
        self._mw.StepV_doubleSpinBox_2.setValue(self._bias_logic.stepV_2)#
        self._mw.RepN_doubleSpinBox_2.setValue(self._bias_logic.reps_2)
        self._mw.correction_doubleSpinBox_2.setValue(self._bias_logic.v_offset_correction_2)#
        self._mw.constV_doubleSpinBox_2.setValue(self._bias_logic.constV_2)#

        self._mw.progressBar.setValue(self._bias_logic.equal_length)


        self._mw.MinV_doubleSpinBox_2.valueChanged.connect(self.voltage_min_Edited)
        self._mw.MaxV_doubleSpinBox_2.valueChanged.connect(self.voltage_max_Edited)
        self._mw.Amp_doubleSpinBox_2.valueChanged.connect(self.amplification_Edited)
        self._mw.StartV_doubleSpinBox_2.valueChanged.connect(self.startV_Edited)
        self._mw.StopV_doubleSpinBox_2.valueChanged.connect(self.stopV_Edited)
        self._mw.StepV_doubleSpinBox_2.valueChanged.connect(self.stepV_Edited)
        self._mw.RepN_doubleSpinBox_2.valueChanged.connect(self.reps_Edited)
        self._mw.correction_doubleSpinBox_2.valueChanged.connect(self.v_offset_correction_Edited)
        self._mw.constV_doubleSpinBox_2.valueChanged.connect(self.constV_Edited)

        self._mw.Custom_lineEdit_2.textEdited.connect(self.Custom_lineEdit_textEdited)
        self._mw.Custom_lineEdit_2.setText(str(self._bias_logic.custom_seq_2))

        self._mw.Reverse_checkBox_2.stateChanged.connect(self.Reverse_checkBox_StateChanged)
        self._mw.Reverse_checkBox_2.setChecked(self._bias_logic.reverse_2)
        
        
        

    def on_deactivate(self):
        """ Deactivate the module properly.
        """
        self._mw.MinV_doubleSpinBox.valueChanged.disconnect()
        self._mw.MaxV_doubleSpinBox.valueChanged.disconnect()
        self._mw.Amp_doubleSpinBox.valueChanged.disconnect()
        self._mw.StartV_doubleSpinBox.valueChanged.disconnect()
        self._mw.StopV_doubleSpinBox.valueChanged.disconnect()
        self._mw.StepV_doubleSpinBox.valueChanged.disconnect()
        self._mw.RepN_doubleSpinBox.valueChanged.disconnect()
        self._mw.correction_doubleSpinBox.valueChanged.disconnect()
        self._mw.constV_doubleSpinBox.valueChanged.disconnect()

        self._mw.Custom_lineEdit.textEdited.disconnect()
        self._mw.Reverse_checkBox.stateChanged.disconnect()


        self._mw.MinV_doubleSpinBox_2.valueChanged.disconnect()
        self._mw.MaxV_doubleSpinBox_2.valueChanged.disconnect()
        self._mw.Amp_doubleSpinBox_2.valueChanged.disconnect()
        self._mw.StartV_doubleSpinBox_2.valueChanged.disconnect()
        self._mw.StopV_doubleSpinBox_2.valueChanged.disconnect()
        self._mw.StepV_doubleSpinBox_2.valueChanged.disconnect()
        self._mw.RepN_doubleSpinBox_2.valueChanged.disconnect()
        self._mw.correction_doubleSpinBox_2.valueChanged.disconnect()
        self._mw.constV_doubleSpinBox_2.valueChanged.disconnect()

        self._mw.Custom_lineEdit_2.textEdited.disconnect()
        self._mw.Reverse_checkBox_2.stateChanged.disconnect()

        
        self._mw.settings_checkBox.stateChanged.disconnect()


        self._mw.close()

    def show(self):
        """Make window visible and put it above all other windows.
        """
        QtWidgets.QMainWindow.show(self._mw)
        self._mw.activateWindow()
        self._mw.raise_()
    
    def restoreDefaultView(self):
        """ Restore the arrangement of DockWidgets to the default
        """
        # Show any hidden dock widgets
        self._mw.adjustDockWidget.show()
        self._mw.plotDockWidget.show()

        # re-dock any floating dock widgets
        self._mw.adjustDockWidget.setFloating(False)
        self._mw.plotDockWidget.setFloating(False)

        # Arrange docks widgets
        self._mw.addDockWidget(QtCore.Qt.DockWidgetArea(1), self._mw.adjustDockWidget)
        self._mw.addDockWidget(QtCore.Qt.DockWidgetArea(2), self._mw.plotDockWidget)
    
    def voltage_min_Edited(self):
        self._bias_logic.voltage_min = self._mw.MinV_doubleSpinBox.value()
        self._bias_logic.voltage_min_2 = self._mw.MinV_doubleSpinBox_2.value()
        self._bias_logic.parameters_changed()
        self.refresh_voltage_length()
    
    def voltage_max_Edited(self):
        self._bias_logic.voltage_max = self._mw.MaxV_doubleSpinBox.value()
        self._bias_logic.voltage_max_2 = self._mw.MaxV_doubleSpinBox_2.value()
        self._bias_logic.parameters_changed()
        self.refresh_voltage_length()
    
    def amplification_Edited(self):
        self._bias_logic.amplification = self._mw.Amp_doubleSpinBox.value()
        self._bias_logic.amplification_2 = self._mw.Amp_doubleSpinBox_2.value()
        self._bias_logic.parameters_changed()
        self.refresh_voltage_length()

    def startV_Edited(self):
        self._bias_logic.startV = self._mw.StartV_doubleSpinBox.value()
        self._bias_logic.startV_2 = self._mw.StartV_doubleSpinBox_2.value()
        self._bias_logic.parameters_changed()
        self.refresh_voltage_length()

    def stopV_Edited(self):
        self._bias_logic.stopV = self._mw.StopV_doubleSpinBox.value()
        self._bias_logic.stopV_2 = self._mw.StopV_doubleSpinBox_2.value()
        self._bias_logic.parameters_changed()
        self.refresh_voltage_length()

    def stepV_Edited(self):
        self._bias_logic.stepV = self._mw.StepV_doubleSpinBox.value()
        self._bias_logic.stepV_2 = self._mw.StepV_doubleSpinBox_2.value()
        self._bias_logic.parameters_changed()
        self.refresh_voltage_length()

    def reps_Edited(self):
        self._bias_logic.reps = int(self._mw.RepN_doubleSpinBox.value())
        self._bias_logic.reps_2 = int(self._mw.RepN_doubleSpinBox_2.value())
        self._bias_logic.parameters_changed()
        self.refresh_voltage_length()

    def v_offset_correction_Edited(self):
        self._bias_logic.v_offset_correction = self._mw.correction_doubleSpinBox.value()
        self._bias_logic.v_offset_correction_2 = self._mw.correction_doubleSpinBox_2.value()
        self._bias_logic.parameters_changed()
        self.refresh_voltage_length()
    
    def constV_Edited(self):
        self._bias_logic.constV = self._mw.constV_doubleSpinBox.value()
        self._bias_logic.constV_2 = self._mw.constV_doubleSpinBox_2.value()

    def Custom_lineEdit_textEdited(self):
        self._bias_logic.custom_seq = self._mw.Custom_lineEdit.text()
        self._bias_logic.custom_seq_2 = self._mw.Custom_lineEdit_2.text()
        self._bias_logic.parameters_changed()
        self.refresh_voltage_length()

    def Reverse_checkBox_StateChanged(self):
        self._bias_logic.reverse = self._mw.Reverse_checkBox.isChecked()
        self._bias_logic.reverse_2 = self._mw.Reverse_checkBox_2.isChecked()
        self._bias_logic.parameters_changed()
        self.refresh_voltage_length()

    def settings_checkBox_StateChanged(self):
        if self._mw.settings_checkBox.isChecked():
            self._mw.MaxV_doubleSpinBox.setEnabled(True)
            self._mw.MinV_doubleSpinBox.setEnabled(True)
            self._mw.Amp_doubleSpinBox.setEnabled(True)
            self._mw.correction_doubleSpinBox.setEnabled(True)
            self._mw.MaxV_doubleSpinBox_2.setEnabled(True)
            self._mw.MinV_doubleSpinBox_2.setEnabled(True)
            self._mw.Amp_doubleSpinBox_2.setEnabled(True)
            self._mw.correction_doubleSpinBox_2.setEnabled(True)
        elif not self._mw.settings_checkBox.isChecked():
            self._mw.MaxV_doubleSpinBox.setEnabled(False)
            self._mw.MinV_doubleSpinBox.setEnabled(False)
            self._mw.Amp_doubleSpinBox.setEnabled(False)
            self._mw.correction_doubleSpinBox.setEnabled(False)
            self._mw.MaxV_doubleSpinBox_2.setEnabled(False)
            self._mw.MinV_doubleSpinBox_2.setEnabled(False)
            self._mw.Amp_doubleSpinBox_2.setEnabled(False)
            self._mw.correction_doubleSpinBox_2.setEnabled(False)

    def SetVoltage_pushButton_clicked(self):
        self._bias_logic.set_voltage([self._bias_logic.constV,self._bias_logic.constV_2])

    def refresh_voltage_length(self):
        self._mw.Length_Display.display(len(self._bias_logic.voltages))
        self._mw.Length_Display_2.display(len(self._bias_logic.voltages_2))

    def update_IndicatorBar(self, val):
        self._mw.progressBar.setValue(val)