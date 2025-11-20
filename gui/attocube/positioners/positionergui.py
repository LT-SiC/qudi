# -*- coding: utf-8 -*-

"""
This file contains a gui for the setupcontroll.

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
from interface.simple_laser_interface import ControlMode, ShutterState, LaserState
from qtpy import QtCore
from qtpy import QtWidgets
from qtpy import uic

from logic.positioner.positioner_logic import PositionControlLogic
from gui.attocube.positioners.connectors_and_set_default import initialize_connections_and_defaultvalue
from hardware.Attocube.anc3XX import AXIS

class PositionControlWindow(QtWidgets.QMainWindow):
    """ Create the Main Window based on the *.ui file. """

    def __init__(self):
        # Get the path to the *.ui file
        this_dir = os.path.dirname(__file__)
        ui_file = os.path.join(this_dir, 'positionergui.ui')

        # Load it
        super().__init__()
        uic.loadUi(ui_file, self)
        self.show()

class PositionControlGUI(GUIBase, initialize_connections_and_defaultvalue):


    ## declare signals
    toggled_radioButton = QtCore.Signal(bool,bool,int)
    

    ## declare connectors
    positioncontrollogic = Connector(interface='PositionControlLogic')



    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)
    
    def on_activate(self):
        """ Definition and initialisation of the GUI plus staring the measurement.
        """

        self._mw = PositionControlWindow()
        self._positioncontrollogic:PositionControlLogic = self.positioncontrollogic() 
        self.initialize_connections_and_defaultvalues()

        self._positioncontrollogic.T_changed_signal.connect(self.reset_VoltFreq_spinBoxes)
        self._positioncontrollogic.update_signal.connect(self.update_VoltFreq_spinBoxes)
        self._positioncontrollogic.C_changed_signal.connect(self.update_Display_C)
        self.toggled_radioButton.connect(self.change_pushButtons)

    #     #####################
    #     # Configuring the dock widgets
    #     # Use the inherited class 'CounterMainWindow' to create the GUI window
    #     # Setup dock widgets
    #     self._mw.setDockNestingEnabled(True)
        
        # self.updateButtonsEnabled()
    #     self._mw.A1_Button.clicked.connect(self._setupcontrol_logic.A1_Button_Clicked)


    def on_deactivate(self):
        """ Deactivate the module properly.
        """
        # self._mw.A1_Button.clicked.disconnect()
        

        self._mw.close()


    def reset_VoltFreq_spinBoxes(self,T):
        V0 = self._positioncontrollogic.Voltage_reset[T]
        F0 = self._positioncontrollogic.Frequency_reset[T]
        for Box1, Box2 in zip(self.Voltage_doubleSpinBox_list,self.Frequency_doubleSpinBox_list):
            Box1.setValue(V0)
            Box2.setValue(F0)
        state = {"RT":0,"LT":1}[T]
        for Slider in self.T_Slider_list:
            Slider.setSliderPosition(state)
        # print("T changed")

    def update_VoltFreq_spinBoxes(self,params):
        for Box1, Box2, V0, F0 in zip(self.Voltage_doubleSpinBox_list,self.Frequency_doubleSpinBox_list,params[0],params[1]):
            Box1.setValue(V0)
            Box2.setValue(F0)

    def update_Display_C(self,axis,value):
        if not np.isnan(value):
            self.lcdNumber_list[axis].display(value*1e9)
        else:
            self.lcdNumber_list[axis].display("Error")


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
    

    def axis_enabled_checkBox_changed(self, value):
        print('done something with axis_enabled_checkBox')
        self._positioncontrollogic.enable_S_x = value == 2
        self.axis_enabled_changed(value,0)
    def axis_enabled_checkBox_2_changed(self, value):
        print('done something with axis_enabled_checkBox_2')
        self._positioncontrollogic.enable_S_y = value == 2
        self.axis_enabled_changed(value,1)
    def axis_enabled_checkBox_3_changed(self, value):
        print('done something with axis_enabled_checkBox_3')
        self._positioncontrollogic.enable_S_z = value == 2
        self.axis_enabled_changed(value,2)
    def axis_enabled_checkBox_4_changed(self, value):
        print('done something with axis_enabled_checkBox_4')
        self._positioncontrollogic.enable_F_x = value == 2
        self.axis_enabled_changed(value,3)
    def axis_enabled_checkBox_5_changed(self, value):
        print('done something with axis_enabled_checkBox_5')
        self._positioncontrollogic.enable_F_y = value == 2
        self.axis_enabled_changed(value,4)
    def axis_enabled_checkBox_6_changed(self, value):
        print('done something with axis_enabled_checkBox_6')
        self._positioncontrollogic.enable_F_z = value == 2
        self.axis_enabled_changed(value,5)

    def axis_enabled_changed(self,value,axis):
        if axis < 3:
            if value:
                self._positioncontrollogic._anc300.enable_axis(AXIS[axis+1])
            else:
                self._positioncontrollogic._anc300.disable_axis(AXIS[axis+1])
        else:
            if value:
                self._positioncontrollogic._anc350.enable_axis(AXIS[axis+1])
            else:
                self._positioncontrollogic._anc350.disable_axis(AXIS[axis+1])
        for Button in [element for (i,element) in enumerate(list(self.gui_elements_list.values())) if i in [1,5,6,7,8,9,10]]:
            Button[axis].setEnabled(value)
        if value:
            radio_state = self.gui_elements_list["sing"][axis].isChecked()
            self.toggled_radioButton.emit(radio_state, not radio_state, axis)
        else:
            self.toggled_radioButton.emit(value, value, axis)
   

    def single_radioButton_clicked(self, value):
        print('done something with single_radioButton')
        self._positioncontrollogic.enable_cont_S_x = False
        self.toggled_radioButton.emit(value,not value, 0)
    
    def cont_radioButton_clicked(self, value):
        print('done something with cont_radioButton')
        self._positioncontrollogic.enable_cont_S_x = True
        self.toggled_radioButton.emit(not value, value, 0)

    def single_radioButton_2_clicked(self, value):
        print('done something with single_radioButton_2')
        self._positioncontrollogic.enable_cont_S_y = False
        self.toggled_radioButton.emit(value,not value, 1)
    
    def cont_radioButton_2_clicked(self, value):
        print('done something with cont_radioButton_2')
        self._positioncontrollogic.enable_cont_S_y = True
        self.toggled_radioButton.emit(not value, value, 1)

    def single_radioButton_3_clicked(self, value):
        print('done something with single_radioButton_3')
        self._positioncontrollogic.enable_cont_S_z = False
        self.toggled_radioButton.emit(value,not value, 2)
    
    def cont_radioButton_3_clicked(self, value):
        print('done something with cont_radioButton_3')
        self._positioncontrollogic.enable_cont_S_z = True
        self.toggled_radioButton.emit(not value, value, 2)

    def single_radioButton_4_clicked(self, value):
        print('done something with single_radioButton_4')
        self._positioncontrollogic.enable_cont_F_x = False
        self.toggled_radioButton.emit(value,not value, 3)
    
    def cont_radioButton_4_clicked(self, value):
        print('done something with cont_radioButton_4')
        self._positioncontrollogic.enable_cont_F_x = True
        self.toggled_radioButton.emit(not value, value, 3)

    def single_radioButton_5_clicked(self, value):
        print('done something with single_radioButton_5')
        self._positioncontrollogic.enable_cont_F_y = False
        self.toggled_radioButton.emit(value,not value, 4)
    
    def cont_radioButton_5_clicked(self, value):
        print('done something with cont_radioButton_5')
        self._positioncontrollogic.enable_cont_F_y = True
        self.toggled_radioButton.emit(not value, value, 4)

    def single_radioButton_6_clicked(self, value):
        print('done something with single_radioButton_6')
        self._positioncontrollogic.enable_cont_F_z = False
        self.toggled_radioButton.emit(value,not value, 5)
    
    def cont_radioButton_6_clicked(self, value):
        print('done something with cont_radioButton_6')
        self._positioncontrollogic.enable_cont_F_z = True
        self.toggled_radioButton.emit(not value, value, 5)

    def change_pushButtons(self,single_bool,cont_bool,axis):
        self.gui_elements_list["singUp"][axis].setEnabled(single_bool)
        self.gui_elements_list["singD"][axis].setEnabled(single_bool)
        self.gui_elements_list["conUp"][axis].setEnabled(cont_bool)
        self.gui_elements_list["conD"][axis].setEnabled(cont_bool)

