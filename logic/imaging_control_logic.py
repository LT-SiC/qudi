from core.connector import Connector
from logic.generic_logic import GenericLogic
from PyQt5 import QtCore
from PyQt5 import QtTest
import numpy as np
from core.statusvariable import StatusVar

from logic.setup_control_logic import SetupControlLogic
from hardware.PCINidaq6229 import StreamUSBNidaqInterface

class ImageLogic(GenericLogic):
    
    ''' Config Example
    imagelogic:
            module.Class: 'imaging_control_logic.ImageLogic'
            connect:
                USBnidaq: 'PCINIDAQ6229'
                setupcontrollogic: 'setupcontrollogic'
    '''
    # Implement Config options for voltage_offset and voltage_to_power_ratio
    USBnidaq = Connector(interface='StreamUSBNidaqInterface')
    setupcontrollogic = Connector(interface='SetupControlLogic')

    def on_activate(self):
        self._streaming_device:StreamUSBNidaqInterface = self.USBnidaq()
        self._setupcontrollogic:SetupControlLogic = self.setupcontrollogic()
        
        self.on_voltage = 0.07
        self.voltage_limits = [0,5]
        self.limits_active = True
        self._streaming_device.start_ao_task()
        
        self.set_voltage(0)

        self._setupcontrollogic.sigToggleLight.connect(self.toggle_emission)



    def on_deactivate(self):
        self._streaming_device.on_deactivate()
    
    def set_voltage(self, volt):
        if volt > self.voltage_limits[1]:
            print(f"Input to high! Voltage clipped to {self.voltage_limits[1]}")
        elif volt < self.voltage_limits[0]:
            print(f"Input to low! Voltage clipped to {self.voltage_limits[0]}")
        if self.limits_active == True:
            volt = np.clip(volt,self.voltage_limits[0],self.voltage_limits[1])
        self._streaming_device.goToVoltage(volt)

    def ON(self):
        self.set_voltage(self.on_voltage)

    def OFF(self):
        self.set_voltage(0)

    def toggle_emission(self,state):
        if state:
            self.ON()
        else:
            self.OFF()