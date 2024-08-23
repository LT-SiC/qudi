from core.connector import Connector
from logic.generic_logic import GenericLogic
from PyQt5 import QtCore
from PyQt5 import QtTest
import numpy as np
from core.statusvariable import StatusVar

class BiasLogic(GenericLogic):
    
    ''' Config Example
    biaslogic:
            module.Class: 'biaslogic.BiasLogic'
            connect:
                USBnidaq: 'USBNIDAQ6001'
                laserscannerlogic: 'laserscannerlogic'
    '''
    # Implement Config options for voltage_offset and voltage_to_power_ratio
    USBnidaq = Connector(interface='StreamUSBNidaqInterface')
    laserscannerlogic = Connector(interface='LaserScannerLogic')

    def on_activate(self):
        self._streaming_device = self.USBnidaq()
        self._laser_scanner_logic = self.laserscannerlogic()
        
        self.voltage_limits = [-30,15]
        self.limits_active = True
        self.amplification = 10
        self.v_offset_correction = 0.129 
        self.voltages = [0,0]# [0,0.1,0.2,0.3,0.4,0.5,0.4,.3, 2., .1, 0, -.1, -.2, -.3, -.4]
        self._laser_scanner_logic.sigScanNextLine.connect(self.change_voltage)
        self.step_line = 1
        self._streaming_device.start_ao_task()
        
        self.set_voltage(0)



    def on_deactivate(self):
        self._streaming_device.on_deactivate()
    
    def set_voltage(self, volt):
        if volt > self.voltage_limits[1]:
            print(f"Input to high! Voltage clipped to {self.voltage_limits[1]}")
        elif volt < self.voltage_limits[0]:
            print(f"Input to low! Voltage clipped to {self.voltage_limits[0]}")
        if self.limits_active == True:
            volt = np.clip(volt,self.voltage_limits[0],self.voltage_limits[1])
        self._streaming_device.goToVoltage((volt-self.v_offset_correction)/self.amplification)

    def change_voltage(self):
        current_scan_line = self._laser_scanner_logic._scan_counter_up
        while current_scan_line >= len(self.voltages):
            current_scan_line = current_scan_line-len(self.voltages)
        self.set_voltage(self.voltages[int(current_scan_line/self.step_line)])

    def set_voltages(self, start, stop, step):
        v_list = []
        v_list.extend(np.round(np.arange(start,stop+step,step),3))
        v_list.extend(np.round(np.arange(start,stop,step)[::-1][:-1],3))
        self.voltages = v_list