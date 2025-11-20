from cv2 import QT_CHECKBOX
from core.connector import Connector
from logic.generic_logic import GenericLogic
from PyQt5 import QtCore
from PyQt5 import QtTest
import numpy as np
from core.statusvariable import StatusVar
import ast
import re

from logic.laserscanner.duo_laser_scanner_logic import LaserScannerLogic
from hardware.USBNidaq6211 import StreamUSBNidaqInterface
# from logic.laserscanner.laser_scanner_logic import LaserScannerLogic

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

    same_length = QtCore.Signal(float)

    voltage_min = StatusVar("voltage_min",-30)
    voltage_max = StatusVar("voltage_max",15)
    amplification = StatusVar("amplification",10)
    startV = StatusVar("startV",0)
    stopV = StatusVar("stopV",0)
    stepV = StatusVar("stepV",1)
    reps = StatusVar("reps",1)
    reverse = StatusVar("reverse",True)
    v_offset_correction = StatusVar("v_offset_correction",0.115 )
    constV = StatusVar("constV",0.)
    custom_seq = StatusVar("custom_seq",[0,0])  # [0,0.1,0.2,0.3,0.4,0.5,0.4,.3, 2., .1, 0, -.1, -.2, -.3, -.4]
    
    voltage_min_2 = StatusVar("voltage_min_2",-30)#
    voltage_max_2 = StatusVar("voltage_max_2",15)
    amplification_2 = StatusVar("amplification_2",10)
    startV_2 = StatusVar("startV_2",0)
    stopV_2 = StatusVar("stopV_2",0)#
    stepV_2 = StatusVar("stepV_2",19)#
    reps_2 = StatusVar("reps_2",1)
    reverse_2 = StatusVar("reverse_2",True)
    v_offset_correction_2 = StatusVar("v_offset_correction_2",0.115 )#
    constV_2 = StatusVar("constV_2",0.)
    custom_seq_2 = StatusVar("custom_seq_2",[0,0])  # [0,0.1,0.2,0.3,0.4,0.5,0.4,.3, 2., .1, 0, -.1, -.2, -.3, -.4]

    equal_length = 0.


    def on_activate(self):

        self._streaming_device:StreamUSBNidaqInterface = self.USBnidaq()
        self._laser_scanner_logic:LaserScannerLogic = self.laserscannerlogic()
        
        

        self.limits_active = True
        self.voltages = [0,0]
        self.voltages_2 = [0,0]
        self._laser_scanner_logic.sigScanNextLine.connect(self.change_voltage)
        self._laser_scanner_logic.sigScanFinished.connect(self.store_voltages)
        self._streaming_device.start_ao_task()
        
        self.set_voltage([self.constV,self.constV_2])
        self.parameters_changed()



    def on_deactivate(self):
        self._streaming_device.on_deactivate()
    
    def set_voltage(self, volts):
        volt1,volt2 = volts
        
        if volt1 > self.voltage_max:
            print(f"Input to high! Voltage clipped to {self.voltage_max}")
        elif volt1 < self.voltage_min:
            print(f"Input to low! Voltage clipped to {self.voltage_min}")
        if self.limits_active == True:
            volt1 = np.clip(volt1,self.voltage_min,self.voltage_max)
        
        if volt2 > self.voltage_max_2:
            print(f"Input to high! Voltage2 clipped to {self.voltage_max_2}")
        elif volt2 < self.voltage_min_2:
            print(f"Input to low! Voltage2 clipped to {self.voltage_min_2}")
        if self.limits_active == True:
            volt2 = np.clip(volt2,self.voltage_min_2,self.voltage_max_2)
            
        self._streaming_device.MultiplegoToVoltage([(volt1-self.v_offset_correction)/self.amplification,(volt2-self.v_offset_correction_2)/self.amplification_2])

    def change_voltage(self):
        current_scan_line = self._laser_scanner_logic._scan_counter_up
        current_scan_line_2 = self._laser_scanner_logic._scan_counter_up

        volt_list = self.voltages
        volt_list_2 = self.voltages_2

        while current_scan_line >= len(volt_list):
            current_scan_line = current_scan_line-len(volt_list)
        while current_scan_line_2 >= len(volt_list_2):
            current_scan_line_2 = current_scan_line_2-len(volt_list_2)
        self.set_voltage([volt_list[int(current_scan_line)],volt_list_2[int(current_scan_line_2)]])

    def set_voltages(self, start, stop, step, rev, N):
        if step == 0:
            step = 1
        v_list = []
        if start < stop:
            v_list.extend(np.round(np.sort(np.tile(np.arange(start,stop+step,step),int(N))),3))
            if rev:
                v_list.extend(np.round(np.sort(np.tile(np.arange(start,stop+step,step),int(N))),3)[::-1])
        else:
            v_list.extend(np.round(np.sort(np.tile(np.arange(start,stop+step,step),int(N))),3)[::-1])
            if rev:
                v_list.extend(np.round(np.sort(np.tile(np.arange(start,stop+step,step),int(N))),3))
        return v_list

    def parameters_changed(self):
        self.voltages = self.set_voltages(self.startV,self.stopV,self.stepV,self.reverse,self.reps)
        self.voltages_2 = self.set_voltages(self.startV_2,self.stopV_2,self.stepV_2,self.reverse_2,self.reps_2)
        openK = self.custom_seq.count("[")
        closeK = self.custom_seq.count("]")
        if (openK == closeK) and (openK > 0):

        # if ("[" in self.custom_seq) and ("]" in self.custom_seq):
        #     if "-" not in self.custom_seq.split("[")[1].split("]")[0].split(","):
        #         if "+" not in self.custom_seq.split("[")[1].split("]")[0].split(","):
            try:
                self.voltages = self.parse_expression(self.custom_seq)
            except:
                pass
        openK = self.custom_seq_2.count("[")
        closeK = self.custom_seq_2.count("]")
        if (openK == closeK) and (openK > 0):
            try:
                self.voltages_2 = self.parse_expression(self.custom_seq_2)
            except:
                pass

        if len(self.voltages) >= len(self.voltages_2):
            self.equal_length = len(self.voltages_2)/len(self.voltages)*100
        else: 
            self.equal_length = len(self.voltages)/len(self.voltages_2)*100
        self.same_length.emit(self.equal_length)

    
    def store_voltages(self):
        L = self._laser_scanner_logic._scan_counter_up
        V = self.voltages
        while len(V) < L:
            V += V
        self._laser_scanner_logic.bias_voltages = V[:L]

        V2 = self.voltages_2
        while len(V2) < L:
            V2 += V2
        self._laser_scanner_logic.bias_voltages_2 = V2[:L]

    
    def parse_expression(self, expr:str):
        result = []

        tokens = re.findall(r'\[[^\]]*\](?:\*\d+)?',expr)

        for token in tokens:
            arr_str = re.search(r'\[[^\]]*\]', token).group()
            arr = ast.literal_eval(arr_str)

            mult_match = re.search(r'\*(\d+)', token)
            mult = int(mult_match.group(1)) if mult_match else 1

            result.extend(arr * mult)
        return result