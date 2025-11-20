from core.connector import Connector
from logic.generic_logic import GenericLogic
from PyQt5 import QtCore
from PyQt5 import QtTest
import numpy as np
from core.statusvariable import StatusVar

from logic.setup_control_logic import SetupControlLogic
from hardware.PCINidaq6229 import StreamUSBNidaqInterface

class FlipMirrorLogic(GenericLogic):
    
    ''' Config Example
    mirrorlogic:
            module.Class: 'flipmirror_control_logic.FlipMirrorLogic'
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
        
        self._streaming_device.start_ao_task()
        
        self._setupcontrollogic.sigToggleMirror.connect(self.toggle_mirrors)
        self._setupcontrollogic.sigChangeA1.connect(self.adjustA1power)



    def on_deactivate(self):
        self._streaming_device.on_deactivate()


    def toggle_mirrors(self):
        mirror_states = [self._setupcontrollogic.enable_TopSpectrum,self._setupcontrollogic.enable_FiberSpectrum,self._setupcontrollogic.enable_FiberExcitation
                         ,self._setupcontrollogic.enable_TwoPLE,self._setupcontrollogic.enable_PLCLaser]
        self._streaming_device.flip(pos=mirror_states)

    def adjustA1power(self,volt):
        self._streaming_device.goToAOMVoltage(volt)