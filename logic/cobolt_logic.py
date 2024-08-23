from core.connector import Connector
from logic.generic_logic import GenericLogic
from PyQt5 import QtCore
from PyQt5 import QtTest
import numpy as np
from core.statusvariable import StatusVar

from interface.cobolt_interface import CoboltInterface
from hardware.laser.cobolt_laser import HubnerCobolt

class CoboltLogic(GenericLogic,CoboltInterface):
    
    ''' Config Example
    cobolt_logic:
        module.Class: 'cobolt_logic.CoboltLogic'
        connect:
            HubnerCobolt: 'HubnerCobolt'
    '''
    HubnerCobolt = Connector(interface='HubnerCobolt')

    def on_activate(self):
        self._cobolt: HubnerCobolt = self.HubnerCobolt()
        
        # self._cobolt.on_activate()

    def on_deactivate(self):
        pass
        # self._cobolt.on_deactivate()

    def power(self,val: float=None):
        if val:
            """Set the modulation power in mW"""
            self._cobolt.modulation_mode(val)
        else:
            """Get the modulation power setpoint in mW"""
            return self._cobolt.get_modulation_power()

    def current(self,val: float=None):
        if val:
            self._cobolt.set_current(val)
        else:
            return self._cobolt.get_current()

    # def emission(self,val: bool=None):
    #     if val == True:
    #         self._CTL.current_enabled = val
    #     elif val == False:
    #         self._CTL.current_enabled = val
    #     else:
    #         return self._CTL.current_enabled

    # def ON(self):
    #     self._CTL.current_enabled = True

    # def OFF(self):
    #     self._CTL.current_enabled = False
