from core.connector import Connector
from logic.generic_logic import GenericLogic
from PyQt5 import QtCore
from PyQt5 import QtTest
import numpy as np
from core.statusvariable import StatusVar

from interface.CTL_interface import CTLInterface
from hardware.laser.Toptica.toptica_laser import TopticaCTL,TopticaDLCPro, External_Signal_Channels

class CTLLogic(GenericLogic,CTLInterface):
    
    ''' Config Example
    ctllogic:
            module.Class: 'TopticaDLCLogic.CTLLogic'
            connect:
                TopticaCTL: 'TopticaCTL'
    '''
    # Implement Config options for voltage_offset and voltage_to_power_ratio
    TopticaCTL = Connector(interface='TopticaCTL')

    def on_activate(self):
        self._laser:TopticaCTL = self.TopticaCTL()
        self._laser.dlc.open()

    def on_deactivate(self):
        self._laser.on_deactivate()

    def emission(self):
        return self._laser.emission

    def wavelength(self,val: float=None):
        if val:
            print("Setting CTL wavelength to ", val,"nm.")
            self._laser.wavelength_setpoint = val
        else:
            return self._laser.wavelength_setpoint

    def current(self,val: float=None):
        if val:
            self._laser.current_setpoint = val
        else:
            return self._laser.current_setpoint

    def CCemission(self,val: bool=None):
        if val == True:
            self._laser.current_enabled = val
        elif val == False:
            self._laser.current_enabled = val
        else:
            return self._laser.current_enabled

    def ON(self):
        self._laser.current_enabled = True

    def OFF(self):
        self._laser.current_enabled = False

    # Piezo controll properties ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

    
    def pc_enabled(self, val: bool=None):
        if type(val) == bool:
            self._laser.dlc.laser1.dl.pc.enabled.set(val)
        else:
            return self._laser.dlc.laser1.dl.pc.enabled.get()
            
    def ext_enabled(self, val: bool=None):
        if type(val) == bool:
            self._laser.dlc.laser1.dl.pc.external_input.enabled.set(val)
        else:
            return self._laser.dlc.laser1.dl.pc.external_input.enabled.get()
    
    def ext_signal(self, val: int=None):
        if type(val) == int:
            self._laser.dlc.laser1.dl.pc.external_input.signal.set(val)
        else:
            return External_Signal_Channels.LABEL[self._laser.dlc.laser1.dl.pc.external_input.signal.get()]
    
    def ext_factor(self, val: float=None):
        if type(val) == float:
            self._laser.dlc.laser1.dl.pc.external_input.factor.set(val)
        else:
            return self._laser.dlc.laser1.dl.pc.external_input.factor.get()
    
    
    # Motor controll properties ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

    def remote_enabled(self, val: bool=None):
        if type(val) == bool:
            self._laser.dlc.laser1.ctl.remote_control.enabled.set(val)
        else:
            return self._laser.dlc.laser1.ctl.remote_control.enabled.get()
    
    def remote_signal(self, val: int=None):
        if type(val) == int:
            self._laser.dlc.laser1.ctl.remote_control.signal.set(val)
        else:
            return External_Signal_Channels.LABEL[self._laser.dlc.laser1.ctl.remote_control.signal.get()]
    
    def remote_factor(self, val: float=None):
        if type(val) == float or type(val) == int:
            self._laser.dlc.laser1.ctl.remote_control.factor.set(val)
        else:
            return self._laser.dlc.laser1.ctl.remote_control.factor.get()
        



class DLCProLogic(GenericLogic,CTLInterface):
    
    ''' Config Example
    ctllogic:
            module.Class: 'TopticaDLCLogic.DLCProLogic'
            connect:
                TopticaDLCPro: 'TopticaDLCPro'
    '''
    # Implement Config options for voltage_offset and voltage_to_power_ratio
    TopticaDLCPro = Connector(interface='TopticaDLCPro')

    def on_activate(self):
        self._laser:TopticaDLCPro = self.TopticaDLCPro()
        self._laser.dlc.open()

    def on_deactivate(self):
        self._laser.on_deactivate()

    def emission(self):
        return self._laser.emission

    def wavelength(self,val: float=None):
        if val:
            print("Setting CTL wavelength to ", val,"nm.")
            self._laser.wavelength_setpoint = val
        else:
            return self._laser.wavelength_setpoint

    def current(self,val: float=None):
        if val:
            self._laser.current_setpoint = val
        else:
            return self._laser.current_setpoint

    def CCemission(self,val: bool=None):
        if val == True:
            self._laser.current_enabled = val
        elif val == False:
            self._laser.current_enabled = val
        else:
            return self._laser.current_enabled

    def ON(self):
        self._laser.current_enabled = True

    def OFF(self):
        self._laser.current_enabled = False

    # Piezo controll properties ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

        
    def ext_enabled(self, val: bool=None):
        if type(val) == bool:
            self._laser.dlc.laser1.dl.pc.external_input.enabled.set(val)
        else:
            return self._laser.dlc.laser1.dl.pc.external_input.enabled.get()
    
    def ext_signal(self, val: int=None):
        if type(val) == int:
            self._laser.dlc.laser1.dl.pc.external_input.signal.set(val)
        else:
            return External_Signal_Channels.LABEL[self._laser.dlc.laser1.dl.pc.external_input.signal.get()]
    
    def ext_factor(self, val: float=None):
        if type(val) == float or type(val) == int:
            self._laser.dlc.laser1.dl.pc.external_input.factor.set(val)
        else:
            return self._laser.dlc.laser1.dl.pc.external_input.factor.get()
    
    

        