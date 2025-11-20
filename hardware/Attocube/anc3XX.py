from types import new_class
import numpy as np
from datetime import datetime
import time

from core.configoption import ConfigOption
from core.module import Base

from PyQt5 import QtTest

from interface.attocube_anc3XX_interface import AttocubeANC3XXInterface
from pylablib.devices import Attocube

AXIS = {1:6,2:5,3:7,4:0,5:1,6:2} # Sx,Sy,Sz,Fx,Fy,Fz


# quick and dirty done by Timo S. for positioner control

class ANC300(Base, AttocubeANC3XXInterface): #Hardware file for sample positioners controlled with ANC300
   
    ''' Config Example
    anc300:
            module.Class: 'Attocube.anc3XX.ANC300'
            COM: 'COM5'
    '''
    COM = ConfigOption('COM', False, missing='warn') #test if readout from config file works


    def on_activate(self):
        # Parameters
        print("Init ANC300 device...")
        self.anc = Attocube.anc300.ANC300(self.COM)

        self._is_connected = True
        print("Available axis are: ",self.anc.get_all_axes())
        print("The axes are already enabled: ",self.anc.is_enabled())
        self.stop()
        return

    def on_deactivate(self):
        try: #please look away :)
            self.disable_axis()
            self.anc.close()
        except:
            pass

    def enable_axis(self,axis=None):
        self.anc.enable_axis(axis)

    def disable_axis(self,axis=None):
        self.anc.disable_axis(axis)

    def measure_cap(self,axis=None):
        try:
            self.anc.measure_capacitance(axis)
            return self.get_cap(axis)
        except Attocube.AttocubeError as e:
            print(e)
            return np.NaN

    def get_cap(self,axis=None):
        self.C = self.anc.get_capacitance(axis)
        return self.C

    def perform_step(self,axis=None,steps=1):
        self.anc.move_by(axis,steps)

    def move(self,axis=None,direction:str="+"):
        self.anc.jog(axis,direction)

    def stop(self,axis=None):
        self.anc.stop(axis)

    def voltage(self,axis=None,value=None):
        if value:
            self.anc.set_voltage(axis,value)
        else:
            return self.anc.get_voltage(axis)
        
    def frequency(self,axis=None,value=None):
        if value:
            self.anc.set_frequency(axis,value)
        else:
            return self.anc.get_frequency(axis)
        
    def toogle_ext_input(self,ch=7,ac=None,dc=None):
        self.anc.set_external_input_modes(ch,acin=ac,dcin=dc)
        print("Input modes for channel {0} are AC: {1} & DC: {2}".format(ch,*self.anc.get_external_input_modes(ch)))



class ANC350(Base, AttocubeANC3XXInterface): #Hardware file for fiber positioners controlled with ANC350
   
    ''' Config Example
    anc350:
            module.Class: 'Attocube.anc3XX.ANC350'
    '''
    # COM = ConfigOption('COM', False, missing='warn') #test if readout from config file works


    def on_activate(self):
        # Parameters
        print("Find ANC350 devices...",Attocube.get_usb_devices_number_ANC350())
        try:
            self.anc = Attocube.anc350.ANC350()
        except:
            self.anc = None
            self._is_connected = False
            print("Connection of ANC350 failed...!")
            print("Please start the Daisy.exe softare once and close it directly afterwards. Then restart hardware module!")
            return 0

        self._is_connected = True
        print("Available axis are: ",self.anc.get_all_axes())
        print("The axes are already enabled: ",self.anc.is_enabled())
        self.stop()
        return

    def on_deactivate(self):
        try: #please look away :)
            self.disable_axis()
            self.anc.close()
        except:
            pass

    def enable_axis(self,axis=None):
        self.anc.enable_axis(axis)

    def disable_axis(self,axis=None):
        self.anc.disable_axis(axis)

    def measure_cap(self,axis=None):
        try:
            self.C = self.anc.get_capacitance(axis,measure=True)
            return self.C
        except Attocube.AttocubeError as e:
            print(e)
            return np.NaN

    def get_cap(self,axis=None):
        self.C = self.anc.get_capacitance(axis)
        return self.C

    def perform_step(self,axis=None,steps=1):
        self.anc.move_by_steps(axis,steps)

    def move(self,axis=None,direction:str="+"):
        self.anc.jog(axis,direction)

    def stop(self,axis=None):
        self.anc.stop(axis)

    def voltage(self,axis=None,value=None):
        if value:
            self.anc.set_voltage(axis,value)
        else:
            return self.anc.get_voltage(axis)
        
    def frequency(self,axis=None,value=None):
        if value:
            self.anc.set_frequency(axis,value)
        else:
            return self.anc.get_frequency(axis)


    # a.is_moving()
