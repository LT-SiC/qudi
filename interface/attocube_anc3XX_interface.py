# -*- coding: utf-8 -*-
"""
Interface for a spectrometer.

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

from core.interface import abstract_interface_method
from core.meta import InterfaceMetaclass


class AttocubeANC3XXInterface(metaclass=InterfaceMetaclass):
    def on_activate(self):
        pass
    
    def on_deactivate(self):
        pass
   
    def enable_axis(self,axis=None):
        pass

    def disable_axis(self,axis=None):
        pass

    def measure_cap(self,axis=None):
        pass

    def get_cap(self,axis=None):
        pass

    def perform_step(self,axis=None,steps=1):
        pass

    def move(self,axis=None,direction:str="+"):
        pass

    def stop(self,axis=None):
        pass

    def voltage(self,axis=None,value=None):
        pass

    def frequency(self,axis=None,value=None):  
        pass