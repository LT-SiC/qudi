import numpy as np
from datetime import datetime
import time

from core.configoption import ConfigOption
from core.module import Base

import nidaqmx
from nidaqmx.stream_readers import AnalogMultiChannelReader
from nidaqmx import constants

from PyQt5 import QtTest

from interface.stream_usb_nidaq_interface import StreamUSBNidaqInterface

# quick and dirty done by Timo S. for Imaging Control with unused NI card

class streamPCInidaq(Base, StreamUSBNidaqInterface): #Hardware file
   
    ''' Config Example
    streamusbnidaq:
            module.Class: 'PCINidaq6229.streamPCInidaq'
            chan_out: 'Dev4/ao0'
            chan_ao_aom: 'Dev4/ao1'
            chan_do_flipper_1: 'Dev4/port0/line16'
            chan_do_flipper_2: 'Dev4/port0/line18'
            chan_do_flipper_3: 'Dev4/port0/line23'
            chan_do_flipper_4: 'Dev4/port0/line28'
            chan_do_flipper_5: 'Dev4/port0/line29'
            voltagerange: [0,5]
    '''
    chan_out = ConfigOption('chan_out', False, missing='warn') 
    chan_ao_aom = ConfigOption('chan_ao_aom', False, missing='warn') 
    chan_do_flipper_1 = ConfigOption('chan_do_flipper_1', False, missing='warn') 
    chan_do_flipper_2 = ConfigOption('chan_do_flipper_2', False, missing='warn') 
    chan_do_flipper_3 = ConfigOption('chan_do_flipper_3', False, missing='warn') 
    chan_do_flipper_4 = ConfigOption('chan_do_flipper_4', False, missing='warn') 
    chan_do_flipper_5 = ConfigOption('chan_do_flipper_5', False, missing='warn') 
    voltagerange = ConfigOption('voltagerange', False, missing='warn')
    _buffer_in_size = ConfigOption('buffer_size', 100, missing='warn')
    _sampling_freq_in = ConfigOption('sampling_frequency', 42, missing='warn')


    def on_activate(self):
        # Parameters
        print("Init PCI Nidaq...")
        self.sampling_freq_in = int(self._sampling_freq_in)  # in Hz
        self.buffer_in_size = int(self._buffer_in_size)
        
        self.bufsize_callback = self.buffer_in_size
        self.buffer_in_size_cfg = round(self.buffer_in_size * 1)  # clock configuration
        self.crop = 10  # number of seconds to drop at acquisition start before saving #What does this do?
        self.my_filename = 'test_3_opms'  # with full path if target folder different from current folder (do not leave trailing /)
        
        # Initialize data placeholders
        self.buffer_in = np.zeros((1, self.buffer_in_size))
        self.data = np.zeros((1, 1))  # will contain a first column with zeros but that's fine
        #self.start_acquisition()
        self.task_out = nidaqmx.Task()
        self.task_out.ao_channels.add_ao_voltage_chan(self.chan_out,min_val=self.voltagerange[0], max_val=self.voltagerange[1])
        self.task_ao_aom = nidaqmx.Task()
        self.task_ao_aom.ao_channels.add_ao_voltage_chan(self.chan_ao_aom,min_val=-10.0, max_val=10.0)
        self.task_do_flipper = nidaqmx.Task()
        self.task_do_flipper.do_channels.add_do_chan(self.chan_do_flipper_1,line_grouping=nidaqmx.constants.LineGrouping.CHAN_FOR_ALL_LINES)
        self.task_do_flipper.do_channels.add_do_chan(self.chan_do_flipper_2,line_grouping=nidaqmx.constants.LineGrouping.CHAN_FOR_ALL_LINES)
        self.task_do_flipper.do_channels.add_do_chan(self.chan_do_flipper_3,line_grouping=nidaqmx.constants.LineGrouping.CHAN_FOR_ALL_LINES)
        self.task_do_flipper.do_channels.add_do_chan(self.chan_do_flipper_4,line_grouping=nidaqmx.constants.LineGrouping.CHAN_FOR_ALL_LINES)
        self.task_do_flipper.do_channels.add_do_chan(self.chan_do_flipper_5,line_grouping=nidaqmx.constants.LineGrouping.CHAN_FOR_ALL_LINES)

        return

    def on_deactivate(self):
        try: #please look away :)
            # self.task_in.is_task_done()
            self.shut_down_streaming()
        except:
            pass

       

    # def start_ao_task(self):
    #     self.task_out = nidaqmx.Task()
    #     self.cfg_write_task(self.task_out)
    #     self.task_out.start()

    # def cfg_read_task(self, acquisition):
    #     acquisition.ai_channels.add_ai_voltage_chan(self.chan_in)
    #     acquisition.timing.cfg_samp_clk_timing(rate=self.sampling_freq_in, sample_mode=constants.AcquisitionType.CONTINUOUS,
    #                                         samps_per_chan=self.buffer_in_size_cfg)
    #     # print("Connected read task.")
    
    # def cfg_write_task(self, voltage_out_task):
    #     self.numberOfUsedChannels = 0
    #     print("Channels:")
    #     print(self.chan_out)
    #     if self.chan_out != 'None':
    #         print("1")
    #         voltage_out_task.ao_channels.add_ao_voltage_chan(self.chan_A2, min_val=self.voltagerange[0], max_val=self.voltagerange[-1])
    #         self.numberOfUsedChannels +=1      
    #     print("Connected write task.")

    # def reading_task_callback(self, task_idx, event_type, num_samples, callback_data):
    #     #print("buffer in", self.buffer_in)
    #     #print("num samples", num_samples)
    #     self.stream_in.read_many_sample(self.buffer_in, num_samples, timeout=constants.WAIT_INFINITELY)

    #     return 0  # Absolutely needed for this callback to be well defined (see nidaqmx doc).
    
    def goToVoltage(self, V):
        if V > self.voltagerange[-1] or V < self.voltagerange[0]:
            print("Target voltage does not in voltage range.")
        else:
            print("Setting Outputvoltage to", V, "V.")
            self.task_out.write(V,auto_start=True)
            QtTest.QTest.qSleep(10) # needed at all?
            self.current_voltage = V

    def goToAOMVoltage(self, V):
        if V > 10 or V < -10:
            print("Target voltage is not in voltage range [-10,10].")
        else:
            print("Setting AOM Outputvoltage to", V, "V.")
            self.task_ao_aom.write(V,auto_start=True)
            QtTest.QTest.qSleep(10) # needed at all?
            self.current_aom_voltage = V
    

    def shut_down_streaming(self):
        # Close task to clear connection
        # self.goToVoltage(0)
        # self.task_in.close()
        self.task_out.close()
        self.task_do_flipper.close()

    def flip(self, pos:list):
        self.task_do_flipper.write(pos,auto_start=True)
        QtTest.QTest.qSleep(10) # needed at all?
        print("Flip mirros moved")