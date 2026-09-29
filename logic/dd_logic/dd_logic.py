#DD-Logic for mcas-module from Javid which combines AWG and ps and uses AWG as master.

import numpy as np
import sys
sys.path.append('C:\src\qudi\hardware\Keysight_AWG_M8190\pyarbtools_master') #quickfix to proceed, should be improved

from core.module import Base
from core.connector import Connector
#from hardware.swabian_instruments.timetagger import TT as TimeTagger
from logic.generic_logic import GenericLogic
#import hardware.Keysight_AWG_M8190.pyarbtools_master.pyarbtools as pyarbtools

from logic.dd_logic.dd_default_values_and_widget_functions import dd_default_values_and_widget_functions as dd_default

import os

from qtpy import QtCore
from PyQt5 import QtTest

import re
import inspect
import logging
logger = logging.getLogger(__name__)
import time
#import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt
import datetime
import hardware.Keysight_AWG_M8190.elements as E
import notebooks.UserScripts.helpers.snippets_awg as sna
import notebooks.UserScripts.helpers.shared as shared
from core.statusvariable import StatusVar

from logic.qudip_enhanced import sc

class DDLogic(GenericLogic,dd_default):
    #declare connectors
    counter_device = Connector(interface='TimeTaggerInterface')# Savelogic just for testing purposes
    savelogic = Connector(interface='SaveLogic')
    mcas_holder = Connector(interface='McasDictHolderInterface')
    fitlogic = Connector(interface='FitLogic')
    transition_tracker = Connector(interface = 'TransitionTracker') # Should be a name of the class
    confocal = Connector('ConfocalLogic')
    optimizer= Connector('OptimizerLogic')
    PLE_logic= Connector("LaserScannerLogic")
    #transition_tracker = Connector(interface="TransitionTracker")

    CHANNEL_APD0 = 0
    CHANNEL_APD1 = 1
    CHANNEL_DETECT = 2
    CHANNEL_SEQUENCE = 3
    

    # time_tagger = counter_device().createTimeTagger()

    # print('time_tagger created')
    # print(time_tagger)
    # time_tagger.setTriggerLevel(0, 1)  #Supra
    # time_tagger.setTriggerLevel(1, 1)  #Supra
    # time_tagger.setTriggerLevel(2, 1)
    # time_tagger.setTriggerLevel(3, 1)
    # time_tagger.setTriggerLevel(4, 1)
    # time_tagger.setTriggerLevel(5, 1)
    # time_tagger.setTriggerLevel(6, 1)
    # time_tagger.setTriggerLevel(7, 1)

    #create the signals:
    sigDDPlotsUpdated = QtCore.Signal()
    SigClock= QtCore.Signal()
    SigCheckReady_Beacon = QtCore.Signal()
    sigFitPerformed= QtCore.Signal(np.float)
    sigUnits=QtCore.Signal(int)

    starting_time=0
    cooldown_loops=2

    T=True
    F=False
    dd_Filename = StatusVar("dd_Filename", "Filename")
    dd_Tau_Min = StatusVar("dd_Tau_Min", 1000)
    dd_Tau_Max = StatusVar("dd_Tau_Max", 1500)
    dd_Tau_Step = StatusVar("dd_Tau_Step", 2)
    dd_Tau_Decay = StatusVar("dd_Tau_Decay",  500)
    dd_MW1 = StatusVar("dd_MW1", T)
    dd_MW2 = StatusVar("dd_MW2", T)
    dd_MW3 = StatusVar("dd_MW3", F)
    dd_MW4 = StatusVar("dd_MW4", F)
    dd_MW5 = StatusVar("dd_MW5", F)
    dd_MW6 = StatusVar("dd_MW6", F)
    dd_MW1_Amp = StatusVar("dd_MW1_Amp", 0.29796)
    dd_MW2_Amp = StatusVar("dd_MW2_Amp", 0.33477)
    dd_MW3_Amp = StatusVar("dd_MW3_Amp", 0.21976)
    dd_MW4_Amp = StatusVar("dd_MW4_Amp", 0.19739)
    dd_MW5_Amp = StatusVar("dd_MW5_Amp", 0.14306)
    dd_MW6_Amp = StatusVar("dd_MW6_Amp", 0.12047)
    dd_readout_power = StatusVar("dd_readout_power", 0.2)
    dd_MW1_Pi2_Amp = StatusVar("dd_MW1_Pi2_Amp", 0.14988)
    dd_MW2_Pi2_Amp = StatusVar("dd_MW2_Pi2_Amp", 0.17488)
    dd_MW3_Pi2_Amp = StatusVar("dd_MW3_Pi2_Amp", 0.10779)
    dd_MW4_Pi2_Amp = StatusVar("dd_MW4_Pi2_Amp", 0.09303)
    dd_MW5_Pi2_Amp = StatusVar("dd_MW5_Pi2_Amp", 0.06336)
    dd_MW6_Pi2_Amp = StatusVar("dd_MW6_Pi2_Amp", 0.05009)
    dd_MW1_Freq = StatusVar("dd_MW1_Freq", 5000.)
    dd_MW2_Freq = StatusVar("dd_MW2_Freq", 5000+5.)
    dd_MW3_Freq = StatusVar("dd_MW3_Freq", 5000+70.)
    dd_MW4_Freq = StatusVar("dd_MW4_Freq", 5000+75.)
    dd_MW5_Freq = StatusVar("dd_MW5_Freq", 5000+140.)
    dd_MW6_Freq = StatusVar("dd_MW6_Freq", 5000+145.)
    dd_Pi_dur = StatusVar("dd_Pi_dur", 1.)
    dd_Pi2_dur = StatusVar("dd_Pi2_dur", 1.)
    dd_DD_type = StatusVar("dd_DD_type", "hahn")
    dd_nrep = StatusVar("dd_nrep", 2)
    dd_Init_State = StatusVar("dd_Init_State", "-3/2")
    dd_enhanced_readout = StatusVar("dd_enhanced_readout", T)
    dd_A1 = StatusVar("dd_A1", F)
    dd_A2 = StatusVar("dd_A2", T)
    dd_A1Readout = StatusVar("dd_A1Readout", F)
    dd_A2Readout = StatusVar("dd_A2Readout", T)
    dd_InitTime = StatusVar("dd_InitTime",  30)
    dd_DecayInit = StatusVar("dd_DecayInit",  1)
    dd_RepumpDecay = StatusVar("dd_RepumpDecay", 1)
    dd_CWRepump = StatusVar("dd_CWRepump", F)
    dd_PulsedRepump = StatusVar("dd_PulsedRepump", T)
    dd_RepumpDuration = StatusVar("dd_RepumpDuration", 5)
    dd_AOMDelay = StatusVar("dd_AOMDelay", 0)
    dd_IntegrationTime = StatusVar("dd_IntegrationTime",  20000)
    dd_Binning = StatusVar("dd_Binning", 5000)
    dd_Interval = StatusVar("dd_Interval", 0)
    dd_PeriodicSaving = StatusVar("dd_PeriodicSaving", F)
    dd_Stoptime = StatusVar("dd_Stoptime", 0)
    dd_ReadoutTime = StatusVar("dd_ReadoutTime", 200000)
    dd_ReadoutDecay = StatusVar("dd_ReadoutDecay", 500)
    dd_MaxIterations = StatusVar("dd_MaxIterations", 20)
    dd_PerformFit = StatusVar("dd_PerformFit", F)
    dd_NumberOfPeaks = StatusVar("dd_NumberOfPeaks", 0)
    dd_cb_max = StatusVar("dd_cb_max", 100)
    dd_cb_high_percentile = StatusVar("dd_cb_high_percentile", 100)
    dd_cb_low_percentile = StatusVar("dd_cb_low_percentile", 0)
    dd_cb_min = StatusVar("dd_cb_min", 0)
    dd_Runtime = StatusVar("dd_Runtime", 0)
    dd_Frequencies_Fit = StatusVar("dd_Frequencies_Fit", 70)
    dd_Linewidths_Fit = StatusVar("dd_Linewidths_Fit", 1)
    dd_FitFunction = StatusVar("dd_FitFunction", 'Cosinus')
    dd_FitParams = StatusVar("dd_FitParams", "")
    update_after_stop = StatusVar("update_after_stop", F)
    pi_pulse = StatusVar("pi_pulse", 100)
    dd_unit = StatusVar("dd_unit", False)
    first_dd_last_pi2_phase= StatusVar("first_dd_last_pi2_phase",0)
    second_dd_last_pi2_phase= StatusVar("second_dd_last_pi2_phase",round(np.pi/2,4))
    dd_waveform=StatusVar("dd_waveform","sinehermite")
    round_to=1
    

    def on_activate(self):
        #super().__init__() #maybe this will make it possible to reload the default val and widget functions
        self._time_tagger=self.counter_device()
        self._time_tagger.setup_TT()
        self._save_logic = self.savelogic()
        self._awg = self.mcas_holder()#mcas_dict()
        self._fit_logic = self.fitlogic()
        self._tt=self.transition_tracker()
        self._optimizer = self.optimizer()
        self._PLE_logic = self.PLE_logic()
        self._confocal = self.confocal()
        
        self.start_pause_time = 2.75 
        #self.end_pause_time = 5.25#6.1 
        self.end_pause_time = 6.01 
        self.cter=0

        self.trans={}
        for t in ['L', 'C', 'R']:
            for i in [1, 2]:
                self.trans[t + str(i)] = self._tt.mfl(td={self._tt.si29_list[0]: [-0.5, 0.5]}, ms_trans=t)[i - 1]

        self.dd_MW1_Freq=round(self.trans["L1"],6)
        self.dd_MW2_Freq=round(self.trans["L2"],6)
        self.dd_MW3_Freq=round(self.trans["C1"],6)
        self.dd_MW4_Freq=round(self.trans["C2"],6)
        self.dd_MW5_Freq=round(self.trans["R1"],6)
        self.dd_MW6_Freq=round(self.trans["R2"],6)
        
        self.duration=self.dd_Tau_Min
        self.parameter_to_sweep="duration"
        self.wait=E.round_length_mus_to_x_multiple_ps(0,self.round_to)

        self.sweep = np.unique(E.round_length_mus_to_x_multiple_ps(
            np.arange(self.dd_Tau_Min, self.dd_Tau_Max + self.dd_Tau_Step, self.dd_Tau_Step) / 1000,self.round_to )) # in µs
        
        
        self.stop_awg = self._awg.mcas_dict.stop_awgs
        self.Timer = RepeatedTimer(1, self.clock) # this clock is not very precise, maybe the solution proposed on https://stackoverflow.com/questions/474528/what-is-the-best-way-to-repeatedly-execute-a-function-every-x-seconds can be helpful.
        #self.SigCheckReady_Beacon.connect(self.print_counter)
        self.CheckReady_Beacon = RepeatedTimer(1, self.get_data)
        self.periodic_refocus_Timer=RepeatedTimer(300,self.periodic_refocus)

        #self.CheckReady_Beacon.start()
        self.number_of_points_per_line=self._time_tagger._time_diff["n_histograms"]
        self.measurement_running=False
        self.counter=self._time_tagger.counter()
        self.time_differences = self._time_tagger.time_differences()
        self.scanmatrix=np.zeros(np.array(self.time_differences.getData(),dtype=object).shape)
        self.scanmatrix_data=np.zeros(np.array(self.time_differences.getData(),dtype=object).shape)
        self.fit_data = np.zeros_like(self.scanmatrix)[0]
        self.interpolated_x_data = np.linspace(0,1,len(self.fit_data))
        self.Amplitude_Fit:str=''
        self.Frequency_Fit:str=''
        self.Phase_Fit:str=''
        self.tau_pulse:float=0 #ns
        self.SigCheckReady_Beacon.connect(self.get_data)
   
        self.syncing=False

        self.continuing=False
        return 

    def on_deactivate(self):
        self.Timer.stop()
        self.CheckReady_Beacon.stop()
        self.stop_awg()
        try: #checkready_beacon may not be launched
            self.checkready.stop()
        except:
            pass
        
        return 

    def set_new_refocus_interval(self,time):
        self.periodic_refocus_Timer.stop()
        self.periodic_refocus_Timer=RepeatedTimer(time,self.periodic_refocus)
    
    def get_data(self):
        if time.time()-self.starting_time>self.dd_Stoptime and self.dd_Stoptime!=0:
            self.dd_Stop_Button_Clicked(True)

        #print("checkready:",self.measurement_running)
        if not(self.measurement_running):
                return
            
        else:
            indexes=np.array(self.time_differences.getIndex()) #readout binwidth (ps)
            self.new_data=np.array(self.time_differences.getData(),dtype=object) #readout data from timetagger
            self.scanmatrix=self.scanmatrix_data+self.new_data
            self.measured_times_ns=indexes/1e3 #indexes is in ps
            mask=((self.measured_times_ns>=self.dd_AOMDelay) & (self.measured_times_ns<=self.dd_IntegrationTime+self.dd_AOMDelay)) #create mask to filter counts depending on arrival time
            self.data=np.sum(self.scanmatrix[:,mask],axis=1) #sum up the readout-histogram after a single Tau
            self.data_detect=np.sum(self.scanmatrix,axis=0) #sum up the histograms to see the emission decay
            self.measured_times=indexes/1e12 #binwidth in seconds
            self.sigDDPlotsUpdated.emit()


    def clock(self):
        self.SigClock.emit()

    def CheckReady(self):
        self.SigCheckReady_Beacon.emit()

    def setup_time_tagger(self,**kwargs):
        self._time_tagger._time_diff.update(**kwargs)
        return self._time_tagger.time_differences()

    def save_dd_data(self, tag=None, colorscale_range=None, percentile_range=None):
        """ Saves the current DD data to a file."""
        timestamp = datetime.datetime.now()
        filepath = self._save_logic.get_path_for_module(module_name='DD')
        tag = self.dd_Filename
        if tag is None:
            tag = ''

        if len(tag) > 0:
            filelabel_raw = '{0}_DD_raw'.format(tag)
            filelabel_detection = '{0}_DD_detection'.format(tag)
            filelabel_matrix = '{0}_DD_matrix'.format(tag)
            filelabl_awg_file= '{0}_DD_awg_file'.format(tag)
        else:
            filelabel_raw = '_DD_raw'
            filelabel_detection = '_DD_detection'
            filelabel_matrix = '_DD_matrix'
            filelabl_awg_file= '_DD_awg_file'

        data_raw = OrderedDict()
        data_detection = OrderedDict()
        data_matrix = OrderedDict()
        data_raw['count data (counts)'] = self.data[::2]
        data_raw['count data 2 (counts)'] = self.data[1::2]
        data_raw['count data difference (counts)'] = self.data[::2]-self.data[1::2]
        data_raw['Tau (ns)'] = self.sweep
        data_detection['Detection Time (ns)'] = self.measured_times*1e9 # save in [ns]
        data_detection['Detection Counts (counts)'] = self.data_detect
        data_matrix['Detection Time + Tau'] = self.scanmatrix_data
        parameters = OrderedDict()


        
        parameters['Runtime)'] = time.time()-self.starting_time # TODO This does not work with continue measurement
        parameters['Enable Microwave1 (bool)'] = self.dd_MW1
        parameters['Enable Microwave2 (bool)'] = self.dd_MW2
        parameters['Enable Microwave3 (bool)'] = self.dd_MW3
        parameters['Microwave1 CW Power (dBm)'] = self.dd_MW1_Amp
        parameters['Microwave2 CW Power (dBm)'] = self.dd_MW2_Amp
        parameters['Microwave3 CW Power (dBm)'] = self.dd_MW3_Amp
        parameters['Microwave1 CW Power (dBm)'] = self.dd_MW1_Freq
        parameters['Microwave2 CW Power (dBm)'] = self.dd_MW2_Freq
        parameters['Microwave3 CW Power (dBm)'] = self.dd_MW3_Freq
        parameters['Tau min (ns)'] = self.dd_Tau_Min
        parameters['Tau max (ns)'] = self.dd_Tau_Max
        parameters['Tau Step (ns)'] = self.dd_Tau_Step
        parameters['Tau Decay (ns)'] = self.dd_Tau_Decay
        parameters['A1 (bool)'] = self.dd_A1
        parameters['A2 (bool)'] = self.dd_A2
        parameters['Pulsed Repump (bool)'] = self.dd_PulsedRepump
        parameters['Pulsed Duration (µs)'] = self.dd_RepumpDuration
        parameters['Pulsed Decay (µs)'] = self.dd_RepumpDecay
        parameters['CW Repump (bool)'] = self.dd_CWRepump
        parameters['Init Time (µs)'] = self.dd_InitTime
        parameters['Init Decay (µs)'] = self.dd_DecayInit
        parameters['Readout Time (µs)'] = self.dd_ReadoutTime
        parameters['Readout Decay (µs)'] = self.dd_ReadoutDecay
        parameters['Readout via A1 (bool)'] = self.dd_A1Readout
        parameters['Readout via A2 (bool)'] = self.dd_A2Readout
        parameters['AOM Delay (ns)'] = self.dd_AOMDelay
        parameters['Integration Window (ns)'] = self.dd_IntegrationTime
        parameters['Binning (ns)'] = self.dd_Binning
        parameters['Amplitude Fit'] = self.Amplitude_Fit
        parameters['Frequency Fit'] = self.Frequency_Fit
        parameters['Phase Fit'] = self.Phase_Fit
        parameters['Pi pulse']= self.pi_pulse

        fig = self.draw_figure(
            data_raw['Tau (ns)'],
            data_raw['count data (counts)'],
            data_raw['count data 2 (counts)'],
            data_raw['count data difference (counts)'],
            data_matrix['Detection Time + Tau'],
            data_detection['Detection Time (ns)'],
            data_detection['Detection Counts (counts)'],
            self.interpolated_x_data,
            self.fit_data,
            cbar_range=colorscale_range,
            percentile_range=percentile_range
        )


        self._save_logic.save_data(data_matrix,
                                    filepath=filepath,
                                    parameters=parameters,
                                    filelabel=filelabel_matrix,
                                    fmt='%.6e',
                                    delimiter='\t',
                                    timestamp=timestamp)
        
        self._save_logic.save_data(data_detection,
                                    filepath=filepath,
                                    parameters=parameters,
                                    filelabel=filelabel_detection,
                                    fmt='%.6e',
                                    delimiter='\t',
                                    timestamp=timestamp)
        
        self._save_logic.save_data(data_raw,
                                    filepath=filepath,
                                    parameters=parameters,
                                    filelabel=filelabel_raw,
                                    fmt='%.6e',
                                    delimiter='\t',
                                    timestamp=timestamp,
                                    plotfig=fig)
        

        awg_fp = os.path.join(filepath, filelabl_awg_file)
        seq_message=self._awg.mcas_dict["DD"].sequences["2g"][1].ret_info()
        
        with open(awg_fp+".txt", 'w') as fp:
            for page in seq_message:
                fp.write(page)
            fp.write('\n-------------------------------------------------------------------\n')


        self.log.info('DD data saved to:\n{0}'.format(filepath))
        return

    def draw_figure(self, time_data, count_data,count_data2,count_data_diff, matrix_data, detection_time, detection_counts, fit_freq_vals, fit_count_vals, cbar_range=None, percentile_range=None):
        """ Draw the summary figure to save with the data.

        @param: list cbar_range: (optional) [color_scale_min, color_scale_max].
                                 If not supplied then a default of data_min to data_max
                                 will be used.

        @param: list percentile_range: (optional) Percentile range of the chosen cbar_range.

        @return: fig fig: a matplotlib figure object to be saved to file.
        """
        #key = 'range: {1}'.format(frequencies)
        matrix_data=matrix_data.astype(float)
        # If no colorbar range was given, take full range of data
        if cbar_range is None:
            cbar_range = np.array([np.min(matrix_data), np.max(matrix_data)])
        else:
            cbar_range = np.array(cbar_range)

        prefix = ['', 'k', 'M', 'G', 'T']
        prefix_index = 0

        # Rescale counts data with SI prefix
        while np.max(count_data) > 1000:
            count_data = count_data / 1000
            #fit_count_vals = fit_count_vals / 1000
            prefix_index = prefix_index + 1

        counts_prefix = prefix[prefix_index]

        # Rescale frequency data with SI prefix
        prefix_index = 0

        while np.max(time_data) > 1000:
            time_data = time_data / 1000
            fit_freq_vals = fit_freq_vals / 1000
            prefix_index = prefix_index + 1

        mw_prefix = prefix[prefix_index]

        # Rescale matrix counts data with SI prefix
        prefix_index = 0

        while np.max(matrix_data) > 1000:
            matrix_data = matrix_data / 1000
            cbar_range = cbar_range / 1000
            prefix_index = prefix_index + 1

        cbar_prefix = prefix[prefix_index]

        # Use qudi style
        plt.style.use(self._save_logic.mpl_qd_style)

        # Create figure
        fig, (ax_mean, ax_matrix, ax_detection) = plt.subplots(nrows=3, ncols=1)

        ax_mean.plot(time_data, count_data, linestyle=':', linewidth=0.5,c="blue")
        ax_mean.plot(time_data, count_data2, linestyle=':', linewidth=0.5,c="orange")
        ax_mean.plot(time_data, count_data_diff, linestyle=':', linewidth=0.5,c="cyan")

        # Do not include fit curve if there is no fit calculated.
        if max(fit_count_vals) > 0:
            ax_mean.plot(fit_freq_vals, fit_count_vals, marker='None')
        ax_mean.set_ylabel('Fluorescence (' + counts_prefix + 'counts)')
        ax_mean.set_xlim(np.min(time_data), np.max(time_data))
        matrixplot = ax_matrix.imshow(
            matrix_data,
            cmap=plt.get_cmap('inferno'),  # reference the right place in qd
            origin='lower',
            vmin=cbar_range[0],
            vmax=cbar_range[1],
            extent=[np.min(time_data),
                    np.max(time_data),
                    0,
                    np.shape(matrix_data)[0]
                    ],
            aspect='auto',
            interpolation='nearest')

        ax_matrix.set_xlabel('Frequency (' + mw_prefix + 'Hz)')
        ax_matrix.set_ylabel('Scan #')

        ax_detection.plot(detection_time, detection_counts, linestyle=':', linewidth=0.5)
        ax_detection.set_ylabel('Fluorescence (' + counts_prefix + 'counts)')
        ax_detection.set_xlim(np.min(detection_time), np.max(detection_time))

        # Adjust subplots to make room for colorbar
        fig.subplots_adjust(right=0.8)

        # Add colorbar axis to figure
        cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])

        # Draw colorbar
        cbar = fig.colorbar(matrixplot, cax=cbar_ax)
        cbar.set_label('Fluorescence (' + cbar_prefix + 'c/s)')

        # remove ticks from colorbar for cleaner image
        cbar.ax.tick_params(which=u'both', length=0)

        # If we have percentile information, draw that to the figure
        if percentile_range is not None:
            cbar.ax.annotate(str(percentile_range[0]),
                             xy=(-0.3, 0.0),
                             xycoords='axes fraction',
                             horizontalalignment='right',
                             verticalalignment='center',
                             rotation=90
                             )
            cbar.ax.annotate(str(percentile_range[1]),
                             xy=(-0.3, 1.0),
                             xycoords='axes fraction',
                             horizontalalignment='right',
                             verticalalignment='center',
                             rotation=90
                             )
            cbar.ax.annotate('(percentile)',
                             xy=(-0.3, 0.5),
                             xycoords='axes fraction',
                             horizontalalignment='right',
                             verticalalignment='center',
                             rotation=90
                             )

        return fig

    def power_to_amp(self, power_dBm, impedance=50):
        power_dBm = np.atleast_1d(power_dBm)
        P_watts = 10**(power_dBm / 10) * 1e-3
        V_rms = np.sqrt(P_watts * impedance)
        V_pp = V_rms * 2 * np.sqrt(2)
        return V_pp / self._awg.mcas_dict.awgs['2g'].ch[1].output_amplitude
        #return V_pp / float(self.awg_device.amp1) #awg_amplitude

    def amp_to_power(self, power_amp, impedance=50):
        power_amp = np.atleast_1d(power_amp)
        V_pp= power_amp* self._awg.mcas_dict.awgs['2g'].ch[1].output_amplitude
        V_rms=V_pp/(2 * np.sqrt(2))
        P_watts=V_rms**2/impedance
        power_dBm=np.log10(P_watts/1e-3)*10
        return power_dBm

    
    def setup_seq(
            self,
            dd_Tau_Min=None,
            dd_Tau_Max=None,
            dd_Tau_Step=None,
            dd_Tau_Decay=None,
    
            dd_MW1=None,
            dd_MW1_Freq=None,
            dd_MW1_Amp=None,
            dd_MW2=None,
            dd_MW2_Freq=None,
            dd_MW2_Amp=None,
            dd_MW3=None,
            dd_MW3_Freq=None,
            dd_MW3_Amp=None,
            dd_MW4=None,
            dd_MW4_Freq=None,
            dd_MW4_Amp=None,
            dd_MW5=None,
            dd_MW5_Freq=None,
            dd_MW5_Amp=None,
    
            dd_A1=None,
            dd_A2=None,
            dd_A1Readout=None,
            dd_A2Readout=None,
            dd_InitTime=None,
            dd_DecayInit=None,
            dd_RepumpDecay=None,
            dd_CWRepump=None,
            dd_PulsedRepump=None,
            dd_RepumpDuration=None,
            dd_AOMDelay=None,
            dd_IntegrationTime=None,
            dd_Binning=None,
            dd_Interval=None,
            dd_PeriodicSaving=None,
            dd_Stoptime=None,
    
            dd_ReadoutTime=None,
            dd_ReadoutDecay=None
    ):
        

        def erabi(freq, length, amp, phase=0.0):
                sna.electron_rabi(seq,
                                  name='electron rabi',
                                  length_mus=length,
                                  amplitudes=amp,
                                  frequencies=freq,
                                  phases=np.rad2deg(phase),
                                  new_segment=False,
                                  gateMW=True,
                                  mixer_deg=-90,
                                  pulseshape= self.dd_waveform
                                  )
        
        def waveform(sequence):
                #seq.start_new_segment('waveform')
                #sna.polarize(mcas, new_segment=False)
                mw = sequence.times_fields_aphi('mw')
                wait = sequence.times_fields_aphi('wait')
                for k, step in enumerate(sequence.sequence_steps):
                    idx = int(step[1]) - 1
                    if step[0] == 'mw':
                        if k == 0 or k == len(sequence.sequence_steps)-1:
                            erabi(freq=[*self.freqs],
                                length=self.dd_Pi2_dur,
                                amp=[*self.amps_pi2],
                                #amp=self.queue.tt.rp('e_rabi_ou350deg-90-R'+trans_idx, period=1 / mw[idx, 1]).amp, # TODO is trans_idx correct here?
                                phase=mw[idx, 2])
                        else:
                            erabi(freq=[*self.freqs],
                                length=self.dd_Pi_dur,
                                amp=[*self.amps],
                                #amp=self.queue.tt.rp('e_rabi_ou350deg-90-R'+trans_idx, period=1 / mw[idx, 1]).amp, # TODO is trans_idx correct here?
                                phase=mw[idx, 2])
                    elif step[0] == 'wait':
                        if wait[idx, 0] > 0.256:
                            seq.asc(length_mus=E.round_length_mus_to_x_multiple_ps(wait[idx, 0]-0.256,self.round_to))
                            seq.asc(length_mus=E.round_length_mus_to_x_multiple_ps(0.256,self.round_to), gateMW = True)
                        else:
                            seq.asc(length_mus=E.round_length_mus_to_x_multiple_ps(wait[idx, 0],self.round_to), gateMW = True)

        
        def dd(tau,last_phase=0):
                rabi_period = self.dd_Pi_dur * 2
                pi2x = sc.Rabi(t_rabi=0.25 * rabi_period, omega=1 / rabi_period, phase=0.0, control_field='mw')
                pi2_2 = sc.Rabi(t_rabi=0.25 * rabi_period, omega=1 / rabi_period, phase=last_phase, control_field='mw')
                
                match = re.search(r"\d+", self.dd_DD_type)
                number = int(match.group()) if match else 1
                dd = sc.DD(dd_type='{}_{}'.format(self.dd_nrep, self.dd_DD_type), rabi_period=rabi_period,
                          tau=tau)
                sequence = sc.Concatenated([pi2x, dd, pi2_2], controls=['mw', 'wait'])
                waveform(sequence)


        def long_ssr(mcas,**kwargs):
                mcas.asc(length_mus=sna.__TT_TRIGGER_LENGTH__, gate=True, name='gate1')  # Gated counter

                #print(self.kwargs,self.kwargs.keys(),print(self.frequencies))
                
                pd2g1 = {'frequencies': kwargs["frequencies"], 'type': self.dd_waveform, 'amplitudes': kwargs["amplitudes"]}
                #print("DURRATION::::::::::",self.dur_step[alt_step][5])

                mcas.asc(length_mus=E.round_length_mus_to_x_multiple_ps(kwargs["laser_dur"],self.round_to ),
                                A1=kwargs.get("A1",False), A2=kwargs.get("A2",False), laser = True, gateMW=True, pd2g1=pd2g1,
                                name='long_readout',gate=True)

        def readout():
            #seq.start_new_segment("Readout")

            if "-" in self.dd_Init_State:
                ssr_freqs=[self.dd_MW1_Freq,self.dd_MW2_Freq]
                amps=[self.dd_readout_power,self.dd_readout_power]
            else:
                ssr_freqs=[self.dd_MW5_Freq,self.dd_MW6_Freq]
                amps=[self.dd_readout_power,self.dd_readout_power]

            if self.dd_enhanced_readout:
                
                seq.asc(name="gateMW", gateMW=True, length_mus=E.round_length_mus_to_x_multiple_ps(gateMW_dur, self.round_to ))
                seq.asc(name="contrast_C_pulse",
                    pd2g1={"type": self.dd_waveform, "frequencies": [self.dd_MW3_Freq,self.dd_MW4_Freq],
                            "amplitudes":[self.dd_MW3_Amp,self.dd_MW4_Amp]},
                    length_mus=self.dd_Pi_dur,
                    gateMW=True)  # self.rabi_piPulseDuration is divided by 1000 to be in µs
            
                long_ssr(mcas=seq, queue=None,
                        frequencies=ssr_freqs,
                        laser_dur=E.round_length_mus_to_x_multiple_ps(self.dd_ReadoutTime/1000,self.round_to ),
                        amplitudes=amps,
                        A2=self.dd_A2Readout,
                        A1=self.dd_A1Readout,
                        gate=True,
                        )
            else:
                seq.asc(name='readout',
                    length_mus=E.round_length_mus_to_x_multiple_ps(self.dd_ReadoutTime/1000, self.round_to ),
                    A1=self.dd_A1Readout, A2=self.dd_A2Readout, repump = self.dd_CWRepump, gate=True,laser = True)
                
            seq.asc(name='readout_decay',
                length_mus=E.round_length_mus_to_x_multiple_ps(self.dd_ReadoutDecay/1000, self.round_to ), A1=False,
                A2=False, gate=True)
            
        def rf_pulse(freq, amp=None, dur=None, phase=0):
                amp = si1_amp if amp is None else amp
                seq.asc(length_mus=E.round_length_mus_to_x_multiple_ps(1), name='RFSafetyWait1', gateRF = True)
                if dur is None:
                    dur = si1_pi
                seq.asc(
                    length_mus=E.round_length_mus_to_x_multiple_ps(dur,1),
                    name='nuc',
                    gateRF = True,
                    pd2g2={"type": "sinehermite_rf", "frequencies": [freq], "amplitudes": [amp], "phases": [phase],
                           "phase_offset_type": 'coherent'}
                )
                seq.asc(length_mus=E.round_length_mus_to_x_multiple_ps(2), name='RFSafetyWait2', gateRF = True)
            
        def prepare_dd_MWes():        #DD MWes
            self.amps=[]
            self.amps_pi2=[]
            self.freqs=[]
            for MW in ["MW1","MW2","MW3","MW4","MW5","MW6"]:
                active = getattr(self, f"dd_{MW}", False)
                amp = getattr(self, f"dd_{MW}_Amp", None)
                amp_pi2 = getattr(self, f"dd_{MW}_Pi2_Amp", None)
                freq = getattr(self, f"dd_{MW}_Freq", None)

                if active and amp is not None and freq is not None and amp_pi2 is not None:
                    self.amps.append(amp)
                    self.freqs.append(freq)
                    self.amps_pi2.append(amp_pi2)

            if (np.sum(self.amps) > 1):
                logger.error(
                    "Combined Microwave power of all active channels too high! Need value below 1. Value of {} was given. Check MW 4 and 5.",
                    np.sum(self.amps),self.amps)
                raise Exception

        gateMW_dur = 0.256
        self.round_to = 1
        round_to = self.round_to
    
        ancient_self_variables = {}
        sig = inspect.signature(self.setup_seq)
        for parameter in sig.parameters.keys():
            # print(parameter)
            exec(f"ancient_self_variables['{parameter}']=self.{parameter}")
            if locals()[parameter] != None:
                exec(f"self.{parameter}={parameter}")
                # print(parameter)
                # exec(f'print(self.{parameter})')
    
        # Setup list of all frequencies which the sequence should output.
        # self.tau_duration = np.unique(E.round_length_mus_to_x_multiple_ps(
        #     np.arange(self.dd_Tau_Min, self.dd_Tau_Max + self.dd_Tau_Step, self.dd_Tau_Step) / 1000,self.round_to )) # in µs
        # self.time_differences.stop()
        # time.sleep(0.02)  # maybe the timetagger would get too much commands at the same time
    
        # self.time_differences = self.setup_time_tagger(
        #     n_histograms=len(self.tau_duration)*2,
        #     binwidth=int(self.dd_Binning * 1000), # dd_Binning input is in ns.
        #     n_bins=int(self.dd_ReadoutTime / self.dd_Binning)
        #     )


        # self.sweep = np.unique(E.round_length_mus_to_x_multiple_ps(
        #     np.arange(self.dd_Tau_Min, self.dd_Tau_Max + self.dd_Tau_Step, self.dd_Tau_Step) / 1000,self.round_to )) # in µs

        self.time_differences.stop()
        time.sleep(0.02)  # maybe the timetagger would get too much commands at the same time
    
        self.time_differences = self.setup_time_tagger(
            n_histograms=len(self.sweep)*2,
            binwidth=int(self.dd_Binning * 1000), # dd_Binning input is in ns.
            n_bins=int(self.dd_ReadoutTime / self.dd_Binning)
            )
        
    
        #DD MWes
        self.amps=[]
        self.amps_pi2=[]
        self.freqs=[]
        prepare_dd_MWes()

        #init_MWes
        self.init_amps=[]
        self.init_freqs=[]
        for MW in ["MW1","MW2","MW3","MW4","MW5","MW6"]:
            init = getattr(self, f"dd_{MW}_init", False)
            amp = getattr(self, f"dd_{MW}_Amp", None)
            freq = getattr(self, f"dd_{MW}_Freq", None)

            if init and amp is not None and freq is not None:
                self.init_amps.append(amp)
                self.init_freqs.append(freq)

    
        seq = self._awg.mcas(name="DD", ch_dict={"2g": [1, 2], "ps": [1]})
        # generate segment of repump which starts at each repetition of the sequence.
        seq.start_new_segment("SYNCING")
        seq.asc(name='tt_sync1', length_mus=E.round_length_mus_to_x_multiple_ps(0.016, self.round_to), memory=True)        
        seq.asc(name='tt_sync2', length_mus=E.round_length_mus_to_x_multiple_ps(0.016, self.round_to), gate=True)
        
        seq.start_new_segment("Start")
        if self.dd_PulsedRepump:
            print("DD Pulsed Repump: ", self.dd_PulsedRepump)
            seq.asc(name='repumpOn',
                    length_mus=E.round_length_mus_to_x_multiple_ps(self.dd_RepumpDuration, round_to),
                    repump=True)
            seq.asc(name='repumpOff',
                    length_mus=E.round_length_mus_to_x_multiple_ps(self.dd_RepumpDecay, round_to),
                    repump=False)
    
    
        if (self.dd_DecayInit - gateMW_dur) > 0:
            dd_DecayInit = self.dd_DecayInit - gateMW_dur
        else:
            dd_DecayInit = self.dd_DecayInit


        #init nuclear spin for better contrast
        if len(self.freqs)==1:
           sna.electron_init(seq,self.dd_Init_State,self.dd_InitTime,
                               [self.dd_MW1_Freq,self.dd_MW2_Freq], #L transitions
                               [self.dd_MW5_Freq,self.dd_MW6_Freq], #R transitions
                               segment_length = 10, max_amp = .15, newsegment=False)
           
           erabi(freq=self.freqs,length=self.dd_Pi_dur,amp=self.amps)
           
           si_bath = self._tt.si_bath
           si1_freq, si1_amp, si1_pi, si1_pi2 = sna.get_hardcoded_si1_values(self.dd_Init_State,si_bath)
           rf_pulse(si1_freq)
           
    


        for value in self.sweep:
            exec(f"self.{self.parameter_to_sweep}={value}")

            prepare_dd_MWes()
            #seq.start_new_segment("Init")
            sna.electron_init(seq,self.dd_Init_State,self.dd_InitTime,
                              [self.dd_MW1_Freq,self.dd_MW2_Freq], #L transitions
                              [self.dd_MW5_Freq,self.dd_MW6_Freq], #R transitions
                              segment_length = 10, max_amp = .15, newsegment=False)

            if self.dd_DD_type.lower()=="rabi":
                print("doing rabi")
                seq.asc(name="gateMW", gateMW=True, length_mus=E.round_length_mus_to_x_multiple_ps(gateMW_dur, self.round_to ))
                seq.asc(name="Tau_pulse",
                    pd2g1={"type": self.dd_waveform, "frequencies": [*self.freqs],
                            "amplitudes":[ *self.amps]},
                    length_mus=self.duration,
                    gateMW=True)  # self.rabi_piPulseDuration is divided by 1000 to be in µs
            else:
                dd(self.duration,last_phase=self.first_dd_last_pi2_phase)
            
            #some_wait_before readout
            seq.asc(length_mus=E.round_length_mus_to_x_multiple_ps(self.wait,self.round_to))
            readout()
            
            #seq.start_new_segment("Reinit")
            pd2g1 = {'frequencies': [self.dd_MW3_Freq,self.dd_MW4_Freq], 'type': self.dd_waveform, 'amplitudes': [self.dd_MW3_Amp,self.dd_MW4_Amp]}
                #print("DURRATION::::::::::",self.dur_step[alt_step][5])

            seq.asc(length_mus=E.round_length_mus_to_x_multiple_ps(self.dd_Pi_dur,round_to), gateMW=True, pd2g1=pd2g1,name='reinit')

            #DD with second phase
            #seq.start_new_segment("Init")
            sna.electron_init(seq,self.dd_Init_State,self.dd_InitTime,
                              [self.dd_MW1_Freq,self.dd_MW2_Freq], #L transitions
                              [self.dd_MW5_Freq,self.dd_MW6_Freq], #R transitions
                              segment_length = 10, max_amp = .24, newsegment=False)

            if self.dd_DD_type.lower()=="rabi":
                print("doing rabi")
                seq.asc(name="gateMW", gateMW=True, length_mus=E.round_length_mus_to_x_multiple_ps(gateMW_dur, self.round_to ))
                seq.asc(name="Tau_pulse",
                    pd2g1={"type": self.dd_waveform, "frequencies": [*self.freqs],
                            "amplitudes":[ *self.amps]},
                    length_mus=self.duration,
                    gateMW=True)  # self.rabi_piPulseDuration is divided by 1000 to be in µs
            else:
                dd(self.duration,last_phase=self.second_dd_last_pi2_phase)
                   
            seq.asc(length_mus=E.round_length_mus_to_x_multiple_ps(self.wait,self.round_to))
            readout()
            
            #seq.start_new_segment("Reinit")
            pd2g1 = {'frequencies': [self.dd_MW3_Freq,self.dd_MW4_Freq], 'type': self.dd_waveform, 'amplitudes': [self.dd_MW3_Amp,self.dd_MW4_Amp]}
                #print("DURRATION::::::::::",self.dur_step[alt_step][5])

            seq.asc(length_mus=E.round_length_mus_to_x_multiple_ps(self.dd_Pi_dur,round_to), gateMW=True, pd2g1=pd2g1,name='reinit')

        seq.start_new_segment("cooldown",loop_count=self.cooldown_loops)
        seq.asc(length_mus=E.round_length_mus_to_x_multiple_ps(100,round_to),name='cooldown') # I noticed heating


        # self.awg.mcas.status = 1
        self._awg.mcas_dict.stop_awgs()
        self._awg.mcas_dict['DD'] = seq
        self._awg.mcas_dict.print_info()
        self._awg.mcas_dict['DD'].run()

        for key, val in ancient_self_variables.items():  # restore the ancient variables
            exec(f"self.{key}={val}")

    def periodic_refocus(self):
        #print("asking periodic refocus")
        

        t=datetime.datetime.now()
        current_time = int(t.hour)+int(t.minute)/60
        if current_time > self.start_pause_time and current_time < self.end_pause_time:
            print('3 am pause. Good night, rest well.')
            return
               

        if self.measurement_running:
            self.cter+=1
            print("doing periodic refocus")
            self.dd_Stop_Button_Clicked(True)

            QtTest.QTest.qSleep(1000)

            if self.cter>5:
                self.cter=0
                try:
                    self.save_dd_data()
                except:
                    print("could not save")

            self._PLE_logic.Lock_laser=True
            self._PLE_logic.happy=True

            self._PLE_logic.start_scanning()
            #print(self.queue.module_state)
            #print("inside")
            while self._PLE_logic.module_state() == 'locked':
                #print(self._PLE_logic.module_state())
                QtTest.QTest.qSleep(500)

            QtTest.QTest.qSleep(10000)
            
            self._awg.mcas_dict.stop_awgs()
            self._PLE_logic.happy = False

            self._awg.mcas_dict.awgs["ps"].constant(pulse=(0, ['A2','A1',"repump","laser"], self._awg.mcas_dict.awgs["ps"].analog_volt, 0))

            pos=self._confocal.get_position()
            print("initial_pos1:",pos)
            self._optimizer.start_refocus(pos,caller_tag = "DD")
            while self._optimizer.module_state() =='locked':
                QtTest.QTest.qSleep(500)
            #self._awg.mcas_dict.stop_awgs()
            print("pos after refocus:",pos)
            QtTest.QTest.qSleep(2000)

            
            self.dd_Continue_Button_Clicked(True)

 


    def do_fit(self, x_data, y_data, tag):
        
        x_data=x_data.astype(np.float)
        y_data=y_data.astype(np.float)
        self.interpolated_x_data=np.linspace(x_data.min(),x_data.max(),len(x_data)*10) # for the fitting part

        if tag == 'Cosinus':
            #print("Doing Cosinus")
            model,params=self._fit_logic.make_sine_model()

            result = self._fit_logic.make_sine_fit(
                                x_axis=x_data,
                                data=y_data,
                                units='Hz',
                                estimator=self._fit_logic.estimate_sine
                                )

            #fit_func=self._fit_logic.sine
            #result=fit_func.fit(x_data,y_data)
            #self.fit_data = fit_func(x=self.interpolated_x_data, *result.x)

        if tag == 'Cosinus+Phase':
            #print("Doing Cosinus+Phase")
            model,params=self._fit_logic.make_sine_model()

            result = self._fit_logic.make_sine_fit(
                                x_axis=x_data,
                                data=y_data,
                                units='Hz',
                                estimator=self._fit_logic.estimate_sine
                                )
        
        if tag == 'Cosinus+Decay':
            #print("Doing Cosinus+Phase")
            model,params=self._fit_logic.make_sineexponentialdecay_model()

            result = self._fit_logic.make_sine_fit(
                                x_axis=x_data,
                                data=y_data,
                                units='Hz',
                                estimator=self._fit_logic.estimate_sineexponentialdecay
                                )

        
        self.fit_data = model.eval(x=self.interpolated_x_data, params=result.params)
        self.Amplitude_Fit:str=''
        self.Frequency_Fit:str=''
        self.Phase_Fit:str=''
        self.tau_pulse:float=0 #ns

        try:
            self.Amplitude_Fit=str(round(result.params["amplitude"].value,2))
            self.Frequency_Fit=str(round(result.params["frequency"].value*1e3,2))
            self.Phase_Fit=str(round(result.params["phase"].value*180/np.pi,2))
            self.pi_pulse=round(1/(result.params["frequency"].value)/2,6)
            target=np.pi
            self.pi_pulse=round((target-self.Phase_Fit)/self.Frequency_Fit,6)
            print((target-self.Phase_Fit)/self.Frequency_Fit)

        except Exception as e:
            print("an error occured during fitting in DD:\n", e)

        self.dd_FitParams="Amplitude: "+self.Amplitude_Fit+"\n"+"Frequency  (MHz): "+self.Frequency_Fit+"\n"+"Pi pulse (µs): "+str(self.pi_pulse)+"\n"+"Phase: "+self.Phase_Fit
        
        self.sigFitPerformed.emit(1/(result.params["frequency"].value)/2)

        return self.interpolated_x_data,self.fit_data,result

from threading import Timer
class RepeatedTimer(object):
    def __init__(self, interval, function, *args, **kwargs):
        self._timer     = None
        self.interval   = interval
        self.function   = function
        self.args       = args
        self.kwargs     = kwargs
        self.is_running = False
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False