#Arb Seq-Logic for mcas-module from Javid which combines AWG and ps and uses AWG as master.

import numpy as np
import sys
sys.path.append('C:\src\qudi\hardware\Keysight_AWG_M8190\pyarbtools_master') #quickfix to proceed, should be improved

from core.module import Base
from core.connector import Connector
#from hardware.swabian_instruments.timetagger import TT as TimeTagger
from logic.generic_logic import GenericLogic
#import hardware.Keysight_AWG_M8190.pyarbtools_master.pyarbtools as pyarbtools

from logic.arb_seq_logic.arb_seq_default_values_and_widget_functions import arb_seq_default_values_and_widget_functions as arb_seq_default

#import connector files for easy coding and fast file finding
from logic.save_logic import SaveLogic
from logic.fit_logic import FitLogic
# from hardware.Keysight_AWG_M8190.pym8190a import mcas_dict_holder


from qtpy import QtCore
from PyQt5 import QtTest

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

class ArbSeqLogic(GenericLogic,arb_seq_default):
    #declare connectors
    counter_device = Connector(interface='TimeTaggerInterface')# Savelogic just for testing purposes
    savelogic = Connector(interface='SaveLogic')
    mcas_holder = Connector(interface='McasDictHolderInterface')
    fitlogic = Connector(interface='FitLogic')
    #transition_tracker = Connector(interface="TransitionTracker")

    CHANNEL_APD0 = 0
    CHANNEL_APD1 = 1
    CHANNEL_DETECT = 2
    CHANNEL_SEQUENCE = 3

    arbseq_Filename = StatusVar('arbseq_Filename', "Filename")
    arbseq_Tau_Min = StatusVar(default=20)#ms
    arbseq_Tau_Max = StatusVar(default=40) #ms
    arbseq_Tau_Step = StatusVar(default=20)#ms
    arbseq_Tau_Decay = StatusVar(default=1000) #rep

    arbseq_InitTime = StatusVar(default=100) #ms
    arbseq_DecayInit = StatusVar(default=1000) #rep
    arbseq_RepumpDecay = StatusVar(default=1000) #rep
    arbseq_RepumpDuration = StatusVar(default=50) #ms
    arbseq_CTLDecay = StatusVar(default=1000) #rep
    arbseq_CTLDuration = StatusVar(default=50) #ms
    arbseq_AOMDelay = StatusVar(default=100) #ms
    arbseq_ReadoutTime = StatusVar(default=100) #ms
    arbseq_ReadoutDecay = StatusVar(default=1000) #rep
    arbseq_Binning = StatusVar(default=10e6) #ns

    meas_time = 0 #s
    decay_time = 0 #s


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
    sigArbSeqPlotsUpdated = QtCore.Signal()
    SigClock= QtCore.Signal()
    SigCheckReady_Beacon = QtCore.Signal()
    sigFitPerformed= QtCore.Signal(np.float)

    starting_time=0

    def on_activate(self):
        #super().__init__() #maybe this will make it possible to reload the default val and widget functions
        self._time_tagger=self.counter_device()
        self._time_tagger.setup_TT()
        self._save_logic:SaveLogic = self.savelogic()
        self._awg = self.mcas_holder()#mcas_dict()
        self._fit_logic:FitLogic = self.fitlogic()
        #self._transition_tracker=self.transition_tracker()
        
        self.stop_awg = self._awg.mcas_dict.stop_awgs
        self.Timer = RepeatedTimer(1, self.clock) # this clock is not very precise, maybe the solution proposed on https://stackoverflow.com/questions/474528/what-is-the-best-way-to-repeatedly-execute-a-function-every-x-seconds can be helpful.
        #self.SigCheckReady_Beacon.connect(self.print_counter)
        self.CheckReady_Beacon = RepeatedTimer(1, self.get_data)
        #self.CheckReady_Beacon.start()
        self.number_of_points_per_line=self._time_tagger._time_diff["n_histograms"]
        self.measurement_running=False
        self.counter=self._time_tagger.counter()
        self.time_differences = self._time_tagger.time_differences()
        self.scanmatrix=np.zeros(np.array(self.time_differences.getData(),dtype=object).shape)
        self.fit_data = np.zeros_like(self.scanmatrix)[0]
        self.interpolated_x_data = np.linspace(0,1,len(self.fit_data))
        self.Amplitude_Fit:str=''
        self.Beta_Fit:str=''
        self.Lifetime_Fit:str=''
        self.Rate_Fit:str=''
        self.Offset_Fit:float=0 #ns
        self.SigCheckReady_Beacon.connect(self.get_data)
        self.readout_loops = 1
        
   
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
    
    def get_data(self):
        if time.time()-self.starting_time>self.arbseq_Stoptime and self.arbseq_Stoptime!=0 and self.measurement_running:
            self.arbseq_Stop_Button_Clicked(True)

        #print("checkready:",self.measurement_running)
        if not(self.measurement_running):
                return
            
        else:
            indexes=np.array(self.time_differences.getIndex()) #readout binwidth (ps)
            self.scanmatrix=np.array(self.time_differences.getData(),dtype=object) #readout data from timetagger
            self.measured_times_ns=indexes/1e3 #indexes is in ps
            mask=((self.measured_times_ns>=self.arbseq_AOMDelay) & (self.measured_times_ns<=self.arbseq_IntegrationTime+self.arbseq_AOMDelay)) #create mask to filter counts depending on arrival time
            self.data=np.sum(self.scanmatrix[:,mask],axis=1) #sum up the readout-histogram after a single Tau
            self.data_detect=np.sum(self.scanmatrix,axis=0) #sum up the histograms to see the emission decay
            self.measured_times=indexes/1e12 #binwidth in seconds

            # new for live plotting
            self.data_ionize = np.zeros_like(self.data)
            self.data_ionize_err = np.zeros_like(self.data)
            data = self.scanmatrix
            factor_repump   = self.arbseq_RepumpDecay / 1e3                  #Pulsed Decay
            factor_init   = self.arbseq_DecayInit / 1e3                    #Init Decay
            factor_read   = self.arbseq_ReadoutDecay / 1e3                    #Readout Decay
            factor_ion   = self.arbseq_Tau_Decay/ 1e3                     #Tau Decay

            Time = self.measured_times*1e9  # ns
            CTL_Time  = self.tau_duration
            Time = np.array(Time) / 1e9 # s
            CTL_Time = np.array(CTL_Time) / 1e3 *factor_ion# s

            repump_t  = self.arbseq_RepumpDuration / 1e3 *factor_repump # s     #Pulsed Duration
            init_t    = self.arbseq_InitTime / 1e3 *factor_init # s       #Init Time
            readout_t = self.arbseq_ReadoutTime / 1e3 *factor_read # s       #Readout Time
            bg_CTL   = self.arbseq_AOMDelay /1e3   *factor_read # s       #AOM Delay

            if self.meas_time == 0:
                meas_time = init_t/5
            else:
                meas_time = min(self.meas_time,init_t)

            if self.decay_time == 0:
                decay_time = min(meas_time,init_t)/10# s
            else:
                decay_time = min(self.decay_time,meas_time)


            for i in range(data.shape[0]):
                min_1, max_1 = repump_t+init_t-meas_time+decay_time, repump_t+init_t-decay_time
                min_2, max_2 = repump_t+init_t+CTL_Time[i]+min(decay_time,readout_t/5), repump_t+init_t+CTL_Time[i] + min(meas_time,readout_t)-min(decay_time,readout_t/5)
                min_off, max_off = Time[-1] - min(meas_time,bg_CTL)+min(decay_time,bg_CTL/5),Time[-1]-min(decay_time,bg_CTL/5)

                line = data[i]

                mask_off = (Time >= min_off) & (Time <= max_off)
                off = np.sum(line[mask_off])
                doff = np.sqrt(off)/len(line[mask_off])
                off /= len(line[mask_off])

                mask_1 = (Time >= min_1) & (Time <= max_1)
                c1= np.sum(line[mask_1])
                dc1 = np.sqrt(c1)/len(line[mask_1])
                c1 /= len(line[mask_1])

                mask_2 = (Time >= min_2) & (Time <= max_2)
                c2 = np.sum(line[mask_2])
                dc2 = np.sqrt(c2)/len(line[mask_2])
                c2 /= len(line[mask_2])

                try:
                    ratio = (c2-off)/(c1-off)
                    err = np.abs((c2-off)/(c1-off)**2) * dc1 + np.abs((1)/(c1-off)) * dc2 + np.abs((c2-c1)/(c1-off)**2) * doff
                except ZeroDivisionError:
                    try:
                        ratio = (c2)/(c1)
                        err = 0
                    except ZeroDivisionError:
                        ratio = 1
                        err = 0

                self.data_ionize[i] = ratio
                self.data_ionize_err[i] = err

            self.sigArbSeqPlotsUpdated.emit()



    def clock(self):
        self.SigClock.emit()

    def CheckReady(self):
        self.SigCheckReady_Beacon.emit()

    def setup_time_tagger(self,**kwargs):
        self._time_tagger._time_diff.update(**kwargs)
        return self._time_tagger.time_differences()

    def save_arbseq_data(self, tag=None, colorscale_range=None, percentile_range=None):
        """ Saves the current Arb Seq data to a file."""
        timestamp = datetime.datetime.now()
        filepath = self._save_logic.get_path_for_module(module_name='ArbSeq')
        
        if tag is None:
            tag = ''

        if len(tag) > 0:
            filelabel_raw = '{0}_ArbSeq_raw'.format(tag)
            filelabel_detection = '{0}_ArbSeq_detection'.format(tag)
            filelabel_matrix = '{0}_ArbSeq_matrix'.format(tag)
        else:
            filelabel_raw = '_ArbSeq_raw'
            filelabel_detection = '_ArbSeq_detection'
            filelabel_matrix = '_ArbSeq_matrix'

        data_raw = OrderedDict()
        data_detection = OrderedDict()
        data_matrix = OrderedDict()
        data_raw['ionization prob.'] = self.data_ionize
        data_raw['ionization prob. error'] = self.data_ionize_err
        data_raw['count data (counts)'] = self.data
        # data_raw['Tau (ms)'] = self.tau_duration*self.arbseq_Tau_Decay/1e3
        data_raw['Tau (ms)'] = self.tau_duration
        data_detection['Detection Time (ns)'] = self.measured_times*1e9 # save in [ns]
        data_detection['Detection Counts (counts)'] = self.data_detect
        data_matrix['Detection Time + Tau'] = self.scanmatrix
        parameters = OrderedDict()

        
        parameters['Runtime (s)'] = time.time()-self.starting_time # TODO This does not work with continue measurement
        parameters['Enable Microwave1 (bool)'] = self.arbseq_MW1
        parameters['Enable Microwave2 (bool)'] = self.arbseq_MW2
        parameters['Enable Microwave3 (bool)'] = self.arbseq_MW3
        parameters['Microwave1 CW Power (dBm)'] = self.arbseq_MW1_Power
        parameters['Microwave2 CW Power (dBm)'] = self.arbseq_MW2_Power
        parameters['Microwave3 CW Power (dBm)'] = self.arbseq_MW3_Power
        parameters['Microwave1 Frequency (MHz)'] = self.arbseq_MW1_Freq
        parameters['Microwave2 Frequency (MHz)'] = self.arbseq_MW2_Freq
        parameters['Microwave3 Frequency (MHz)'] = self.arbseq_MW3_Freq
        parameters['Tau min (ns)'] = self.arbseq_Tau_Min
        parameters['Tau max (ns)'] = self.arbseq_Tau_Max
        parameters['Tau Step (ns)'] = self.arbseq_Tau_Step
        parameters['Tau Decay (ns)'] = self.arbseq_Tau_Decay
        parameters['A1 (bool)'] = self.arbseq_A1
        parameters['A2 (bool)'] = self.arbseq_A2
        parameters['Pulsed Repump (bool)'] = self.arbseq_PulsedRepump
        parameters['Pulsed Duration (us)'] = self.arbseq_RepumpDuration
        parameters['Pulsed Decay (us)'] = self.arbseq_RepumpDecay
        parameters['CW Repump (bool)'] = self.arbseq_CWRepump
        parameters['Init Time (us)'] = self.arbseq_InitTime
        parameters['Init Decay (us)'] = self.arbseq_DecayInit
        parameters['Readout Time (us)'] = self.arbseq_ReadoutTime
        parameters['Readout Decay (us)'] = self.arbseq_ReadoutDecay
        parameters['Readout via A1 (bool)'] = self.arbseq_A1Readout
        parameters['Readout via A2 (bool)'] = self.arbseq_A2Readout
        parameters['AOM Delay (ns)'] = self.arbseq_AOMDelay
        parameters['Integration Window (ns)'] = self.arbseq_IntegrationTime
        parameters['Binning (ns)'] = self.arbseq_Binning
        parameters['Amplitude Fit'] = self.Amplitude_Fit
        parameters['Beta Fit'] = self.Beta_Fit
        parameters['Lifetime Fit'] = self.Lifetime_Fit
        # parameters['Rate Fit'] = self.Rate_Fit
        parameters['Offset Fit']= self.Offset_Fit
       

        print(data_raw['Tau (ms)'])
        print(data_raw['count data (counts)'])
        fig = self.draw_figure(
            data_raw['Tau (ms)'],
            data_raw['ionization prob.'],
            data_matrix['Detection Time + Tau'],
            data_detection['Detection Time (ns)'],
            data_detection['Detection Counts (counts)'],
            self.interpolated_x_data,
            self.fit_data,
            cbar_range=colorscale_range,
            percentile_range=percentile_range,
        )

        self._save_logic.save_data(data_matrix,
                                    filepath=filepath,
                                    parameters=parameters,
                                    filelabel=filelabel_matrix,
                                    fmt='%.6e',
                                    delimiter='\t',
                                    timestamp=timestamp,
                                    plotfig=fig)
        
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
                                    timestamp=timestamp)

        self.log.info('ArbSeq data saved to:\n{0}'.format(filepath))
        return

    def draw_figure(self, time_data, count_data, matrix_data, detection_time, detection_counts, fit_freq_vals, fit_count_vals, cbar_range=None, percentile_range=None):
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

        # # Rescale counts data with SI prefix
        # while np.max(count_data) > 1000:
        #     count_data = count_data / 1000
        #     #fit_count_vals = fit_count_vals / 1000
        #     prefix_index = prefix_index + 1

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



        


        ax_mean.plot(time_data, count_data, linestyle=':', linewidth=0.5)
        # Do not include fit curve if there is no fit calculated.
        if max(fit_count_vals) > 0:
            ax_mean.plot(fit_freq_vals, fit_count_vals, marker='None')
        # ax_mean.set_ylabel('Fluorescence (' + counts_prefix + 'counts)')
        ax_mean.set_xlabel('Laser pulse durations (ms)')
        ax_mean.set_xlim(np.min(time_data), np.max(time_data))
        matrixplot = ax_matrix.imshow(
            matrix_data,
            cmap=plt.get_cmap('inferno'),  # reference the right place in qd
            origin='lower',
            vmin=cbar_range[0],
            vmax=cbar_range[1],
            extent=[np.min(detection_time),
                    np.max(detection_time),
                    np.min(time_data)-(np.max(time_data)-np.min(time_data))/(np.shape(matrix_data)[0]-1)/2,
                    np.max(time_data)+(np.max(time_data)-np.min(time_data))/(np.shape(matrix_data)[0]-1)/2
                    ],
            aspect='auto',
            interpolation='nearest')

        ax_matrix.set_xlabel('Sequence duration (' + mw_prefix + 's)')
        ax_matrix.set_ylabel('Pulse dur. (ms)')

        ax_detection.plot(detection_time, detection_counts, linestyle=':', linewidth=0.5)
        ax_detection.set_xlabel('Sequence duration (' + mw_prefix + 's)')
        ax_detection.set_ylabel('Fluorescence (' + counts_prefix + 'counts)')
        ax_detection.set_xlim(np.min(detection_time), np.max(detection_time))

        # Adjust subplots to make room for colorbar & upper xlabel
        fig.subplots_adjust(right=0.8,hspace=0.6)

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


    def setup_seq(
            self,
            arbseq_Tau_Min=None,
            arbseq_Tau_Max=None,
            arbseq_Tau_Step=None,
            arbseq_Tau_Decay=None,
    
            arbseq_MW1=None,
            arbseq_MW1_Freq=None,
            arbseq_MW1_Power=None,
            arbseq_MW2=None,
            arbseq_MW2_Freq=None,
            arbseq_MW2_Power=None,
            arbseq_MW3=None,
            arbseq_MW3_Freq=None,
            arbseq_MW3_Power=None,
            arbseq_MW4=None,
            arbseq_MW4_Freq=None,
            arbseq_MW4_Power=None,
            arbseq_MW4_Duration=None,
            arbseq_MW5=None,
            arbseq_MW5_Freq=None,
            arbseq_MW5_Power=None,
            arbseq_MW5_Duration=None,
    
            arbseq_A1=None,
            arbseq_A2=None,
            arbseq_A1Readout=None,
            arbseq_A2Readout=None,
            arbseq_InitTime=None,
            arbseq_DecayInit=None,
            arbseq_RepumpDecay=None,
            arbseq_CWRepump=None,
            arbseq_PulsedRepump=None,
            arbseq_RepumpDuration=None,
            arbseq_AOMDelay=None,
            arbseq_IntegrationTime=None,
            arbseq_Binning=None,
            arbseq_Interval=None,
            arbseq_PeriodicSaving=None,
            arbseq_Stoptime=None,
    
            arbseq_ReadoutTime=None,
            arbseq_ReadoutDecay=None
    ):
        gateMW_dur = 0.256
        self.round_to = 16
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
        #     np.arange(self.arbseq_Tau_Min, self.arbseq_Tau_Max + self.arbseq_Tau_Step, self.arbseq_Tau_Step) / 1000,16)) #FIXME just uncomment when using normal arbseq again
        self.tau_duration = np.arange(self.arbseq_Tau_Min, self.arbseq_Tau_Max + self.arbseq_Tau_Step, self.arbseq_Tau_Step) #FIXME just uncomment when using normal arbseq again
        self.time_differences.stop()
        time.sleep(0.02)  # maybe the timetagger would get too much commands at the same time
        # self.time_differences = self.setup_time_tagger(n_histograms=len(self.tau_duration),
        #                                                 binwidth=int(self.arbseq_Binning * 1000),
        #                                                 # arbseq_Binning input is in ns.
        #                                                 n_bins=int(self.arbseq_ReadoutTime*self.readout_loops / self.arbseq_Binning)
        #                                                 )
    
        self.power = []
        if self.arbseq_MW2:
            self.power += [self.arbseq_MW2_Power]
        # if self.arbseq_MW3:
        #     self.power += [self.arbseq_MW3_Power]
    
        self.power = np.asarray(self.power)
        self.power = self.power_to_amp(self.power)
        if np.sum(self.power) > 1:
            logger.error(
                "Combined Microwavepower of all active channels too high! Need value below 1. Value of {} was given.",
                np.sum(self.power))
            raise Exception
    
        meas_time = (self.arbseq_RepumpDuration*int(self.arbseq_RepumpDecay)
            +self.arbseq_InitTime*int(self.arbseq_DecayInit)
            +self.arbseq_MW4_Duration*int(self.arbseq_MW4_Power)*self.arbseq_MW4
            +self.arbseq_MW5_Duration*int(self.arbseq_MW5_Power)*self.arbseq_MW5
            +(max(self.tau_duration))*int(self.arbseq_Tau_Decay)
            +self.arbseq_ReadoutTime*int(self.arbseq_ReadoutDecay)
            +self.arbseq_AOMDelay*int(self.arbseq_ReadoutDecay)
            )*1000
        # print("n_bins", meas_time / self.arbseq_Binning)
        # print("int(n_bins)", int(meas_time / self.arbseq_Binning))
        # print("meas_time", meas_time)
        # print("n_hist", len(self.tau_duration))
        self.time_differences = self.setup_time_tagger(n_histograms=len(self.tau_duration),
                                                binwidth=int(self.arbseq_Binning * 1000),
                                                n_bins=int(meas_time / self.arbseq_Binning)
                                                )
    
        
        seq = self._awg.mcas(name="ArbSeq", ch_dict={"2g": [1, 2], "ps": [1]})
        seq.start_new_segment("Startup", loop_count=1)
        seq.asc(name='tt_sync1', length_mus=E.round_length_mus_to_x_multiple_ps(0.064, round_to),
                memory=True)
        seq.asc(name='wait', length_mus=E.round_length_mus_to_x_multiple_ps(0.064, round_to))
        # short pulses to SYNC and TRIGGER the timedifferences module of TimeTagger.
        seq.asc(name='tt_sync2', length_mus=E.round_length_mus_to_x_multiple_ps(0.064, round_to),
                gate=True)
        seq.asc(name='wait1', length_mus=E.round_length_mus_to_x_multiple_ps(0.064, round_to))
        
        for tau in self.tau_duration:
            seq.start_new_segment("wait2", loop_count=1)
            seq.asc(name='wait_asc', length_mus=E.round_length_mus_to_x_multiple_ps(0.064, round_to), gate = False)
        
            seq.start_new_segment("Repump", loop_count=int(self.arbseq_RepumpDecay))
            seq.asc(name='repumpOn', length_mus=self.arbseq_RepumpDuration, repump=True, gate=True) # just one gate=True, otherwise odd numbers cause des-sychronization
            
            seq.start_new_segment("Res", loop_count=int(self.arbseq_DecayInit))
            if self.arbseq_MW3:
                seq.asc(name='OnlyResonant1', pd2g1={"type": "sine", "frequencies": [self.arbseq_MW2_Freq],"amplitudes": self.power_to_amp(self.arbseq_MW2_Power)}, 
                                        length_mus=self.arbseq_InitTime, A2=self.arbseq_A2Readout, A1=self.arbseq_A1Readout)
            else:
                seq.asc(name='OnlyResonant1', length_mus=self.arbseq_InitTime, A2=self.arbseq_A2Readout, A1=self.arbseq_A1Readout)

            if self.arbseq_MW4:
                seq.start_new_segment("PreIon", loop_count=int(self.arbseq_MW4_Power))
                seq.asc(name='AddCTL', length_mus=self.arbseq_MW4_Duration, A2=True, A1=True, CTL=True)

            if self.arbseq_MW5:
                seq.start_new_segment("ResAfterPreIon", loop_count=int(self.arbseq_MW5_Power))
                seq.asc(name='OnlyResonant11', length_mus=self.arbseq_MW5_Duration, A2=True, A1=True, CTL=False)
            
            seq.start_new_segment("Ionization", loop_count=int(self.arbseq_Tau_Decay))
            if self.arbseq_MW2:
                seq.asc(name='AddCTL', pd2g1={"type": "sine", "frequencies": [self.arbseq_MW2_Freq],"amplitudes": self.power_to_amp(self.arbseq_MW2_Power)}, 
                                        length_mus=tau, A2=self.arbseq_A2, A1=self.arbseq_A1, CTL=self.arbseq_CWRepump,repump=self.arbseq_PulsedRepump)
            else:
                seq.asc(name='AddCTL', length_mus=tau, A2=self.arbseq_A2, A1=self.arbseq_A1, CTL=self.arbseq_CWRepump,repump=self.arbseq_PulsedRepump)
            
            seq.start_new_segment("Readout", loop_count=int(self.arbseq_ReadoutDecay))
            if self.arbseq_MW3:
                seq.asc(name='OnlyResonant2', pd2g1={"type": "sine", "frequencies": [self.arbseq_MW2_Freq],"amplitudes": self.power_to_amp(self.arbseq_MW2_Power)},
                                        length_mus=self.arbseq_ReadoutTime, A2=self.arbseq_A2Readout, A1=self.arbseq_A1Readout)
            else:
                seq.asc(name='OnlyResonant2', length_mus=self.arbseq_ReadoutTime, A2=self.arbseq_A2Readout, A1=self.arbseq_A1Readout)
            
            seq.start_new_segment("Readout2", loop_count=int(self.arbseq_Tau_Decay))
            if self.arbseq_MW3:
                seq.asc(name='OnlyResonant3', pd2g1={"type": "sine", "frequencies": [self.arbseq_MW2_Freq],"amplitudes": self.power_to_amp(self.arbseq_MW2_Power)},
                                        length_mus=max(self.tau_duration)-tau, A2=self.arbseq_A2Readout, A1=self.arbseq_A1Readout)
            else:
                seq.asc(name='OnlyResonant3', length_mus=max(self.tau_duration)-tau, A2=self.arbseq_A2Readout, A1=self.arbseq_A1Readout)

            seq.start_new_segment("CTLBG", loop_count=int(self.arbseq_ReadoutDecay))
            seq.asc(name='allOff', length_mus=self.arbseq_AOMDelay)
    
        seq.start_new_segment("wait3", loop_count=1)
        seq.asc(name='wait3_asc', length_mus=E.round_length_mus_to_x_multiple_ps(0.064, round_to))

        self._awg.mcas_dict.stop_awgs()
        self._awg.mcas_dict['ArbSeq']=seq
        self._awg.mcas_dict.print_info()
        self._awg.mcas_dict['ArbSeq'].run()
    
        for key, val in ancient_self_variables.items():  # restore the ancient variables
            exec(f"self.{key}={val}")

    def do_fit(self, x_data, y_data, tag):
        
        x_data=x_data.astype(np.float)
        y_data=y_data.astype(np.float)
        self.interpolated_x_data=np.linspace(x_data.min(),x_data.max(),len(x_data)*10) # for the fitting part

        if tag == 'Exponential':
            #print("Doing Cosinus+Phase")
            model,params=self._fit_logic.make_decayexponential_model()

            result = self._fit_logic.make_decayexponential_fit(
                                x_axis=x_data,
                                data=y_data,
                                units='sy',
                                estimator=self._fit_logic.estimate_decayexponential
                                )

        
        self.fit_data = model.eval(x=self.interpolated_x_data, params=result.params)
        self.Amplitude_Fit:str=''
        self.Beta_Fit:str=''
        self.Lifetime_Fit:str=''
        self.Rate_Fit:str=''
        self.Offset_Fit:float=0 #ns

        try:
            self.Amplitude_Fit=str(round(result.params["amplitude"].value,2))
            self.Beta_Fit=str(round(result.params["beta"].value*1e3,2))
            self.Lifetime_Fit=str(round(result.params["lifetime"].value,2))
            self.Rate_Fit=str(round(1/result.params["lifetime"].value*1e3,2))
            self.Offset_Fit=round(1/(result.params["offset"].value)/2,2)
        except Exception as e:
            print("an error occured during fitting in ArbSeq:\n", e)

        self.arbseq_FitParams="Amplitude: "+str(self.Amplitude_Fit)+"\n"+"Beta (): "+str(self.Beta_Fit)+"\n"+"Lifetime (ms): "+str(self.Lifetime_Fit)+"\n"+"Rate (Hz): "+str(self.Rate_Fit)+"\n"+"Offset: "+str(self.Offset_Fit)
        
        # self.sigFitPerformed.emit(1/(result.params["frequency"].value)/2)

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