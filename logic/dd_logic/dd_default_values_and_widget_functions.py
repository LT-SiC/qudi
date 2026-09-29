import numpy as np
import time
from core.statusvariable import StatusVar
import hardware.Keysight_AWG_M8190.elements as E
import notebooks.UserScripts.helpers.snippets_awg as sna
import re

pattern = re.compile(
    r"""
    ^\s*
    (?P<a>-?\d+(?:\.\d+)?)     # first number
    \s*->\s*
    (?P<b>-?\d+(?:\.\d+)?)     # second number
    \s*,\s*
    (?P<c>-?\d+(?:\.\d+)?)     # third number
    \s*$
    """,
    re.VERBOSE,
)

def parse_range(s: str):
    m = pattern.match(s)
    if not m:
        raise ValueError(f"Invalid format: {s!r}")

    return tuple(float(m.group(k)) for k in ("a", "b", "c"))

class dd_default_values_and_widget_functions:
        F=False
        T=True
        # dd_Filename:str="Filename"

        # dd_Tau_Min:float=1000 #ns
        # dd_Tau_Max:float=1500 #ns
        # dd_Tau_Step:float=2 #ns
        # dd_Tau_Decay:float= 500 #ns

        # dd_MW1:bool=T
        # dd_MW2:bool=T
        # dd_MW3:bool=F
        # dd_MW4:bool=F
        # dd_MW5:bool=F
        # dd_MW6:bool=F

        # dd_MW1_Amp:float=0.29796
        # dd_MW2_Amp:float=0.33477
        # dd_MW3_Amp:float=0.21976
        # dd_MW4_Amp:float=0.19739
        # dd_MW5_Amp:float=0.14306
        # dd_MW6_Amp:float=0.12047

        # dd_readout_power:float=0.2

        # dd_MW1_Pi2_Amp:float=0.14988
        # dd_MW2_Pi2_Amp:float=0.17488
        # dd_MW3_Pi2_Amp:float=0.10779
        # dd_MW4_Pi2_Amp:float=0.09303
        # dd_MW5_Pi2_Amp:float=0.06336
        # dd_MW6_Pi2_Amp:float=0.05009

        # dd_MW1_Freq:float=5000.
        # dd_MW2_Freq:float=5000+5.
        # dd_MW3_Freq:float=5000+70.
        # dd_MW4_Freq:float=5000+75.
        # dd_MW5_Freq:float=5000+140.
        # dd_MW6_Freq:float=5000+145.
        # dd_Pi_dur:float=1.
        # dd_Pi2_dur:float=1.
        # dd_DD_type:str="hahn"
        # dd_nrep:int=2

        # dd_Init_State:str="-3/2"
        # dd_enhanced_readout:bool=T

        # dd_A1:bool=F
        # dd_A2:bool=T
        # dd_A1Readout:bool=F
        # dd_A2Readout:bool=T
        # dd_InitTime:float= 30 #µs
        # dd_DecayInit:float= 1 #µs
        # dd_RepumpDecay:float =1 #µs
        # dd_CWRepump:bool=F
        # dd_PulsedRepump:bool=T
        # dd_RepumpDuration:float=5 #µs
        # dd_AOMDelay:float=0 #ns
        # dd_IntegrationTime:float= 20000 #ns
        # dd_Binning:float=5000 #ns

        # dd_Interval:float=0
        # dd_PeriodicSaving:bool=F
        # dd_Stoptime:float=0 #ns

        # dd_ReadoutTime:float=200000 #ns
        # dd_ReadoutDecay:float=500 #ns

        # dd_MaxIterations:float=20
        # dd_PerformFit:bool=F
        # dd_NumberOfPeaks:float=0

        # dd_cb_max:float=100
        # dd_cb_high_percentile:float=100
        # dd_cb_low_percentile:float=0
        # dd_cb_min:float=0

        # dd_Runtime:float=0
        # dd_Frequencies_Fit:float=70
        # dd_Linewidths_Fit:float=1

        # dd_FitFunction:str='Cosinus'
        # dd_FitParams:str=""
        # update_after_stop:bool=F

        # pi_pulse:float=100 #ns
        # dd_unit:bool=False

        # first_dd_last_pi2_phase=0
        # second_dd_last_pi2_phase=round(np.pi/2,4)

        def dd_Stop_Button_Clicked(self,on):
                #print('done something with dd_Stop_Button')
                self.scanmatrix_data=self.scanmatrix.copy()
                self.stop_awg()
                #self.holder.awg.mcas.status = 0
                self.stoping_time=time.time()
                self.measurement_running=False
                self.time_differences.stop()
                #del self._awg.mcas_dict['DD']

        def dd_Continue_Button_Clicked(self,on):
                #print('done something with dd_Continue_Button')
                self.continuing=True
                #self.setup_seq()
                self._awg.mcas_dict['DD'].run()
                self.starting_time+=time.time()-self.stoping_time
                self.time_differences = self.setup_time_tagger(
                        n_histograms=len(self.sweep)*2,
                        binwidth=int(self.dd_Binning * 1000), # dd_Binning input is in ns.
                        n_bins=int(self.dd_ReadoutTime / self.dd_Binning)
                        )  
                self.time_differences.start()
                self.measurement_running=True

        def dd_Run_Button_Clicked(self,on):
                #print('done something with dd_Run_Button')
                print("setting_up_sequence...")
                self.setup_seq()
                print("setting_up_sequence finished")
                self.scanmatrix=np.zeros(np.array(self.time_differences.getData(),dtype=object).shape)
                self.scanmatrix_data=np.zeros(np.array(self.time_differences.getData(),dtype=object).shape)
                self.starting_time=time.time()
                self.time_differences.clear()
                self.time_differences.start()
                self.measurement_running=True

        def dd_waveform_comboBox_TextChanged(self,text):
                self.dd_waveform=text.lower()

        def dd_InitTime_LineEdit_textEdited(self,text):
                #print('done something with dd_InitTime_LineEdit. Text=',text)
                try:
                        self.dd_InitTime=eval(text.replace(",",".").replace("tt","self._tt").replace("self.self","self"))
                except:
                        try:
                                self.sweep=np.arange(*parse_range(text))
                                self.parameter_to_sweep="dd_InitTime"
                        except:
                                pass

        def dd_readout_power_LineEdit_textEdited(self,text):
                #print('done something with dd_InitTime_LineEdit. Text=',text)
                try:
                        self.dd_readout_power=eval(text.replace(",",".").replace("tt","self._tt").replace("self.self","self"))
                except:
                        try:
                                self.sweep=np.arange(*parse_range(text))
                                self.parameter_to_sweep="dd_readout_power"
                        except:
                                pass

        def first_dd_last_pi2_phase_textEdited(self,text):
                try:
                        self.first_dd_last_pi2_phase=float(eval(text.replace(",",".")))

                except:
                        try:
                                self.sweep=np.arange(*parse_range(text))
                                self.parameter_to_sweep="first_dd_last_pi2_phase"
                        except:
                                pass

        def second_dd_last_pi2_phase_textEdited(self,text):
                try:
                        self.second_dd_last_pi2_phase=float(eval(text.replace(",",".")))
                        
                        self.sweep=np.arange(*parse_range(text))
                        self.parameter_to_sweep="second_dd_last_pi2_phase"
                except:
                        try:
                                self.sweep=np.arange(*parse_range(text))
                                self.parameter_to_sweep="second_dd_last_pi2_phase"
                        except:
                                pass

        def dd_Load_Button_Clicked(self,on):
                print('done something with dd_Load_Button')
                

        def dd_RepumpDecay_LineEdit_textEdited(self,text):
                #print('done something with dd_RepumpDecay_LineEdit. Text=',text)
                try:
                        self.dd_RepumpDecay=eval(text.replace(",",".").replace("tt","self._tt").replace("self.self","self"))
                        
                        self.sweep=np.arange(*parse_range(text))
                        self.parameter_to_sweep="dd_RepumpDecay"
                except:
                        
                        try:

                                self.sweep=np.arange(*parse_range(text))
                                self.parameter_to_sweep="dd_RepumpDecay"
                        except:
                                pass

        def dd_Stoptime_LineEdit_textEdited(self,text):
                #print('done something with dd_Stoptime_LineEdit. Text=',text)
                try:
                        self.dd_Stoptime=eval(text.replace(",",".").replace("tt","self._tt").replace("self.self","self"))
                except:
                        pass

        def dd_CWRepump_CheckBox_StateChanged(self,on):
                #print('done something with dd_CWRepump_CheckBox')
                self.dd_CWRepump=on==2

        def dd_Tau_Decay_lineEdit_textEdited(self,text):
                #print('done something with dd_Tau_Decay_lineEdit. Text=',text)
                try:
                        self.dd_Tau_Decay=eval(text.replace(",",".").replace("tt","self._tt").replace("self.self","self"))
                except:
                        pass

        def dd_Tau_Max_lineEdit_textEdited(self,text):
                #print('done something with dd_Tau_Max_lineEdit. Text=',text)
                try:
                        self.dd_Tau_Max=eval(text.replace(",",".").replace("tt","self._tt").replace("self.self","self"))
                        self.sweep = np.unique(E.round_length_mus_to_x_multiple_ps(
                                np.arange(self.dd_Tau_Min, self.dd_Tau_Max + self.dd_Tau_Step, self.dd_Tau_Step) / 1000,self.round_to )) # in µs
                        
                        self.parameter_to_sweep="duration"
                except:
                        pass

        def dd_A1_CheckBox_StateChanged(self,on):
                #print('done something with dd_A1_CheckBox')
                self.dd_A1=on==2

        def dd_RepumpDuration_LineEdit_textEdited(self,text):
                #print('done something with dd_RepumpDuration_LineEdit. Text=',text)
                try:
                        self.dd_RepumpDuration=eval(text.replace(",",".").replace("tt","self._tt").replace("self.self","self"))
                        
                except:
                        try:
                                self.sweep=np.arange(*parse_range(text))
                                self.parameter_to_sweep="dd_RepumpDuration"
                        except:
                                pass

        def dd_A1Readout_checkBox_StateChanged(self,on):
                #print('done something with dd_A1Readout_checkBox')
                self.dd_A1Readout=on==2

        def dd_Interval_LineEdit_textEdited(self,text):
                #print('done something with dd_Interval_LineEdit. Text=',text)
                try:
                        self.dd_Interval=eval(text.replace(",",".").replace("tt","self._tt").replace("self.self","self"))
                except:
                        pass

        def dd_PeriodicSaving_CheckBox_StateChanged(self,on):
                #print('done something with dd_PeriodicSaving_CheckBox')
                self.dd_PeriodicSaving=on==2


        
        def dd_DecayInit_LineEdit_textEdited(self,text):
                #print('done something with dd_DecayInit_LineEdit. Text=',text)
                try:
                        self.dd_DecayInit=eval(text.replace(",",".").replace("tt","self._tt").replace("self.self","self"))
                except:
                        pass

        def dd_A2Readout_checkBox_StateChanged(self,on):
                #print('done something with dd_A2Readout_checkBox')
                self.dd_A2Readout=on==2

        def dd_cb_max_DoubleSpinBox_Edited(self,value):
                #print('done something with dd_cb_max_DoubleSpinBox. Value=',value)
                self.dd_cb_max=value

        def dd_cb_high_percentile_DoubleSpinBox_Edited(self,value):
                #print('done something with dd_cb_high_percentile_DoubleSpinBox. Value=',value)
                self.dd_cb_high_percentile=value

        def dd_cb_low_percentile_DoubleSpinBox_Edited(self,value):
                #print('done something with dd_cb_low_percentile_DoubleSpinBox. Value=',value)
                self.dd_cb_low_percentile=value

        def dd_cb_min_DoubleSpinBox_Edited(self,value):
                #print('done something with dd_cb_min_DoubleSpinBox. Value=',value)
                self.dd_cb_min=value

        def dd_Save_Button_Clicked(self,on):
                self.save_dd_data()
                print('done something with dd_Save_Button')


        def dd_Tau_Min_lineEdit_textEdited(self,text):
                #print('done something with dd_Tau_Min_lineEdit. Text=',text)
                try:
                        self.dd_Tau_Min=eval(text.replace(",",".").replace("tt","self._tt").replace("self.self","self"))
                        self.duration=self.dd_Tau_Min
                        self.sweep = np.unique(E.round_length_mus_to_x_multiple_ps(
                                np.arange(self.dd_Tau_Min, self.dd_Tau_Max + self.dd_Tau_Step, self.dd_Tau_Step) / 1000,self.round_to )) # in µs
                        self.parameter_to_sweep="duration"
                except:
                        pass


        def dd_ReadoutTime_LineEdit_textEdited(self,text):
                #print('done something with dd_ReadoutTime_LineEdit. Text=',text)
                try:
                        self.dd_ReadoutTime=eval(text.replace(",",".").replace("tt","self._tt").replace("self.self","self"))
                except:
                        try:

                                self.sweep=np.arange(*parse_range(text))
                                self.parameter_to_sweep="dd_ReadoutTime"
                        except:
                                pass

        def dd_PulsedRepump_CheckBox_StateChanged(self,on):
                #print('done something with dd_PulsedRepump_CheckBox')
                self.dd_PulsedRepump=on==2


        def dd_Tau_Step_lineEdit_textEdited(self,text):
                #print('done something with dd_Tau_Step_lineEdit. Text=',text)
                try:
                        self.dd_Tau_Step=eval(text.replace(",",".").replace("tt","self._tt").replace("self.self","self"))
                        self.parameter_to_sweep="duration"
                        self.sweep = np.unique(E.round_length_mus_to_x_multiple_ps(
                                np.arange(self.dd_Tau_Min, self.dd_Tau_Max + self.dd_Tau_Step, self.dd_Tau_Step) / 1000,self.round_to )) # in µs
                        
                        self.parameter_to_sweep="duration"
                        
                except:
                        pass

        def dd_A2_CheckBox_StateChanged(self,on):
                #print('done something with dd_A2_CheckBox')
                self.dd_A2=on==2


        def dd_AOMDelay_LineEdit_textEdited(self,text):
                #print('done something with dd_AOMDelay_LineEdit. Text=',text)
                try:
                        self.dd_AOMDelay=eval(text.replace(",",".").replace("tt","self._tt").replace("self.self","self"))
                except:
                        try:

                                self.sweep=np.arange(*parse_range(text))
                                self.parameter_to_sweep="dd_AOMDelay"
                        except:
                                pass

        def dd_Filename_LineEdit_textEdited(self,text):
                #print('done something with dd_Filename_LineEdit. Text=',text)
                try:
                        self.dd_Filename=text
                except:
                        pass

        def dd_ReadoutDecay_LineEdit_textEdited(self,text):
                #print('done something with dd_ReadoutDecay_LineEdit. Text=',text)
                try:
                        self.dd_ReadoutDecay=eval(text.replace(",",".").replace("tt","self._tt").replace("self.self","self"))
                except:
                        try:

                                self.sweep=np.arange(*parse_range(text))
                                self.parameter_to_sweep="dd_ReadoutDecay"
                        except:
                                pass


        def dd_MaxIterations_LineEdit_textEdited(self,text):
                #print('done something with dd_MaxIterations_LineEdit. Text=',text)
                try:
                        self.dd_MaxIterations=eval(text.replace(",",".").replace("tt","self._tt").replace("self.self","self"))
                except:
                        pass

        def dd_PerformFit_CheckBox_StateChanged(self,on):
                #print('done something with dd_PerformFit_CheckBox')
                self.dd_PerformFit=on==2
                self.update_after_stop=True #sorry magnificient programmer going throug this,      
                self.sigDDPlotsUpdated.emit()

        def dd_NumberOfPeaks_LineEdit_textEdited(self,text):
                #print('done something with dd_NumberOfPeaks_LineEdit. Text=',text)
                try:
                        self.dd_NumberOfPeaks=int(text.replace(",","."))
                except:
                        pass

        def dd_Binning_LineEdit_textEdited(self,text):
                #print('done something with dd_Binning_LineEdit. Text=',text)
                try:
                        self.dd_Binning=eval(text.replace(",",".").replace("tt","self._tt").replace("self.self","self"))
                except:
                        pass

        def dd_IntegrationTime_lineEdit_textEdited(self,text):
                #print('done something with dd_IntegrationTime_lineEdit. Text=',text)
                try:
                        dd_IntegrationTime=eval(text.replace(",",".").replace("tt","self._tt").replace("self.self","self"))
                        if dd_IntegrationTime>=2*self.dd_Binning:
                                self.dd_IntegrationTime=dd_IntegrationTime

                except:
                        pass

        def dd_SelectFit_ComboBox_currentTextChanged(self, text):
                print("combo box sagt: ", text)
                self.dd_FitFunction = text
       

        def dd_MW1_Power_LineEdit_textEdited(self,text):
                print('done something with dd_MW1_Power_LineEdit. Text=',text)
                try:
                        self.dd_MW1_Amp=float(text)
                except:
                        pass

        def dd_MW1_Freq_LineEdit_textEdited(self,text):
                print('done something with dd_MW1_Freq_LineEdit. Text=',text)
                try:
                        self.dd_MW1_Freq=float(text)
                except:
                        pass
        def dd_MW1_CheckBox_StateChanged(self,on):
                print('done something with dd_MW2_CheckBox')
                self.dd_MW1=on==2

        def dd_MW2_CheckBox_StateChanged(self,on):
                print('done something with dd_MW2_CheckBox')
                self.dd_MW2=on==2

        def dd_MW3_CheckBox_StateChanged(self,on):
                print('done something with dd_MW3_CheckBox')
                self.dd_MW3=on==2

        def dd_MW2_Power_LineEdit_textEdited(self,text):
                print('done something with dd_MW2_Power_LineEdit. Text=',text)
                try:
                        self.dd_MW2_Amp=eval(text.replace(",",".").replace("tt","self._tt").replace("self.self","self"))
                        
                except:
                        try:
                                self.sweep=np.arange(*parse_range(text))
                                self.parameter_to_sweep="dd_MW2_Amp"

                        except:
                                pass

        def dd_MW3_Power_LineEdit_textEdited(self,text):
                print('done something with dd_MW3_Power_LineEdit. Text=',text)
                try:
                        self.dd_MW3_Amp=eval(text.replace(",",".").replace("tt","self._tt").replace("self.self","self"))
                        
                except:
                        try:
                                self.sweep=np.arange(*parse_range(text))
                                self.parameter_to_sweep="dd_MW3_Amp"

                        except:
                                pass

        def dd_MW2_Freq_LineEdit_textEdited(self,text):
                print('done something with dd_MW2_Freq_LineEdit. Text=',text)
                try:
                        self.dd_MW2_Freq=eval(text.replace(",",".").replace("tt","self._tt").replace("self.self","self"))
                        
                except:
                        try:
                                self.sweep=np.arange(*parse_range(text))
                                self.parameter_to_sweep="dd_MW2_Freq"

                        except:
                                pass

        def dd_MW3_Freq_LineEdit_textEdited(self,text):
                print('done something with dd_MW3_Freq_LineEdit. Text=',text)
                try:
                        self.dd_MW3_Freq=eval(text.replace(",",".").replace("tt","self._tt").replace("self.self","self"))
                except:
                        try:

                                self.sweep=np.arange(*parse_range(text))
                                self.parameter_to_sweep="dd_MW3_Freq"
                        except:
                                pass

        def dd_MW4_Power_LineEdit_textEdited(self,text):
                print('done something with dd_MW4_Power_LineEdit. Text=',text)
                try:
                        self.dd_MW4_Amp=eval(text.replace(",",".").replace("tt","self._tt").replace("self.self","self"))
                except:
                        try:

                                self.sweep=np.arange(*parse_range(text))
                                self.parameter_to_sweep="dd_MW4_Amp"
                        except:
                                pass

        def dd_MW5_Power_LineEdit_textEdited(self,text):
                print('done something with dd_MW5_Power_LineEdit. Text=',text)
                try:
                        self.dd_MW5_Amp=eval(text.replace(",",".").replace("tt","self._tt").replace("self.self","self"))
                except:
                        try:
                                self.sweep=np.arange(*parse_range(text))
                                self.parameter_to_sweep="dd_MW5_Amp"
                        except:
                                pass

        def dd_MW6_Power_LineEdit_textEdited(self,text):
                print('done something with dd_MW6_Power_LineEdit. Text=',text)
                try:
                        self.dd_MW6_Amp=eval(text.replace(",",".").replace("tt","self._tt").replace("self.self","self"))
                except:
                        try:
                                print("going_into exception")
                                print(parse_range(text))
                                self.sweep=np.arange(*parse_range(text))
                                self.parameter_to_sweep="dd_MW6_Amp"
                        except:
                                pass

        def dd_MW4_Freq_LineEdit_textEdited(self,text):
                print('done something with dd_MW4_Freq_LineEdit. Text=',text)
                try:
                        self.dd_MW4_Freq=eval(text.replace(",",".").replace("tt","self._tt").replace("self.self","self"))
                except:
                        try:

                                self.sweep=np.arange(*parse_range(text))
                                self.parameter_to_sweep="dd_MW4_Freq"
                        except:
                                pass

        def dd_MW5_Freq_LineEdit_textEdited(self,text):
                print('done something with dd_MW5_Freq_LineEdit. Text=',text)
                try:
                        self.dd_MW5_Freq=eval(text.replace(",",".").replace("tt","self._tt").replace("self.self","self"))
                except:
                        try:

                                self.sweep=np.arange(*parse_range(text))
                                self.parameter_to_sweep="dd_MW5_Freq"
                        except:
                                pass

        def dd_MW6_Freq_LineEdit_textEdited(self,text):
                print('done something with dd_MW6_Freq_LineEdit. Text=',text)
                try:
                        self.dd_MW6_Freq=eval(text.replace(",",".").replace("tt","self._tt").replace("self.self","self"))
                except:
                        try:

                                self.sweep=np.arange(*parse_range(text))
                                self.parameter_to_sweep="dd_MW6_Freq"
                        except:
                                pass

        def dd_MW4_CheckBox_StateChanged(self,on):
                print('done something with dd_MW4_CheckBox')
                self.dd_MW4=on==2

        def dd_MW5_CheckBox_StateChanged(self,on):
                print('done something with dd_MW5_CheckBox')
                self.dd_MW5=on==2

        def dd_MW6_CheckBox_StateChanged(self,on):
                print('done something with dd_MW6_CheckBox')
                self.dd_MW6=on==2

        def dd_Pi_dur_LineEdit_textEdited(self,text):
                print('done something with dd_Pi_dur_LineEdit. Text=',text)
                try:
                        self.dd_Pi_dur=eval(text.replace(",",".").replace("tt","self._tt").replace("self.self","self"))
                except:
                        try:

                                self.sweep=np.arange(*parse_range(text))
                                self.parameter_to_sweep="dd_Pi_dur"
                        except:
                                pass

        def dd_Pi2_dur_LineEdit_textEdited(self,text):
                print('done something with dd_Pi2_dur_LineEdit. Text=',text)
                try:
                        self.dd_Pi2_dur=eval(text.replace(",",".").replace("tt","self._tt").replace("self.self","self"))
                except:
                        try:
                                self.sweep=np.arange(*parse_range(text))
                                self.parameter_to_sweep="dd_Pi2_dur"

                        except:
                                pass

        def dd_DD_type_comboBox_TextChanged(self,value):
                print('done something with dd_DD_type_comboBox. Value=',value)
                self.dd_DD_type=value

        def dd_nrep_LineEdit_textEdited(self,text):
                print('done something with dd_nrep_LineEdit. Text=',text)
                try:
                        self.dd_nrep=int(text.replace(",","."))
                except:
                        try:
                                self.sweep=np.arange(*parse_range(text))
                                self.parameter_to_sweep="dd_nrep"

                        except:
                                pass

        def dd_enhanced_readout_checkBox_StateChanged(self,on):
                print('done something with dd_enhanced_readout_checkBox')
                self.dd_enhanced_readout=on==2

        def dd_Init_State_comboBox_TextChanged(self,value):
                print('done something with dd_Init_State_comboBox. Value=',value)
                self.dd_Init_State=value

        def dd_MW1_Pi2_Power_LineEdit_textEdited(self,text):
                print('done something with dd_MW1_Pi2_Power_LineEdit. Text=',text)
                try:
                        self.dd_MW1_Pi2_Amp=float(text)
                except:
                        try:
                                self.sweep=np.arange(*parse_range(text))
                                self.parameter_to_sweep="dd_MW1_Pi2_Amp"

                        except:
                                pass

        def dd_MW2_Pi2_Power_LineEdit_textEdited(self,text):
                print('done something with dd_MW2_Pi2_Power_LineEdit. Text=',text)
                try:
                        self.dd_MW2_Pi2_Amp=float(text)
                except:
                        try:

                                self.sweep=np.arange(*parse_range(text))
                                self.parameter_to_sweep="dd_MW2_Pi2_Amp"
                        except:
                                pass

        def dd_MW3_Pi2_Power_LineEdit_textEdited(self,text):
                print('done something with dd_MW3_Pi2_Power_LineEdit. Text=',text)
                try:
                        self.dd_MW3_Pi2_Amp=float(text)
                except:
                        try:
                                self.sweep=np.arange(*parse_range(text))
                                self.parameter_to_sweep="dd_MW3_Pi2_Amp"

                        except:
                                pass

        def dd_MW4_Pi2_Power_LineEdit_textEdited(self,text):
                print('done something with dd_MW4_Pi2_Power_LineEdit. Text=',text)
                try:
                        self.dd_MW4_Pi2_Amp=float(text)
                except:
                        try:
                                self.sweep=np.arange(*parse_range(text))
                                self.parameter_to_sweep="dd_MW4_Pi2_Amp"

                        except:
                                pass

        def dd_MW5_Pi2_Power_LineEdit_textEdited(self,text):
                print('done something with dd_MW5_Pi2_Power_LineEdit. Text=',text)
                try:
                        self.dd_MW5_Pi2_Amp=float(text)
                except:
                        try:
                                self.sweep=np.arange(*parse_range(text))
                                self.parameter_to_sweep="dd_MW5_Pi2_Amp"

                        except:
                                pass

        def dd_MW6_Pi2_Power_LineEdit_textEdited(self,text):
                print('done something with dd_MW6_Pi2_Power_LineEdit. Text=',text)
                try:
                        self.dd_MW6_Pi2_Amp=float(text)
                except:
                        try:
                                self.sweep=np.arange(*parse_range(text))
                                self.parameter_to_sweep="dd_MW6_Pi2_Amp"
                        except:
                                pass

