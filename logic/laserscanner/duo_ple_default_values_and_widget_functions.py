import numpy as np
import time

class ple_default_values_and_widget_functions:
        lock_laser:bool=False
        _lock_laser:bool=False
        RepumpDuration:float = 10
        RepumpDecay:float = 1000
        CTLDuration:float = 10
        CTLDecay:float = 1000
        BlueDuration:float = 10
        BlueDecay:float = 1000

        current_freq:str = ''
        # current_arc_factor:str = 'ARC Factor (V/V):'
        factor_V_to_GHz:float = 1
        plot_x:float = 1
        
        starttime:float = 0
        stoptime:float = 0
        PerformFit:bool = False

        Contrast_Fit:str=''
        Frequencies_Fit:str=''
        Linewidths_Fit:str=''


        # Laser 2:
        lock_laser_2:bool=False
        _lock_laser_2:bool=False
        RepumpDuration_2:float = 10
        RepumpDecay_2:float = 1000
        CTLDuration_2:float = 10
        CTLDecay_2:float = 1000
        BlueDuration_2:float = 10
        BlueDecay_2:float = 1000

        current_freq_2:str = ''
        # current_arc_factor_2:str = 'ARC Factor (V/V):'
        factor_V_to_GHz_2:float = 1
        plot_x_2:float = 1
        
        starttime_2:float = 0
        stoptime_2:float = 0
        PerformFit_2:bool = False

        Contrast_Fit_2:str=''
        Frequencies_Fit_2:str=''
        Linewidths_Fit_2:str=''




        # for CTL:

        #NumberOfLines: int = 10
        #ValuesPerScan: int = 200
        #Start: float = -3
        #Stop: float = 3
        #Speed: float = 0.3
        #constVoltage: float = 0


        def ple_Load_Button_Clicked(self,on):
                print('done something with ple_Load_Button')
        def ple_Load_Button_2_Clicked(self,on):
                print('done something with ple_Load_Button_2')
        
        def toggle_emission_Button_Clicked(self,on):
                self.toggle_emission()

        def automatic_calibration_checkBox_StateChanged(self,value):
                self.automatic_calibration=value==2

        # def ple_locking_checkBox_StateChanged(self,value):
        #         self.ple_locked=value==2

        def factor_V_to_GHz_pushButton_Clicked(self,on):
                print('done something with factor_V_to_GHz_pushButton')

        def ple_PulsedRepump_CheckBox_StateChanged(self,value):
                self.enable_PulsedRepump=value==2
        def ple_PulsedRepump_CheckBox_2_StateChanged(self,value):
                self.enable_PulsedRepump_2=value==2

        def ple_PulsedCTL_CheckBox_StateChanged(self,value):
                self.enable_PulsedCTL=value==2

        def ple_PulsedBlue_CheckBox_StateChanged(self,value):
                self.enable_PulsedBlue=value==2

        def ple_RepumpWhenIonized_CheckBox_StateChanged(self,value):
                self.RepumpWhenIonized=value==2

        def ple_MW2_CheckBox_StateChanged(self,value):
                self.enable_MW2=value==2

        def ple_Continue_Button_Clicked(self,on):
                print('done something with ple_Continue_Button')

        def ple_MW1_CheckBox_StateChanged(self,value):
                self.enable_MW1=value==2

        def ple_Save_Button_Clicked(self,on):
                self.save_data(tag=self.Filename)

        def ple_MW1_Power_LineEdit_textEdited(self,text):
                try:
                        self.MW1_Power=float(text.replace(",","."))
                except:
                        pass

        def ple_MW3_Power_LineEdit_textEdited(self,text):
                try:
                        self.MW3_Power=float(text.replace(",","."))
                except:
                        pass

        def ple_RepumpDuration_LineEdit_textEdited(self,text):
                try:
                        self.RepumpDuration=float(text.replace(",","."))
                except:
                        pass

        def ple_CTLDuration_LineEdit_textEdited(self,text):
                try:
                        self.CTLDuration=float(text.replace(",","."))
                except:
                        pass

        def ple_BlueDuration_LineEdit_textEdited(self,text):
                try:
                        self.BlueDuration=float(text.replace(",","."))
                except:
                        pass
        
        def ple_NumberOfPeaks_LineEdit_textEdited(self,text):
                try:
                        self.NumberOfPeaks=int(text)
                except:
                        pass

        def ple_A1_CheckBox_StateChanged(self,value):
                self.enable_A1=value==2

        def ple_A2_CheckBox_StateChanged(self,value):
                self.enable_A2=value==2

        def ple_Lock_Laser_CheckBox_StateChanged(self,value):
                self.lock_laser=value==2

        # def ple_RepumpDecay_LineEdit_textEdited(self,text):
        #         try:
        #                 self.RepumpDecay=float(text.replace(",","."))
        #         except:
        #                 pass

        def ple_MW1_Freq_LineEdit_textEdited(self,text):
                try:
                        self.MW1_Freq=float(text.replace(",","."))
                except:
                        pass

        def ple_MW2_Freq_LineEdit_textEdited(self,text):
                try:
                        self.MW2_Freq=float(text.replace(",","."))
                except:
                        pass

        def ple_Run_Button_Clicked(self,on):
                self.start_scanning()

        def ple_MW3_CheckBox_StateChanged(self,value):
                self.enable_MW3=value==2

        def ple_MW2_Power_LineEdit_textEdited(self,text):
                try:
                        self.MW2_Power=float(text.replace(",","."))
                except:
                        pass

        def ple_Stop_Button_Clicked(self,on):
                self.stopRequested = True

        def ple_Filename_LineEdit_textEdited(self,text):
                try:
                        self.Filename=text
                except:
                        pass

        def ple_MW3_Freq_LineEdit_textEdited(self,text):
                try:
                        self.MW3_Freq=float(text.replace(",","."))
                except:
                        pass

        def ple_CWrepump_CheckBox_StateChanged(self,value):
                self.enable_Repump=value==2

        def ple_CWCTL_CheckBox_StateChanged(self,value):
                self.enable_CTL=value==2

        def ple_CWBlue_CheckBox_StateChanged(self,value):
                self.enable_Blue=value==2

        def ple_PerformFit_CheckBox_StateChanged(self,value):
                self.PerformFit=value==2
                # if self.PerformFit:
                self.sigUpdatePlots.emit(1)
   
        def ple_MaxIterations_LineEdit_textEdited(self,text):
                try:
                        self.MaxIterations=float(text.replace(",","."))
                except:
                        pass

        def ple_Stoptime_LineEdit_textEdited(self,text):
                try:
                        self.stoptime=float(text.replace(",","."))
                except:
                        pass

        def factor_V_to_GHz_LineEdit_textEdited(self,text):
                try:
                        self.factor_V_to_GHz=float(text.replace(",","."))
                        self.plot_x_frequency=self.plot_x*self.factor_V_to_GHz
                except:
                        pass

        def fit_methods_ComboBox_StateChanged(self):
                print("Selected new Fitmethod Need to connect the data via ,,initialize_connections_and_defaultvalues,, of the gui. In line 61 or so.")

        def select_scan_range(self, index):
                print(index)
                print("Selected new Fitmethod Need to connect the data via ,,initialize_connections_and_defaultvalues,, of the gui. In line 61 or so.")
        
        
        
        
        
        
        # Laser 2:
        def ple_Load_Button_2_Clicked(self,on):
                print('done something with ple_Load_Button_2')
        
        def toggle_emission_Button_2_Clicked(self,on):
                self.toggle_emission_2()

        def automatic_calibration_checkBox_2_StateChanged(self,value):
                self.automatic_calibration_2=value==2

        def factor_V_to_GHz_pushButton_2_Clicked(self,on):
                print('done something with factor_V_to_GHz_pushButton_2')

        def ple_PulsedRepump_CheckBox_2_StateChanged(self,value):
                self.enable_PulsedRepump_2=value==2
                
        def ple_PulsedRepump_CheckBox_2_StateChanged(self,value):
                self.enable_PulsedRepump_2=value==2

        def ple_PulsedCTL_CheckBox_2_StateChanged(self,value):
                self.enable_PulsedCTL_2=value==2

        def ple_PulsedBlue_CheckBox_2_StateChanged(self,value):
                self.enable_PulsedBlue_2=value==2

        def ple_RepumpWhenIonized_CheckBox_2_StateChanged(self,value):
                self.RepumpWhenIonized_2=value==2

        def ple_MW2_CheckBox_2_StateChanged(self,value):
                self.enable_MW2_2=value==2

        def ple_Continue_Button_2_Clicked(self,on):
                print('done something with ple_Continue_Button_2')

        def ple_MW1_CheckBox_2_StateChanged(self,value):
                self.enable_MW1_2=value==2

        def ple_Save_Button_2_Clicked(self,on):
                self.save_data_2(tag=self.Filename_2)

        def ple_MW1_Power_LineEdit_2_textEdited(self,text):
                try:
                        self.MW1_Power_2=float(text.replace(",","."))
                except:
                        pass

        def ple_MW3_Power_LineEdit_2_textEdited(self,text):
                try:
                        self.MW3_Power_2=float(text.replace(",","."))
                except:
                        pass

        def ple_RepumpDuration_LineEdit_2_textEdited(self,text):
                try:
                        self.RepumpDuration_2=float(text.replace(",","."))
                except:
                        pass

        def ple_CTLDuration_LineEdit_2_textEdited(self,text):
                try:
                        self.CTLDuration_2=float(text.replace(",","."))
                except:
                        pass

        def ple_BlueDuration_LineEdit_2_textEdited(self,text):
                try:
                        self.BlueDuration_2=float(text.replace(",","."))
                except:
                        pass
        
        def ple_NumberOfPeaks_LineEdit_2_textEdited(self,text):
                try:
                        self.NumberOfPeaks_2=int(text)
                except:
                        pass

        def ple_A1_CheckBox_2_StateChanged(self,value):
                self.enable_A1_2=value==2

        def ple_A2_CheckBox_2_StateChanged(self,value):
                self.enable_A2_2=value==2

        def ple_Lock_Laser_CheckBox_2_StateChanged(self,value):
                self.lock_laser_2=value==2

        # def ple_RepumpDecay_LineEdit_2_textEdited(self,text):
        #         try:
        #                 self.RepumpDecay_2=float(text.replace(",","."))
        #         except:
        #                 pass

        def ple_MW1_Freq_LineEdit_2_textEdited(self,text):
                try:
                        self.MW1_Freq_2=float(text.replace(",","."))
                except:
                        pass

        def ple_MW2_Freq_LineEdit_2_textEdited(self,text):
                try:
                        self.MW2_Freq_2=float(text.replace(",","."))
                except:
                        pass

        def ple_Run_Button_2_Clicked(self,on):
                self.start_scanning_2()

        def ple_MW3_CheckBox_2_StateChanged(self,value):
                self.enable_MW3_2=value==2

        def ple_MW2_Power_LineEdit_2_textEdited(self,text):
                try:
                        self.MW2_Power_2=float(text.replace(",","."))
                except:
                        pass

        def ple_Stop_Button_2_Clicked(self,on):
                self.stopRequested_2 = True

        def ple_Filename_LineEdit_2_textEdited(self,text):
                try:
                        self.Filename_2=text
                except:
                        pass

        def ple_MW3_Freq_LineEdit_2_textEdited(self,text):
                try:
                        self.MW3_Freq_2=float(text.replace(",","."))
                except:
                        pass

        def ple_CWrepump_CheckBox_2_StateChanged(self,value):
                self.enable_Repump_2=value==2

        def ple_CWCTL_CheckBox_2_StateChanged(self,value):
                self.enable_CTL_2=value==2

        def ple_CWBlue_CheckBox_2_StateChanged(self,value):
                self.enable_Blue_2=value==2

        def ple_PerformFit_CheckBox_2_StateChanged(self,value):
                self.PerformFit_2=value==2
                # if self.PerformFit_2:
                self.sigUpdatePlots.emit(2)
   
        def ple_MaxIterations_LineEdit_2_textEdited(self,text):
                try:
                        self.MaxIterations_2=float(text.replace(",","."))
                except:
                        pass

        def ple_Stoptime_LineEdit_2_textEdited(self,text):
                try:
                        self.stoptime_2=float(text.replace(",","."))
                except:
                        pass

        def factor_V_to_GHz_LineEdit_2_textEdited(self,text):
                try:
                        self.factor_V_to_GHz_2=float(text.replace(",","."))
                        self.plot_x_frequency_2=self.plot_x_2*self.factor_V_to_GHz_2
                except:
                        pass

        def fit_methods_ComboBox_2_StateChanged(self):
                print("Selected new Fitmethod Need to connect the data via ,,initialize_connections_and_defaultvalues,, of the gui. In line 61 or so.")

        def select_scan_range_2(self, index):
                print(index)
                print("Selected new Fitmethod Need to connect the data via ,,initialize_connections_and_defaultvalues,, of the gui. In line 61 or so.")