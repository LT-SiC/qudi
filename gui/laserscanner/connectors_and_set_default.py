class initialize_connections_and_defaultvalue:
    def initialize_connections_and_defaultvalues(self):
        self._mw.ple_Load_Button.clicked.connect(self._voltscan_logic.ple_Load_Button_Clicked)
        self._mw.ple_PulsedRepump_CheckBox.stateChanged.connect(self._voltscan_logic.ple_PulsedRepump_CheckBox_StateChanged)
        self._mw.ple_MW2_CheckBox.stateChanged.connect(self._voltscan_logic.ple_MW2_CheckBox_StateChanged)
        self._mw.ple_Continue_Button.clicked.connect(self._voltscan_logic.ple_Continue_Button_Clicked)
        self._mw.ple_MW1_CheckBox.stateChanged.connect(self._voltscan_logic.ple_MW1_CheckBox_StateChanged)
        self._mw.ple_Save_Button.clicked.connect(self._voltscan_logic.ple_Save_Button_Clicked)
        self._mw.ple_MW1_Power_LineEdit.textEdited.connect(self._voltscan_logic.ple_MW1_Power_LineEdit_textEdited)
        self._mw.ple_Abort_Button.clicked.connect(self._voltscan_logic.ple_Abort_Button_Clicked)
        self._mw.ple_MW3_Power_LineEdit.textEdited.connect(self._voltscan_logic.ple_MW3_Power_LineEdit_textEdited)
        self._mw.ple_RepumpDuration_LineEdit.textEdited.connect(self._voltscan_logic.ple_RepumpDuration_LineEdit_textEdited)
        self._mw.ple_A1_CheckBox.stateChanged.connect(self._voltscan_logic.ple_A1_CheckBox_StateChanged)
        self._mw.ple_A2_CheckBox.stateChanged.connect(self._voltscan_logic.ple_A2_CheckBox_StateChanged)
        self._mw.ple_RepumpDecay_LineEdit.textEdited.connect(self._voltscan_logic.ple_RepumpDecay_LineEdit_textEdited)
        self._mw.ple_MW1_Freq_LineEdit.textEdited.connect(self._voltscan_logic.ple_MW1_Freq_LineEdit_textEdited)
        self._mw.ple_MW2_Freq_LineEdit.textEdited.connect(self._voltscan_logic.ple_MW2_Freq_LineEdit_textEdited)
        self._mw.ple_Run_Button.clicked.connect(self._voltscan_logic.ple_Run_Button_Clicked)
        self._mw.ple_MW3_CheckBox.stateChanged.connect(self._voltscan_logic.ple_MW3_CheckBox_StateChanged)
        self._mw.ple_MW2_Power_LineEdit.textEdited.connect(self._voltscan_logic.ple_MW2_Power_LineEdit_textEdited)
        self._mw.ple_NumberOfPeaks_LineEdit.textEdited.connect(self._voltscan_logic.ple_NumberOfPeaks_LineEdit_textEdited)
        self._mw.ple_Stop_Button.clicked.connect(self._voltscan_logic.ple_Stop_Button_Clicked)
        self._mw.ple_Filename_LineEdit.textEdited.connect(self._voltscan_logic.ple_Filename_LineEdit_textEdited)
        self._mw.ple_MW3_Freq_LineEdit.textEdited.connect(self._voltscan_logic.ple_MW3_Freq_LineEdit_textEdited)
        self._mw.ple_CWrepump_CheckBox.stateChanged.connect(self._voltscan_logic.ple_CWrepump_CheckBox_StateChanged)
        self._mw.ple_PeriodicSaving_CheckBox.stateChanged.connect(self._voltscan_logic.ple_PeriodicSaving_CheckBox_StateChanged)
        self._mw.ple_PerformFit_CheckBox.stateChanged.connect(self._voltscan_logic.ple_PerformFit_CheckBox_StateChanged)
        self._mw.ple_Interval_LineEdit.textEdited.connect(self._voltscan_logic.ple_Interval_LineEdit_textEdited)
        self._mw.ple_Stoptime_LineEdit.textEdited.connect(self._voltscan_logic.ple_Stoptime_LineEdit_textEdited)
        self._mw.startDoubleSpinBox.valueChanged.connect(self.change_start_volt)
        self._mw.speedDoubleSpinBox.valueChanged.connect(self.change_speed)
        self._mw.stopDoubleSpinBox.valueChanged.connect(self.change_stop_volt)
        self._mw.resolutionSpinBox.valueChanged.connect(self.change_resolution)
        self._mw.linesSpinBox.valueChanged.connect(self.change_lines)
        self._mw.constDoubleSpinBox.valueChanged.connect(self.change_voltage)
    
        self._mw.ple_A1_CheckBox.setChecked(self._voltscan_logic.enable_A1)
        self._mw.ple_A2_CheckBox.setChecked(self._voltscan_logic.enable_A2)
        self._mw.ple_CWrepump_CheckBox.setChecked(self._voltscan_logic.enable_Repump)
        self._mw.ple_PulsedRepump_CheckBox.setChecked(self._voltscan_logic.enable_PulsedRepump)
        self._mw.ple_MW1_CheckBox.setChecked(self._voltscan_logic.enable_MW1)
        self._mw.ple_MW2_CheckBox.setChecked(self._voltscan_logic.enable_MW2)
        self._mw.ple_MW3_CheckBox.setChecked(self._voltscan_logic.enable_MW3)
        self._mw.ple_MW1_Power_LineEdit.setText(str(self._voltscan_logic.MW1_Power))
        self._mw.ple_MW3_Power_LineEdit.setText(str(self._voltscan_logic.MW3_Power))
        self._mw.ple_RepumpDuration_LineEdit.setText(str(self._voltscan_logic.RepumpDuration))
        self._mw.ple_RepumpDecay_LineEdit.setText(str(self._voltscan_logic.RepumpDecay))
        self._mw.ple_MW1_Freq_LineEdit.setText(str(self._voltscan_logic.MW1_Freq))
        self._mw.ple_MW2_Freq_LineEdit.setText(str(self._voltscan_logic.MW2_Freq))
        self._mw.ple_MW2_Power_LineEdit.setText(str(self._voltscan_logic.MW2_Power))
        self._mw.ple_Filename_LineEdit.setText(str(self._voltscan_logic.Filename))
        self._mw.ple_MW3_Freq_LineEdit.setText(str(self._voltscan_logic.MW3_Freq))
        self._mw.ple_PeriodicSaving_CheckBox.setChecked(self._voltscan_logic.PeriodicSaving)
        self._mw.ple_Interval_LineEdit.setText(str(self._voltscan_logic.Interval))
        self._mw.ple_Stoptime_LineEdit.setText(str(self._voltscan_logic.Stoptime))
        self._mw.startDoubleSpinBox.setValue(self._voltscan_logic.scan_range[0])
        self._mw.speedDoubleSpinBox.setValue(self._voltscan_logic._scan_speed)
        self._mw.stopDoubleSpinBox.setValue(self._voltscan_logic.scan_range[1])
        self._mw.constDoubleSpinBox.setValue(self._voltscan_logic._static_v)
        self._mw.resolutionSpinBox.setValue(self._voltscan_logic.resolution)
        self._mw.linesSpinBox.setValue(self._voltscan_logic.number_of_repeats)
        self._mw.ple_NumberOfPeaks_LineEdit.setText(str(self._voltscan_logic.NumberOfPeaks))

        self._mw.ple_Contrast_Fit_Label.setText(str(self._voltscan_logic.Contrast_Fit))
        self._mw.ple_Frequencies_Fit_Label.setText(str(self._voltscan_logic.Frequencies_Fit))
        self._mw.ple_Linewidths_Fit_Label.setText(str(self._voltscan_logic.Linewidths_Fit))
        

    def disconnect_all(self):
        self._mw.ple_Load_Button.clicked.disconnect()
        self._mw.ple_PulsedRepump_CheckBox.stateChanged.disconnect()
        self._mw.ple_MW2_CheckBox.stateChanged.disconnect()
        self._mw.ple_Continue_Button.clicked.disconnect()
        self._mw.ple_MW1_CheckBox.stateChanged.disconnect()
        self._mw.ple_Save_Button.clicked.disconnect()
        self._mw.ple_MW1_Power_LineEdit.textEdited.disconnect()
        self._mw.ple_Abort_Button.clicked.disconnect()
        self._mw.ple_MW3_Power_LineEdit.textEdited.disconnect()
        self._mw.ple_RepumpDuration_LineEdit.textEdited.disconnect()
        self._mw.ple_A1_CheckBox.stateChanged.disconnect()
        self._mw.ple_A2_CheckBox.stateChanged.disconnect()
        self._mw.ple_RepumpDecay_LineEdit.textEdited.disconnect()
        self._mw.ple_MW1_Freq_LineEdit.textEdited.disconnect()
        self._mw.ple_MW2_Freq_LineEdit.textEdited.disconnect()
        self._mw.ple_Run_Button.clicked.disconnect()
        self._mw.ple_MW3_CheckBox.stateChanged.disconnect()
        self._mw.ple_MW2_Power_LineEdit.textEdited.disconnect()
        self._mw.ple_Stop_Button.clicked.disconnect()
        self._mw.ple_Filename_LineEdit.textEdited.disconnect()
        self._mw.ple_MW3_Freq_LineEdit.textEdited.disconnect()
        self._mw.ple_CWrepump_CheckBox.stateChanged.disconnect()
        self._mw.ple_PeriodicSaving_CheckBox.stateChanged.disconnect()
        self._mw.ple_PerformFit_CheckBox.stateChanged.disconnect()
        self._mw.ple_Interval_LineEdit.textEdited.disconnect()
        self._mw.ple_Stoptime_LineEdit.textEdited.disconnect()
        self._mw.ple_NumberOfPeaks_LineEdit.textEdited.disconnect()

        