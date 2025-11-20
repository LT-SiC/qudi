class initialize_connections_and_defaultvalue:
    def initialize_connections_and_defaultvalues(self):
         # GUI-Events
        self._mw.thresholdSpinBox.valueChanged.connect(self.update_filter_params)
        self._mw.nFilterSpinBox.valueChanged.connect(self.update_filter_params)
        self._mw.binwidth_spinBox.valueChanged.connect(self.update_collection_params)
        self._mw.datapoints_spinBox.valueChanged.connect(self.update_collection_params)
        self._mw.window_size_spinBox.valueChanged.connect(self.update_plot_params)

        # Initialwerte setzen
        self._mw.thresholdSpinBox.setValue(self.logic.threshold)
        self._mw.nFilterSpinBox.setValue(self.logic.n_filter)
        self._mw.binwidth_spinBox.setValue(self.logic.binwidth/1e3)
        self._mw.datapoints_spinBox.setValue(self.logic.n_values)
        self._mw.window_size_spinBox.setValue(self.logic.plot_length)

        self._mw.refilterButton.clicked.connect(self.refilter)
        self._mw.saveButton.clicked.connect(self.save_histogram)
        self._mw.startButton.clicked.connect(self.start_measurement)
        self._mw.stopButton.clicked.connect(self.stop_measurement)
        self._mw.resubmitButton.clicked.connect(self.resubmit_measurement)
        self._mw.raw_data_checkBox.stateChanged.connect(self._update_hist_plot)

        self._mw.Filename_LineEdit.setText(str(self.logic.Filename))


    def disconnect_all(self):
        self._mw.thresholdSpinBox.valueChanged.disconnect()
        self._mw.nFilterSpinBox.valueChanged.disconnect()
        self._mw.binwidth_spinBox.valueChanged.disconnect()
        self._mw.datapoints_spinBox.valueChanged.disconnect()
        self._mw.window_size_spinBox.valueChanged.disconnect()
        self._mw.refilterButton.clicked.disconnect()
        self._mw.saveButton.clicked.disconnect()
        self._mw.startButton.clicked.disconnect()
        self._mw.stopButton.clicked.disconnect()
        self._mw.resubmitButton.clicked.disconnect()
        self._mw.raw_data_checkBox.stateChanged.disconnect()

        # self._mw.ple_Load_Button.clicked.disconnect()
        # self._mw.ple_PulsedRepump_CheckBox.stateChanged.disconnect()
        # self._mw.ple_MW1_Power_LineEdit.textEdited.disconnect()