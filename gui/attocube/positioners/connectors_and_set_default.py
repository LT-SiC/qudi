class initialize_connections_and_defaultvalue:
    def initialize_connections_and_defaultvalues(self):
        self._mw.axis_enabled_checkBox.stateChanged.connect(self.axis_enabled_checkBox_changed)
        self._mw.Capacitance_Button.clicked.connect(self._positioncontrollogic.Capacitance_Button_clicked)
        self._mw.T_Slider.valueChanged.connect(self._positioncontrollogic.T_Slider_changed)
        self._mw.Voltage_doubleSpinBox.valueChanged.connect(self._positioncontrollogic.Voltage_doubleSpinBox_changed)
        self._mw.Frequency_doubleSpinBox.valueChanged.connect(self._positioncontrollogic.Frequency_doubleSpinBox_changed)
        self._mw.single_down_pushButton.clicked.connect(self._positioncontrollogic.single_down_pushButton_clicked)
        self._mw.single_up_pushButton.clicked.connect(self._positioncontrollogic.single_up_pushButton_clicked)
        self._mw.single_radioButton.clicked.connect(self.single_radioButton_clicked)
        self._mw.cont_down_pushButton.clicked.connect(self._positioncontrollogic.cont_down_pushButton_clicked)
        self._mw.cont_up_pushButton.clicked.connect(self._positioncontrollogic.cont_up_pushButton_clicked)
        self._mw.cont_radioButton.clicked.connect(self.cont_radioButton_clicked)

        self._mw.axis_enabled_checkBox_2.stateChanged.connect(self.axis_enabled_checkBox_2_changed)
        self._mw.Capacitance_Button_2.clicked.connect(self._positioncontrollogic.Capacitance_Button_2_clicked)
        self._mw.T_Slider_2.valueChanged.connect(self._positioncontrollogic.T_Slider_changed)
        self._mw.Voltage_doubleSpinBox_2.valueChanged.connect(self._positioncontrollogic.Voltage_doubleSpinBox_2_changed)
        self._mw.Frequency_doubleSpinBox_2.valueChanged.connect(self._positioncontrollogic.Frequency_doubleSpinBox_2_changed)
        self._mw.single_down_pushButton_2.clicked.connect(self._positioncontrollogic.single_down_pushButton_2_clicked)
        self._mw.single_up_pushButton_2.clicked.connect(self._positioncontrollogic.single_up_pushButton_2_clicked)
        self._mw.single_radioButton_2.clicked.connect(self.single_radioButton_2_clicked)
        self._mw.cont_down_pushButton_2.clicked.connect(self._positioncontrollogic.cont_down_pushButton_2_clicked)
        self._mw.cont_up_pushButton_2.clicked.connect(self._positioncontrollogic.cont_up_pushButton_2_clicked)
        self._mw.cont_radioButton_2.clicked.connect(self.cont_radioButton_2_clicked)

        self._mw.axis_enabled_checkBox_3.stateChanged.connect(self.axis_enabled_checkBox_3_changed)
        self._mw.Capacitance_Button_3.clicked.connect(self._positioncontrollogic.Capacitance_Button_3_clicked)
        self._mw.T_Slider_3.valueChanged.connect(self._positioncontrollogic.T_Slider_changed)
        self._mw.Voltage_doubleSpinBox_3.valueChanged.connect(self._positioncontrollogic.Voltage_doubleSpinBox_3_changed)
        self._mw.Frequency_doubleSpinBox_3.valueChanged.connect(self._positioncontrollogic.Frequency_doubleSpinBox_3_changed)
        self._mw.single_down_pushButton_3.clicked.connect(self._positioncontrollogic.single_down_pushButton_3_clicked)
        self._mw.single_up_pushButton_3.clicked.connect(self._positioncontrollogic.single_up_pushButton_3_clicked)
        self._mw.single_radioButton_3.clicked.connect(self.single_radioButton_3_clicked)
        self._mw.cont_down_pushButton_3.clicked.connect(self._positioncontrollogic.cont_down_pushButton_3_clicked)
        self._mw.cont_up_pushButton_3.clicked.connect(self._positioncontrollogic.cont_up_pushButton_3_clicked)
        self._mw.cont_radioButton_3.clicked.connect(self.cont_radioButton_3_clicked)

        self._mw.axis_enabled_checkBox_4.stateChanged.connect(self.axis_enabled_checkBox_4_changed)
        self._mw.Capacitance_Button_4.clicked.connect(self._positioncontrollogic.Capacitance_Button_4_clicked)
        self._mw.T_Slider_4.valueChanged.connect(self._positioncontrollogic.T_Slider_changed)
        self._mw.Voltage_doubleSpinBox_4.valueChanged.connect(self._positioncontrollogic.Voltage_doubleSpinBox_4_changed)
        self._mw.Frequency_doubleSpinBox_4.valueChanged.connect(self._positioncontrollogic.Frequency_doubleSpinBox_4_changed)
        self._mw.single_down_pushButton_4.clicked.connect(self._positioncontrollogic.single_down_pushButton_4_clicked)
        self._mw.single_up_pushButton_4.clicked.connect(self._positioncontrollogic.single_up_pushButton_4_clicked)
        self._mw.single_radioButton_4.clicked.connect(self.single_radioButton_4_clicked)
        self._mw.cont_down_pushButton_4.clicked.connect(self._positioncontrollogic.cont_down_pushButton_4_clicked)
        self._mw.cont_up_pushButton_4.clicked.connect(self._positioncontrollogic.cont_up_pushButton_4_clicked)
        self._mw.cont_radioButton_4.clicked.connect(self.cont_radioButton_4_clicked)

        self._mw.axis_enabled_checkBox_5.stateChanged.connect(self.axis_enabled_checkBox_5_changed)
        self._mw.Capacitance_Button_5.clicked.connect(self._positioncontrollogic.Capacitance_Button_5_clicked)
        self._mw.T_Slider_5.valueChanged.connect(self._positioncontrollogic.T_Slider_changed)
        self._mw.Voltage_doubleSpinBox_5.valueChanged.connect(self._positioncontrollogic.Voltage_doubleSpinBox_5_changed)
        self._mw.Frequency_doubleSpinBox_5.valueChanged.connect(self._positioncontrollogic.Frequency_doubleSpinBox_5_changed)
        self._mw.single_down_pushButton_5.clicked.connect(self._positioncontrollogic.single_down_pushButton_5_clicked)
        self._mw.single_up_pushButton_5.clicked.connect(self._positioncontrollogic.single_up_pushButton_5_clicked)
        self._mw.single_radioButton_5.clicked.connect(self.single_radioButton_5_clicked)
        self._mw.cont_down_pushButton_5.clicked.connect(self._positioncontrollogic.cont_down_pushButton_5_clicked)
        self._mw.cont_up_pushButton_5.clicked.connect(self._positioncontrollogic.cont_up_pushButton_5_clicked)
        self._mw.cont_radioButton_5.clicked.connect(self.cont_radioButton_5_clicked)

        self._mw.axis_enabled_checkBox_6.stateChanged.connect(self.axis_enabled_checkBox_6_changed)
        self._mw.Capacitance_Button_6.clicked.connect(self._positioncontrollogic.Capacitance_Button_6_clicked)
        self._mw.T_Slider_6.valueChanged.connect(self._positioncontrollogic.T_Slider_changed)
        self._mw.Voltage_doubleSpinBox_6.valueChanged.connect(self._positioncontrollogic.Voltage_doubleSpinBox_6_changed)
        self._mw.Frequency_doubleSpinBox_6.valueChanged.connect(self._positioncontrollogic.Frequency_doubleSpinBox_6_changed)
        self._mw.single_down_pushButton_6.clicked.connect(self._positioncontrollogic.single_down_pushButton_6_clicked)
        self._mw.single_up_pushButton_6.clicked.connect(self._positioncontrollogic.single_up_pushButton_6_clicked)
        self._mw.single_radioButton_6.clicked.connect(self.single_radioButton_6_clicked)
        self._mw.cont_down_pushButton_6.clicked.connect(self._positioncontrollogic.cont_down_pushButton_6_clicked)
        self._mw.cont_up_pushButton_6.clicked.connect(self._positioncontrollogic.cont_up_pushButton_6_clicked)
        self._mw.cont_radioButton_6.clicked.connect(self.cont_radioButton_6_clicked)

        self.axis_enabled_checkBox_list = [
            self._mw.axis_enabled_checkBox,
            self._mw.axis_enabled_checkBox_2,
            self._mw.axis_enabled_checkBox_3,
            self._mw.axis_enabled_checkBox_4,
            self._mw.axis_enabled_checkBox_5,
            self._mw.axis_enabled_checkBox_6
        ]

        self.Capacitance_Button_list = [
            self._mw.Capacitance_Button,
            self._mw.Capacitance_Button_2,
            self._mw.Capacitance_Button_3,
            self._mw.Capacitance_Button_4,
            self._mw.Capacitance_Button_5,
            self._mw.Capacitance_Button_6
        ]

        self.T_Slider_list = [
            self._mw.T_Slider,
            self._mw.T_Slider_2,
            self._mw.T_Slider_3,
            self._mw.T_Slider_4,
            self._mw.T_Slider_5,
            self._mw.T_Slider_6
        ]

        self.Voltage_doubleSpinBox_list = [
            self._mw.Voltage_doubleSpinBox,
            self._mw.Voltage_doubleSpinBox_2,
            self._mw.Voltage_doubleSpinBox_3,
            self._mw.Voltage_doubleSpinBox_4,
            self._mw.Voltage_doubleSpinBox_5,
            self._mw.Voltage_doubleSpinBox_6
        ]

        self.Frequency_doubleSpinBox_list = [
            self._mw.Frequency_doubleSpinBox,
            self._mw.Frequency_doubleSpinBox_2,
            self._mw.Frequency_doubleSpinBox_3,
            self._mw.Frequency_doubleSpinBox_4,
            self._mw.Frequency_doubleSpinBox_5,
            self._mw.Frequency_doubleSpinBox_6
        ]

        self.single_down_pushButton_list = [
            self._mw.single_down_pushButton,
            self._mw.single_down_pushButton_2,
            self._mw.single_down_pushButton_3,
            self._mw.single_down_pushButton_4,
            self._mw.single_down_pushButton_5,
            self._mw.single_down_pushButton_6
        ]

        self.single_up_pushButton_list = [
            self._mw.single_up_pushButton,
            self._mw.single_up_pushButton_2,
            self._mw.single_up_pushButton_3,
            self._mw.single_up_pushButton_4,
            self._mw.single_up_pushButton_5,
            self._mw.single_up_pushButton_6
        ]

        self.single_radioButton_list = [
            self._mw.single_radioButton,
            self._mw.single_radioButton_2,
            self._mw.single_radioButton_3,
            self._mw.single_radioButton_4,
            self._mw.single_radioButton_5,
            self._mw.single_radioButton_6
        ]

        self.cont_down_pushButton_list = [
            self._mw.cont_down_pushButton,
            self._mw.cont_down_pushButton_2,
            self._mw.cont_down_pushButton_3,
            self._mw.cont_down_pushButton_4,
            self._mw.cont_down_pushButton_5,
            self._mw.cont_down_pushButton_6
        ]

        self.cont_up_pushButton_list = [
            self._mw.cont_up_pushButton,
            self._mw.cont_up_pushButton_2,
            self._mw.cont_up_pushButton_3,
            self._mw.cont_up_pushButton_4,
            self._mw.cont_up_pushButton_5,
            self._mw.cont_up_pushButton_6
        ]

        self.cont_radioButton_list = [
            self._mw.cont_radioButton,
            self._mw.cont_radioButton_2,
            self._mw.cont_radioButton_3,
            self._mw.cont_radioButton_4,
            self._mw.cont_radioButton_5,
            self._mw.cont_radioButton_6
        ]

        self.lcdNumber_list = [
            self._mw.lcdNumber,
            self._mw.lcdNumber_2,
            self._mw.lcdNumber_3,
            self._mw.lcdNumber_4,
            self._mw.lcdNumber_5,
            self._mw.lcdNumber_6
        ]

        self.gui_elements_list = {
            "axis":self.axis_enabled_checkBox_list,
            "C":self.Capacitance_Button_list,
            "T":self.T_Slider_list,
            "F":self.Frequency_doubleSpinBox_list,
            "V":self.Voltage_doubleSpinBox_list,
            "singD":self.single_down_pushButton_list,
            "singUp":self.single_up_pushButton_list,
            "sing":self.single_radioButton_list,
            "conD":self.cont_down_pushButton_list,
            "conUp":self.cont_up_pushButton_list,
            "con":self.cont_radioButton_list
        }




    def disconnect_all(self):
        for checkBox in self.gui_elements_list["axis"]:
            checkBox.stateChanged.disconnect()
        for Button in self.gui_elements_list["C"]:
            Button.clicked.disconnect()
        for Slider in self.gui_elements_list["T"]:
            Slider.valueChanged.disconnect()
        for SpinBox in self.gui_elements_list["F"]:
            SpinBox.valueChanged.disconnect()
        for SpinBox in self.gui_elements_list["V"]:
            SpinBox.valueChanged.disconnect()
        for Button in self.gui_elements_list["singD"]:
            Button.clicked.disconnect()
        for Button in self.gui_elements_list["singUp"]:
            Button.clicked.disconnect()
        for Button in self.gui_elements_list["sing"]:
            Button.clicked.disconnect()
        for Button in self.gui_elements_list["conD"]:
            Button.clicked.disconnect()
        for Button in self.gui_elements_list["conUp"]:
            Button.clicked.disconnect()
        for Button in self.gui_elements_list["con"]:
            Button.clicked.disconnect()

        # self._mw.axis_enabled_checkBox.stateChanged.disconnect()
        # self._mw.Capacitance_Button.clicked.disconnect()
        # self._mw.T_Slider.valueChanged.disconnect()
        # self._mw.Voltage_doubleSpinBox.valueChanged.disconnect()
        # self._mw.Frequency_doubleSpinBox.valueChanged.disconnect()
        # self._mw.single_down_pushButton.clicked.disconnect()
        # self._mw.single_up_pushButton.clicked.disconnect()
        # self._mw.single_radioButton.clicked.disconnect()
        # self._mw.cont_down_pushButton.clicked.disconnect()
        # self._mw.cont_up_pushButton.clicked.disconnect()
        # self._mw.cont_radioButton.clicked.disconnect()

        # self._mw.axis_enabled_checkBox_2.stateChanged.disconnect()
        # self._mw.Capacitance_Button_2.clicked.disconnect()
        # self._mw.T_Slider.valueChanged_2.disconnect()
        # self._mw.Voltage_doubleSpinBox_2.valueChanged.disconnect()
        # self._mw.Frequency_doubleSpinBox_2.valueChanged.disconnect()
        # self._mw.single_down_pushButton_2.clicked.disconnect()
        # self._mw.single_up_pushButton_2.clicked.disconnect()
        # self._mw.single_radioButton_2.clicked.disconnect()
        # self._mw.cont_down_pushButton_2.clicked.disconnect()
        # self._mw.cont_up_pushButton_2.clicked.disconnect()
        # self._mw.cont_radioButton_2.clicked.disconnect()

        # self._mw.axis_enabled_checkBox_3.stateChanged.disconnect()
        # self._mw.Capacitance_Button_3.clicked.disconnect()
        # self._mw.T_Slider.valueChanged_3.disconnect()
        # self._mw.Voltage_doubleSpinBox_3.valueChanged.disconnect()
        # self._mw.Frequency_doubleSpinBox_3.valueChanged.disconnect()
        # self._mw.single_down_pushButton_3.clicked.disconnect()
        # self._mw.single_up_pushButton_3.clicked.disconnect()
        # self._mw.single_radioButton_3.clicked.disconnect()
        # self._mw.cont_down_pushButton_3.clicked.disconnect()
        # self._mw.cont_up_pushButton_3.clicked.disconnect()
        # self._mw.cont_radioButton_3.clicked.disconnect()

        # self._mw.axis_enabled_checkBox_4.stateChanged.disconnect()
        # self._mw.Capacitance_Button_4.clicked.disconnect()
        # self._mw.T_Slider.valueChanged_4.disconnect()
        # self._mw.Voltage_doubleSpinBox_4.valueChanged.disconnect()
        # self._mw.Frequency_doubleSpinBox_4.valueChanged.disconnect()
        # self._mw.single_down_pushButton_4.clicked.disconnect()
        # self._mw.single_up_pushButton_4.clicked.disconnect()
        # self._mw.single_radioButton_4.clicked.disconnect()
        # self._mw.cont_down_pushButton_4.clicked.disconnect()
        # self._mw.cont_up_pushButton_4.clicked.disconnect()
        # self._mw.cont_radioButton_4.clicked.disconnect()

        # self._mw.axis_enabled_checkBox_5.stateChanged.disconnect()
        # self._mw.Capacitance_Button_5.clicked.disconnect()
        # self._mw.T_Slider.valueChanged_5.disconnect()
        # self._mw.Voltage_doubleSpinBox_5.valueChanged.disconnect()
        # self._mw.Frequency_doubleSpinBox_5.valueChanged.disconnect()
        # self._mw.single_down_pushButton_5.clicked.disconnect()
        # self._mw.single_up_pushButton_5.clicked.disconnect()
        # self._mw.single_radioButton_5.clicked.disconnect()
        # self._mw.cont_down_pushButton_5.clicked.disconnect()
        # self._mw.cont_up_pushButton_5.clicked.disconnect()
        # self._mw.cont_radioButton_5.clicked.disconnect()

        # self._mw.axis_enabled_checkBox_6.stateChanged.disconnect()
        # self._mw.Capacitance_Button_6.clicked.disconnect()
        # self._mw.T_Slider.valueChanged_6.disconnect()
        # self._mw.Voltage_doubleSpinBox_6.valueChanged.disconnect()
        # self._mw.Frequency_doubleSpinBox_6.valueChanged.disconnect()
        # self._mw.single_down_pushButton_6.clicked.disconnect()
        # self._mw.single_up_pushButton_6.clicked.disconnect()
        # self._mw.single_radioButton_6.clicked.disconnect()
        # self._mw.cont_down_pushButton_6.clicked.disconnect()
        # self._mw.cont_up_pushButton_6.clicked.disconnect()
        # self._mw.cont_radioButton_6.clicked.disconnect()
        