import numpy as np
import time

class positioner_default_values_and_widget_functions:
        enable_S_x:bool=False
        enable_S_y:bool=False
        enable_S_z:bool=False
        enable_F_x:bool=False
        enable_F_y:bool=False
        enable_F_z:bool=False

        enable_cont_S_x:bool=False
        enable_cont_S_y:bool=False
        enable_cont_S_z:bool=False
        enable_cont_F_x:bool=False
        enable_cont_F_y:bool=False
        enable_cont_F_z:bool=False

        enable_cont_S_x_down:bool=False
        enable_cont_S_x_up:bool=False
        enable_cont_S_y_down:bool=False
        enable_cont_S_y_up:bool=False
        enable_cont_S_z_down:bool=False
        enable_cont_S_z_up:bool=False
        enable_cont_F_x_down:bool=False
        enable_cont_F_x_up:bool=False
        enable_cont_F_y_down:bool=False
        enable_cont_F_y_up:bool=False
        enable_cont_F_z_down:bool=False
        enable_cont_F_z_up:bool=False

        Voltage_reset = {"RT":30,"LT":50}
        Voltage:float = 30
        Voltage_2:float = 30
        Voltage_3:float = 30
        Voltage_4:float = 30
        Voltage_5:float = 30
        Voltage_6:float = 30

        Frequency_reset = {"RT":300,"LT":50}
        Frequency:float = 50
        Frequency_2:float = 50
        Frequency_3:float = 50
        Frequency_4:float = 50
        Frequency_5:float = 50
        Frequency_6:float = 50
        
        Capacitance:float = 0
        Capacitance_2:float = 0
        Capacitance_3:float = 0
        Capacitance_4:float = 0
        Capacitance_5:float = 0
        Capacitance_6:float = 0

        


        def Capacitance_Button_clicked(self, on):
                print('done something with Capacitance_Button')

        def T_Slider_changed(self, value):
                print('done something with T_Slider')
                T = "LT" if value == 1 else "RT"
                self.reset_values(T)

        def Voltage_doubleSpinBox_changed(self, value):
                print('done something with Voltage_doubleSpinBox')

        def Frequency_doubleSpinBox_changed(self, value):
                print('done something with Frequency_doubleSpinBox')

        def single_down_pushButton_clicked(self, on):
                print('done something with single_down_pushButton')

        def single_up_pushButton_clicked(self, on):
                print('done something with single_up_pushButton')

        def cont_down_pushButton_clicked(self, on):
                print('done something with cont_down_pushButton')
                self.enable_cont_S_x_down = on

        def cont_up_pushButton_clicked(self, on):
                print('done something with cont_up_pushButton')
                self.enable_cont_S_x_up = on

        